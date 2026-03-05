"""
rl_agent.py
===========
Reinforcement learning agent using Stable Baselines3.

Supported algorithms
--------------------
  PPO  – Proximal Policy Optimisation (default, best stability)
  DQN  – Deep Q-Network (off-policy, sample-efficient)
  A2C  – Advantage Actor-Critic (fast, lower memory)

The agent wraps an SB3 model and adds:
  • checkpoint save/load helpers
  • action-logging for debugging
  • a custom MLP policy that accepts the goal-conditioned observation
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.pokemon_blue_env import PokemonBlueEnv

logger = logging.getLogger(__name__)

AlgoType = Literal["PPO", "DQN", "A2C"]
ALGO_MAP = {"PPO": PPO, "DQN": DQN, "A2C": A2C}


# ---------------------------------------------------------------------------
# Custom SB3 callback: log episode stats
# ---------------------------------------------------------------------------

class EpisodeStatsCallback(BaseCallback):
    """
    Logs per-episode statistics (reward, badges, Pokédex, etc.)
    to the console and optionally to a CSV file.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir) if log_dir else None
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = self.log_dir / "episode_stats.csv"
            with open(self._csv_path, "w") as f:
                f.write("episode,reward,length,badges,pokedex,maps,coords,goal\n")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                badges = info.get("badges", 0)
                dex    = info.get("pokedex_caught", 0)
                maps   = info.get("maps_visited", 0)
                coords = info.get("coords_visited", 0)
                goal   = info.get("goal", "?")
                r      = ep["r"]
                l      = ep["l"]
                self._episode_rewards.append(r)
                self._episode_lengths.append(l)

                if self.verbose >= 1:
                    logger.info(
                        "Episode %d | R=%.2f | Len=%d | Badges=%d | "
                        "Dex=%d | Maps=%d | Goal=%s",
                        len(self._episode_rewards), r, l,
                        badges, dex, maps, goal,
                    )

                if self.log_dir:
                    with open(self._csv_path, "a") as f:
                        f.write(
                            f"{len(self._episode_rewards)},{r:.4f},{l},"
                            f"{badges},{dex},{maps},{coords},{goal}\n"
                        )
        return True


# ---------------------------------------------------------------------------
# Action-logging callback
# ---------------------------------------------------------------------------

class ActionLogCallback(BaseCallback):
    """
    Logs the action distribution over a window of steps.
    Useful for diagnosing action collapse (agent always presses A, etc.).
    """

    WINDOW = 500
    ACTION_NAMES = ["Up", "Down", "Left", "Right", "A", "B", "Start", "NoOp"]

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._action_counts: dict[int, int] = {i: 0 for i in range(8)}
        self._steps = 0

    def _on_step(self) -> bool:
        for action in np.atleast_1d(self.locals["actions"]):
            self._action_counts[int(action)] += 1
        self._steps += 1

        if self._steps % self.WINDOW == 0:
            total = sum(self._action_counts.values())
            dist  = {
                self.ACTION_NAMES[k]: f"{v / total:.1%}"
                for k, v in sorted(self._action_counts.items())
            }
            logger.info("Action distribution (last %d): %s", self.WINDOW, dist)
            self._action_counts = {i: 0 for i in range(8)}
        return True


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

class RLAgent:
    """
    High-level wrapper around an SB3 RL algorithm.

    Parameters
    ----------
    env : PokemonBlueEnv
        The Pokémon environment. Will be wrapped in Monitor + DummyVecEnv.
    algorithm : AlgoType
        "PPO" | "DQN" | "A2C"
    model_dir : str
        Directory for checkpoints and final model saves.
    log_dir : str
        Directory for TensorBoard logs and CSV episode stats.
    normalize_obs : bool
        Wrap env in VecNormalize for running-mean normalisation.
    """

    def __init__(
        self,
        env: PokemonBlueEnv,
        algorithm: AlgoType = "PPO",
        model_dir: str = "models",
        log_dir: str = "logs",
        normalize_obs: bool = False,
        device: str = "auto",
    ):
        self.algorithm = algorithm
        self.model_dir = Path(model_dir)
        self.log_dir   = Path(log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Wrap environment
        monitored = Monitor(env, filename=str(self.log_dir / "monitor"))
        vec_env   = DummyVecEnv([lambda: monitored])

        if normalize_obs:
            self.vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        else:
            self.vec_env = vec_env

        # Hyperparameters per algorithm
        algo_cls = ALGO_MAP[algorithm]
        kwargs   = self._default_hyperparams(algorithm)

        self.model = algo_cls(
            "MlpPolicy",
            self.vec_env,
            tensorboard_log=str(self.log_dir / "tensorboard"),
            device=device,
            verbose=1,
            **kwargs,
        )
        logger.info("Created %s agent | obs_dim=%d | act_dim=%d",
                    algorithm,
                    env.observation_space.shape[0],
                    env.action_space.n)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(
        self,
        total_timesteps: int = 500_000,
        checkpoint_freq: int = 50_000,
        eval_freq: int = 25_000,
    ) -> None:
        callbacks = self._build_callbacks(checkpoint_freq, eval_freq)
        logger.info("Starting training for %d timesteps…", total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        self.save("final")
        logger.info("Training complete.")

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, Optional[np.ndarray]]:
        """Return (action, state) for one observation."""
        action, state = self.model.predict(obs, deterministic=deterministic)
        return int(action), state

    # ------------------------------------------------------------------ #
    # Save / Load                                                          #
    # ------------------------------------------------------------------ #

    def save(self, tag: str = "checkpoint") -> Path:
        path = self.model_dir / f"{self.algorithm}_{tag}"
        self.model.save(str(path))
        logger.info("Model saved → %s", path)
        return path

    def load(self, path: str) -> None:
        algo_cls = ALGO_MAP[self.algorithm]
        self.model = algo_cls.load(
            path,
            env=self.vec_env,
            device=self.model.device,
        )
        logger.info("Model loaded ← %s", path)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _default_hyperparams(self, algorithm: AlgoType) -> dict:
        if algorithm == "PPO":
            return dict(
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,        # encourage exploration
                vf_coef=0.5,
            )
        elif algorithm == "DQN":
            return dict(
                learning_rate=1e-4,
                buffer_size=100_000,
                learning_starts=10_000,
                batch_size=64,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
            )
        elif algorithm == "A2C":
            return dict(
                n_steps=5,
                learning_rate=7e-4,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25,
            )
        return {}

    def _build_callbacks(
        self,
        checkpoint_freq: int,
        eval_freq: int,
    ) -> CallbackList:
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(self.model_dir),
            name_prefix=self.algorithm,
            verbose=1,
        )
        stats_cb = EpisodeStatsCallback(log_dir=str(self.log_dir))
        action_cb = ActionLogCallback()
        return CallbackList([checkpoint_cb, stats_cb, action_cb])


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_agent(
    env: PokemonBlueEnv,
    algorithm: AlgoType = "PPO",
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> RLAgent:
    """
    Create (or restore) an RLAgent.

    If ``checkpoint_path`` is given the weights are loaded from that file.
    Extra kwargs are forwarded to RLAgent.__init__.
    """
    agent = RLAgent(env, algorithm=algorithm, **kwargs)
    if checkpoint_path and os.path.exists(checkpoint_path + ".zip"):
        agent.load(checkpoint_path)
    return agent
