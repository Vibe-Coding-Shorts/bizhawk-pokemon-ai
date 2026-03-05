"""
train.py
========
Main training script. Orchestrates:
  1. Start EmulatorBridge (waits for BizHawk to connect)
  2. Build goal-conditioned PokemonBlueEnv
  3. Create or restore RLAgent (PPO by default)
  4. Create LLMPlanner (falls back to RuleBasedPlanner if no API key)
  5. Run training loop:
       a. Episode reset
       b. RL agent steps
       c. LLM planner called periodically → updates goal
       d. Checkpoint saves
       e. Logging and visualisation hooks

Run from the project root:
    python -m training.train [options]

Options are parsed from config.py or overridden via CLI flags.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup logging before importing heavy packages
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import numpy as np

from config import TrainingConfig
from environment.communication import EmulatorBridge
from environment.pokemon_blue_env import PokemonBlueEnv
from ai.rl_agent import RLAgent, make_agent
from ai.planner_llm import LLMPlanner, RuleBasedPlanner
from visualization.heatmap import ExplorationHeatmap
from visualization.dashboard import TrainingDashboard


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    logger.warning("Shutdown signal received – finishing current episode…")
    _shutdown_requested = True

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def training_loop(cfg: TrainingConfig) -> None:
    """
    Full training loop integrating RL agent + LLM planner.

    The loop runs episodes one at a time. Inside each episode we call:
      env.step(action) which blocks until BizHawk advances frame_skip frames.
    The LLM planner is consulted every cfg.planner_interval steps.
    """

    # ── 1. Start emulator bridge ─────────────────────────────────────────
    logger.info("Starting EmulatorBridge on %s:%d", cfg.host, cfg.port)
    bridge = EmulatorBridge(host=cfg.host, port=cfg.port)
    bridge.start()

    logger.info("Waiting for BizHawk to connect… (run the Lua script in BizHawk)")
    t0 = time.time()
    while not bridge.connected:
        if time.time() - t0 > cfg.connect_timeout:
            logger.error("BizHawk did not connect within %ds. Exiting.", cfg.connect_timeout)
            bridge.stop()
            return
        time.sleep(0.5)
    logger.info("BizHawk connected.")

    # ── 2. Build environment ─────────────────────────────────────────────
    env = PokemonBlueEnv(
        bridge=bridge,
        goal="explore",
        step_timeout=cfg.step_timeout,
        max_steps=cfg.max_steps_per_episode,
    )

    # ── 3. Build RL agent ────────────────────────────────────────────────
    agent = make_agent(
        env=env,
        algorithm=cfg.algorithm,
        model_dir=cfg.model_dir,
        log_dir=cfg.log_dir,
        checkpoint_path=cfg.resume_checkpoint,
        device=cfg.device,
    )

    # ── 4. Build LLM planner ─────────────────────────────────────────────
    api_key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        planner = LLMPlanner(
            api_key=api_key,
            model=cfg.llm_model,
            cooldown_steps=cfg.planner_interval,
        )
        logger.info("LLM planner active (model=%s)", cfg.llm_model)
    else:
        planner = RuleBasedPlanner()
        logger.warning(
            "ANTHROPIC_API_KEY not set – using rule-based fallback planner. "
            "Set the env var to enable the LLM planner."
        )

    # ── 5. Optional visualisation ────────────────────────────────────────
    heatmap   = ExplorationHeatmap(map_size=(512, 512), log_dir=cfg.log_dir)
    dashboard = TrainingDashboard(log_dir=cfg.log_dir)

    # ── 6. Outer training loop ───────────────────────────────────────────
    global_step    = 0
    episode_num    = 0
    best_reward    = float("-inf")

    logger.info("=" * 60)
    logger.info("Training starts | algo=%s | max_steps=%d",
                cfg.algorithm, cfg.total_timesteps)
    logger.info("=" * 60)

    while global_step < cfg.total_timesteps and not _shutdown_requested:
        episode_num += 1

        # ── Episode reset ────────────────────────────────────────────────
        obs, info = env.reset()
        if obs is None:
            logger.error("Environment reset failed – is BizHawk running?")
            time.sleep(2.0)
            continue

        episode_reward  = 0.0
        episode_steps   = 0
        episode_t0      = time.time()
        done            = False

        logger.info("── Episode %d | goal=%s ──", episode_num, env.goal)

        # ── Per-episode step loop ────────────────────────────────────────
        while not done and not _shutdown_requested:
            global_step   += 1
            episode_steps += 1

            # LLM planner check
            if planner.should_plan(global_step):
                emu_state = bridge._prev_state if hasattr(bridge, "_prev_state") else None
                # Use last known game state from env's internal tracking
                game_state = env._prev_state
                if game_state is not None:
                    planner.update_context(
                        maps_visited=len(env.visited_maps),
                        coords_visited=len(env.visited_coords),
                        episodes=episode_num,
                    )
                    new_goal, reasoning = planner.get_goal(
                        state=game_state,
                        current_step=global_step,
                    )
                    env.set_goal(new_goal)
                    logger.debug(
                        "Step %d | goal=%s | %s", global_step, new_goal, reasoning
                    )

            # RL agent selects action
            action, _ = agent.predict(obs, deterministic=False)
            planner.record_action(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Update heatmap
            if env._prev_state:
                s = env._prev_state
                heatmap.record(s.map_id, s.player_x, s.player_y)

            # Log milestones
            if global_step % 1000 == 0:
                logger.info(
                    "Step %6d | ep=%d | R_ep=%.2f | goal=%s | "
                    "maps=%d | badges=%d",
                    global_step, episode_num, episode_reward,
                    env.goal, len(env.visited_maps), info.get("badges", 0),
                )

            # Model checkpoint
            if global_step % cfg.checkpoint_freq == 0:
                path = agent.save(tag=f"step_{global_step:07d}")
                logger.info("Checkpoint saved: %s", path)

        # ── Episode end ──────────────────────────────────────────────────
        ep_time = time.time() - episode_t0
        fps     = episode_steps / max(ep_time, 0.001)

        logger.info(
            "Episode %d done | R=%.2f | steps=%d | %.1f steps/s | "
            "maps=%d | badges=%d | caught=%d",
            episode_num, episode_reward, episode_steps, fps,
            len(env.visited_maps),
            info.get("badges", 0),
            info.get("pokedex_caught", 0),
        )

        dashboard.record_episode(
            episode=episode_num,
            reward=episode_reward,
            steps=episode_steps,
            badges=info.get("badges", 0),
            pokedex=info.get("pokedex_caught", 0),
            goal=env.goal,
            fps=fps,
        )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(tag="best")
            logger.info("New best episode reward: %.2f", best_reward)

        # Periodically save heatmap and dashboard
        if episode_num % 10 == 0:
            heatmap.save_image("exploration_heatmap.png")
            dashboard.save_plots()

    # ── Cleanup ──────────────────────────────────────────────────────────
    logger.info("Training loop exiting. Saving final model…")
    agent.save(tag="final")
    heatmap.save_image("exploration_heatmap_final.png")
    dashboard.save_plots()
    bridge.stop()
    logger.info("Done. Total steps: %d | Episodes: %d", global_step, episode_num)


# ---------------------------------------------------------------------------
# SB3 native training (alternative: delegate entirely to SB3's learn())
# ---------------------------------------------------------------------------

def sb3_training(cfg: TrainingConfig) -> None:
    """
    Alternative: let SB3 handle the entire training loop via model.learn().
    This is simpler but doesn't integrate the LLM planner mid-episode.
    The planner is called at the start of each episode via a custom callback.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    bridge = EmulatorBridge(host=cfg.host, port=cfg.port)
    bridge.start()
    logger.info("Waiting for BizHawk…")
    deadline = time.time() + cfg.connect_timeout
    while not bridge.connected and time.time() < deadline:
        time.sleep(0.5)

    env = PokemonBlueEnv(bridge=bridge, goal="explore",
                         max_steps=cfg.max_steps_per_episode)

    class PlannerCallback(BaseCallback):
        def __init__(self, planner, env_ref):
            super().__init__()
            self.planner = planner
            self.env_ref = env_ref

        def _on_rollout_start(self):
            """Called at the start of each rollout collection."""
            if self.env_ref._prev_state:
                goal, _ = self.planner.get_goal(
                    self.env_ref._prev_state,
                    current_step=self.num_timesteps,
                    force=True,
                )
                self.env_ref.set_goal(goal)

        def _on_step(self):
            return True

    api_key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    planner = (
        LLMPlanner(api_key=api_key, model=cfg.llm_model)
        if api_key
        else RuleBasedPlanner()
    )

    agent = make_agent(env=env, algorithm=cfg.algorithm,
                       model_dir=cfg.model_dir, log_dir=cfg.log_dir,
                       checkpoint_path=cfg.resume_checkpoint,
                       device=cfg.device)

    planner_cb = PlannerCallback(planner, env)
    agent.train(
        total_timesteps=cfg.total_timesteps,
        checkpoint_freq=cfg.checkpoint_freq,
    )

    bridge.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pokémon Blue AI (RL + LLM planner)"
    )
    parser.add_argument("--algo",       default="PPO",
                        choices=["PPO", "DQN", "A2C"])
    parser.add_argument("--steps",      type=int,   default=500_000)
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Path to existing checkpoint to resume from")
    parser.add_argument("--model-dir",  default="models")
    parser.add_argument("--log-dir",    default="logs")
    parser.add_argument("--host",       default="127.0.0.1")
    parser.add_argument("--port",       type=int,   default=65432)
    parser.add_argument("--llm-model",  default="claude-opus-4-6")
    parser.add_argument("--planner-interval", type=int, default=500)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--sb3",        action="store_true",
                        help="Use SB3 native training loop instead of custom loop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainingConfig(
        algorithm=args.algo,
        total_timesteps=args.steps,
        resume_checkpoint=args.checkpoint,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        host=args.host,
        port=args.port,
        llm_model=args.llm_model,
        planner_interval=args.planner_interval,
        device=args.device,
    )

    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    if args.sb3:
        sb3_training(cfg)
    else:
        training_loop(cfg)


if __name__ == "__main__":
    main()
