"""
pokemon_blue_env.py
===================
Gymnasium-compatible environment that wraps the BizHawk emulator bridge.

Observation Space
-----------------
A flat float32 vector of normalised game-state features:
  [player_x/255, player_y/255, map_id/255,
   in_battle, player_hp_frac, enemy_hp_frac,
   money_norm, badges/8,
   pokedex_seen/151, pokedex_caught/151,
   party_level_avg/100,
   goal_one_hot (8 dims)]
Total: 19 dimensions

Action Space
------------
Discrete(8)
  0 = Up      1 = Down    2 = Left    3 = Right
  4 = A       5 = B       6 = Start   7 = NoOp

Goal Space
----------
Goals are set externally by the LLM planner and injected into the observation:
  0 = explore          4 = heal
  1 = battle           5 = use_item
  2 = catch_pokemon    6 = progress_story
  3 = train_levels     7 = idle
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.communication import EmulatorBridge, EmulatorState
from environment.memory_reader import GameState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Goal definitions
# ---------------------------------------------------------------------------
GOALS = [
    "explore",
    "battle",
    "catch_pokemon",
    "train_levels",
    "heal",
    "use_item",
    "progress_story",
    "idle",
]
N_GOALS = len(GOALS)
GOAL_TO_IDX = {g: i for i, g in enumerate(GOALS)}

# ---------------------------------------------------------------------------
# Observation dimensionality
# ---------------------------------------------------------------------------
N_BASE_OBS = 11
OBS_DIM = N_BASE_OBS + N_GOALS   # 11 + 8 = 19

# ---------------------------------------------------------------------------
# Reward shaping constants
# ---------------------------------------------------------------------------
REWARD_NEW_COORD       = 0.02     # per new tile visited
REWARD_NEW_MAP         = 1.0      # per new map discovered
REWARD_WIN_BATTLE      = 2.0
REWARD_CATCH_POKEMON   = 3.0
REWARD_LEVEL_UP        = 1.5      # per level gained
REWARD_EARN_BADGE      = 10.0
REWARD_POKEDEX_SEEN    = 0.3
REWARD_POKEDEX_CAUGHT  = 0.5
REWARD_HEAL_FULL       = 0.5      # when HP restored to max at Pokémon Center

PENALTY_FAINT          = -5.0
PENALTY_LOSE_BATTLE    = -3.0
PENALTY_STUCK          = -0.01    # per step when not moving
PENALTY_IDLE_STEP      = -0.001   # tiny per-step penalty to discourage dawdling

MAX_STEPS_PER_EPISODE  = 10_000


class PokemonBlueEnv(gym.Env):
    """
    Pokémon Blue Gymnasium environment.

    Parameters
    ----------
    bridge : EmulatorBridge
        Active bridge to BizHawk. Must be started before calling reset().
    goal : str
        Current high-level goal string (set by LLM planner).
    step_timeout : float
        Seconds to wait for each emulator state frame.
    max_steps : int
        Maximum steps before the episode is truncated.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        bridge: EmulatorBridge,
        goal: str = "explore",
        step_timeout: float = 5.0,
        max_steps: int = MAX_STEPS_PER_EPISODE,
    ):
        super().__init__()
        self.bridge = bridge
        self.goal = goal
        self.step_timeout = step_timeout
        self.max_steps = max_steps

        # Action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)

        # Observation space: float32 vector in [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(OBS_DIM, dtype=np.float32),
            high=np.ones(OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )

        # Episode tracking
        self._step_count = 0
        self._prev_state: Optional[GameState] = None
        self._prev_emulator_state: Optional[EmulatorState] = None

        # Progress sets (persist across resets for curriculum)
        self.visited_coords: set[tuple[int, int, int]] = set()
        self.visited_maps: set[int] = set()
        self._episode_visited_coords: set[tuple[int, int, int]] = set()

        # Counts for reward shaping
        self._prev_badges = 0
        self._prev_caught = 0
        self._prev_seen = 0
        self._prev_levels_sum = 0
        self._prev_hp = 0
        self._stuck_steps = 0
        self._last_pos: Optional[tuple[int, int, int]] = None

        # Episode reward accumulator
        self.episode_reward = 0.0
        self.episode_stats: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------ #
    # Goal management                                                      #
    # ------------------------------------------------------------------ #

    def set_goal(self, goal: str) -> None:
        """Update the current high-level goal (called by LLM planner)."""
        if goal not in GOAL_TO_IDX:
            logger.warning("Unknown goal '%s', defaulting to 'explore'", goal)
            goal = "explore"
        self.goal = goal
        logger.debug("Goal updated to: %s", goal)

    def _goal_one_hot(self) -> np.ndarray:
        vec = np.zeros(N_GOALS, dtype=np.float32)
        idx = GOAL_TO_IDX.get(self.goal, 0)
        vec[idx] = 1.0
        return vec

    # ------------------------------------------------------------------ #
    # Gymnasium API                                                        #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step_count = 0
        self.episode_reward = 0.0
        self.episode_stats = defaultdict(float)
        self._episode_visited_coords = set()
        self._stuck_steps = 0

        logger.info("Resetting episode (reloading BizHawk savestate)…")
        emu_state = self.bridge.reset_episode()

        if emu_state is None:
            raise RuntimeError(
                "Timed out waiting for emulator after reset. "
                "Is BizHawk running with the Lua script active?"
            )

        state = emu_state.to_game_state()
        self._prev_state = state
        self._prev_emulator_state = emu_state
        self._prev_badges = state.badges
        self._prev_caught = state.pokedex_caught
        self._prev_seen = state.pokedex_seen
        self._prev_levels_sum = sum(state.party_levels)
        self._prev_hp = state.player_hp
        self._last_pos = state.coord

        obs = self._build_obs(state)
        info = {"goal": self.goal, "map_name": state.map_name}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._step_count += 1

        # Send action to emulator and wait for next state
        self.bridge.send_action(int(action))
        emu_state = self.bridge.get_state(timeout=self.step_timeout)

        if emu_state is None:
            logger.warning("Emulator state timeout at step %d", self._step_count)
            # Return zero-reward and terminate
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            return obs, 0.0, True, False, {"timeout": True}

        state = emu_state.to_game_state()
        reward = self._compute_reward(state, action)
        self.episode_reward += reward

        terminated = self._is_terminated(state)
        truncated = self._step_count >= self.max_steps

        obs = self._build_obs(state)
        info = self._build_info(state, reward)

        # Update previous state
        self._prev_state = state
        self._prev_emulator_state = emu_state
        self._prev_badges = state.badges
        self._prev_caught = state.pokedex_caught
        self._prev_seen = state.pokedex_seen
        self._prev_levels_sum = sum(state.party_levels)
        self._prev_hp = state.player_hp
        self._last_pos = state.coord

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """BizHawk handles its own rendering; nothing needed here."""
        pass

    # ------------------------------------------------------------------ #
    # Observation construction                                             #
    # ------------------------------------------------------------------ #

    def _build_obs(self, state: GameState) -> np.ndarray:
        money_norm = min(state.money / 99999.0, 1.0)
        base = np.array([
            state.player_x / 255.0,
            state.player_y / 255.0,
            state.map_id / 255.0,
            float(state.in_battle > 0),
            state.hp_fraction,
            (state.enemy_hp_cur / state.enemy_hp_max)
                if state.enemy_hp_max > 0 else 0.0,
            money_norm,
            state.badges / 8.0,
            state.pokedex_seen / 151.0,
            state.pokedex_caught / 151.0,
            state.avg_party_level / 100.0,
        ], dtype=np.float32)

        goal_vec = self._goal_one_hot()
        return np.concatenate([base, goal_vec])

    # ------------------------------------------------------------------ #
    # Reward computation                                                   #
    # ------------------------------------------------------------------ #

    def _compute_reward(self, state: GameState, action: int) -> float:
        reward = PENALTY_IDLE_STEP

        if self._prev_state is None:
            return reward

        prev = self._prev_state

        # --- Exploration rewards ---
        coord = state.coord
        if coord not in self.visited_coords:
            self.visited_coords.add(coord)
            self._episode_visited_coords.add(coord)
            r = REWARD_NEW_COORD
            if self.goal == "explore":
                r *= 2.0          # double exploration reward when goal is explore
            reward += r
            self.episode_stats["new_coords"] += 1

        if state.map_id not in self.visited_maps:
            self.visited_maps.add(state.map_id)
            reward += REWARD_NEW_MAP
            self.episode_stats["new_maps"] += 1

        # --- Stuck penalty ---
        if coord == self._last_pos:
            self._stuck_steps += 1
            if self._stuck_steps > 20:
                reward += PENALTY_STUCK
        else:
            self._stuck_steps = 0

        # --- Battle rewards ---
        if prev.is_in_battle and not state.is_in_battle:
            # Exited battle
            if state.player_hp > 0:
                # We survived = win (approximate heuristic)
                reward += REWARD_WIN_BATTLE
                self.episode_stats["battles_won"] += 1
                if self.goal == "battle":
                    reward += REWARD_WIN_BATTLE  # extra reward when goal is battle
            else:
                reward += PENALTY_LOSE_BATTLE
                self.episode_stats["battles_lost"] += 1

        # --- HP loss penalty (faint detection) ---
        if state.player_hp == 0 and prev.player_hp > 0:
            reward += PENALTY_FAINT
            self.episode_stats["faints"] += 1

        # --- HP heal reward (at Pokémon Center) ---
        if (
            state.player_hp > prev.player_hp
            and state.player_hp == state.player_hp_max
            and not state.is_in_battle
        ):
            reward += REWARD_HEAL_FULL
            if self.goal == "heal":
                reward += REWARD_HEAL_FULL
            self.episode_stats["heals"] += 1

        # --- Catch reward ---
        if state.pokedex_caught > prev.pokedex_caught:
            gained = state.pokedex_caught - prev.pokedex_caught
            reward += gained * REWARD_CATCH_POKEMON
            if self.goal == "catch_pokemon":
                reward += gained * REWARD_CATCH_POKEMON
            self.episode_stats["pokemon_caught"] += gained

        # --- Pokédex seen ---
        if state.pokedex_seen > prev.pokedex_seen:
            gained = state.pokedex_seen - prev.pokedex_seen
            reward += gained * REWARD_POKEDEX_SEEN
            self.episode_stats["pokemon_seen"] += gained

        # --- Level-up reward ---
        cur_levels_sum = sum(state.party_levels)
        if cur_levels_sum > self._prev_levels_sum:
            gained = cur_levels_sum - self._prev_levels_sum
            reward += gained * REWARD_LEVEL_UP
            if self.goal == "train_levels":
                reward += gained * REWARD_LEVEL_UP
            self.episode_stats["level_ups"] += gained

        # --- Badge reward ---
        if state.badges > prev.badges:
            gained = state.badges - prev.badges
            reward += gained * REWARD_EARN_BADGE
            self.episode_stats["badges"] += gained

        return float(reward)

    def _is_terminated(self, state: GameState) -> bool:
        """Episode ends if all Pokémon have fainted (white-out)."""
        if state.party_count == 0:
            return False   # no data yet
        return state.player_hp == 0 and not state.is_in_battle

    def _build_info(self, state: GameState, step_reward: float) -> dict:
        return {
            "goal": self.goal,
            "map_name": state.map_name,
            "step_reward": step_reward,
            "episode_reward": self.episode_reward,
            "badges": state.badges,
            "pokedex_caught": state.pokedex_caught,
            "max_level": state.max_party_level,
            "maps_visited": len(self.visited_maps),
            "coords_visited": len(self.visited_coords),
            "episode_coords": len(self._episode_visited_coords),
        }
