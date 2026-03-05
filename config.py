"""
config.py
=========
Central configuration for the Pokémon Blue AI project.

All paths, hyperparameters, and communication settings live here.
Override values by editing this file or by passing CLI flags to train.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

AlgoType = Literal["PPO", "DQN", "A2C"]


@dataclass
class TrainingConfig:
    # ── RL algorithm ──────────────────────────────────────────────────── #
    algorithm: AlgoType = "PPO"
    total_timesteps: int = 500_000
    max_steps_per_episode: int = 10_000
    checkpoint_freq: int = 50_000
    device: str = "auto"                # "cpu" | "cuda" | "auto"

    # ── Communication ─────────────────────────────────────────────────── #
    host: str = "127.0.0.1"
    port: int = 65432
    connect_timeout: float = 120.0      # seconds to wait for BizHawk
    step_timeout: float = 5.0           # seconds per emulator step

    # ── LLM planner ───────────────────────────────────────────────────── #
    llm_model: str = "claude-opus-4-6"
    planner_interval: int = 500         # steps between planner calls
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )

    # ── File paths ────────────────────────────────────────────────────── #
    model_dir: str = "models"
    log_dir: str = "logs"
    savestate_path: str = "savestates/start.State"
    resume_checkpoint: Optional[str] = None

    # ── Reward shaping ────────────────────────────────────────────────── #
    reward_new_coord: float = 0.02
    reward_new_map: float = 1.0
    reward_win_battle: float = 2.0
    reward_catch: float = 3.0
    reward_level_up: float = 1.5
    reward_badge: float = 10.0
    penalty_faint: float = -5.0
    penalty_lose_battle: float = -3.0
    penalty_stuck: float = -0.01
    penalty_idle: float = -0.001

    # ── Emulator settings (matching Lua script) ───────────────────────── #
    frame_skip: int = 4                 # frames between state reads
    action_hold: int = 8                # frames each button is held

    def __post_init__(self) -> None:
        # Ensure directories exist
        for d in [self.model_dir, self.log_dir, Path(self.savestate_path).parent]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        return (
            f"TrainingConfig(\n"
            f"  algorithm={self.algorithm}, steps={self.total_timesteps:,}\n"
            f"  host={self.host}:{self.port}\n"
            f"  llm={self.llm_model}, planner_every={self.planner_interval} steps\n"
            f"  model_dir={self.model_dir}, log_dir={self.log_dir}\n"
            f"  device={self.device}\n"
            f")"
        )


# Default config singleton
DEFAULT_CONFIG = TrainingConfig()
