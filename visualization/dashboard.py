"""
dashboard.py
============
Training statistics dashboard.

Records per-episode metrics and can render them as:
  • matplotlib plots (reward curve, badge progression, Pokédex, FPS)
  • A simple ASCII table to stdout (no dependencies)

Usage
-----
    dashboard = TrainingDashboard(log_dir="logs")
    dashboard.record_episode(episode=1, reward=3.5, steps=1200, ...)
    dashboard.save_plots()          # saves PNGs to logs/
    dashboard.print_summary()       # prints ASCII table
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# Goal → short colour string for plotting
GOAL_COLOURS = {
    "explore":        "steelblue",
    "battle":         "firebrick",
    "catch_pokemon":  "forestgreen",
    "train_levels":   "darkorange",
    "heal":           "orchid",
    "use_item":       "peru",
    "progress_story": "gold",
    "idle":           "silver",
}


class TrainingDashboard:
    """
    Collects episode statistics and renders plots.

    Parameters
    ----------
    log_dir : str
        Directory for saving CSV data and PNG plots.
    smoothing : int
        Moving average window for smoothed reward curve.
    """

    def __init__(self, log_dir: str = "logs", smoothing: int = 20):
        self.log_dir   = Path(log_dir)
        self.smoothing = smoothing
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Per-episode records
        self.episodes:  list[int]   = []
        self.rewards:   list[float] = []
        self.lengths:   list[int]   = []
        self.badges:    list[int]   = []
        self.pokedex:   list[int]   = []
        self.goals:     list[str]   = []
        self.fps_list:  list[float] = []

        # Running accumulators
        self._total_steps = 0
        self._goal_rewards: dict[str, list[float]] = defaultdict(list)

        # CSV file
        self._csv_path = self.log_dir / "training_dashboard.csv"
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "reward", "steps", "badges",
                "pokedex", "goal", "fps"
            ])

    # ------------------------------------------------------------------ #
    # Data ingestion                                                       #
    # ------------------------------------------------------------------ #

    def record_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        badges: int = 0,
        pokedex: int = 0,
        goal: str = "explore",
        fps: float = 0.0,
    ) -> None:
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.lengths.append(steps)
        self.badges.append(badges)
        self.pokedex.append(pokedex)
        self.goals.append(goal)
        self.fps_list.append(fps)
        self._total_steps += steps
        self._goal_rewards[goal].append(reward)

        # Append to CSV
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, f"{reward:.4f}", steps, badges, pokedex, goal, f"{fps:.1f}"])

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #

    def save_plots(self) -> None:
        if not _HAS_MPL:
            logger.warning("matplotlib unavailable – skipping plot generation.")
            return
        if len(self.episodes) < 2:
            return

        self._plot_reward_curve()
        self._plot_badge_dex()
        self._plot_episode_length()
        self._plot_goal_rewards()
        logger.info("Training plots saved to %s", self.log_dir)

    def _moving_average(self, data: list[float], w: int) -> list[float]:
        out = []
        for i in range(len(data)):
            start = max(0, i - w + 1)
            out.append(sum(data[start:i + 1]) / (i - start + 1))
        return out

    def _plot_reward_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.episodes, self.rewards, alpha=0.3, color="steelblue",
                linewidth=0.8, label="Episode reward")
        smooth = self._moving_average(self.rewards, self.smoothing)
        ax.plot(self.episodes, smooth, color="steelblue", linewidth=2,
                label=f"MA-{self.smoothing}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title("Episode Reward Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.log_dir / "reward_curve.png")
        plt.close(fig)

    def _plot_badge_dex(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.episodes, self.badges, color="gold", linewidth=1.5)
        ax1.fill_between(self.episodes, self.badges, alpha=0.2, color="gold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Badges")
        ax1.set_ylim(0, 8)
        ax1.set_title("Gym Badges Earned")
        ax1.grid(alpha=0.3)

        ax2.plot(self.episodes, self.pokedex, color="forestgreen", linewidth=1.5)
        ax2.fill_between(self.episodes, self.pokedex, alpha=0.2, color="forestgreen")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Pokémon caught")
        ax2.set_ylim(0, 151)
        ax2.set_title("Pokédex Progress")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.log_dir / "badge_pokedex.png")
        plt.close(fig)

    def _plot_episode_length(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.episodes, self.lengths, alpha=0.4, color="gray", linewidth=0.8)
        smooth = self._moving_average([float(l) for l in self.lengths], self.smoothing)
        ax.plot(self.episodes, smooth, color="gray", linewidth=2,
                label=f"MA-{self.smoothing}")
        ax2 = ax.twinx()
        ax2.plot(self.episodes, self.fps_list, color="coral", alpha=0.5,
                 linewidth=0.8, label="FPS")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps per episode")
        ax2.set_ylabel("Steps/second", color="coral")
        ax.set_title("Episode Length & Training FPS")
        fig.tight_layout()
        fig.savefig(self.log_dir / "episode_length.png")
        plt.close(fig)

    def _plot_goal_rewards(self) -> None:
        goals_present = [g for g in self._goal_rewards if self._goal_rewards[g]]
        if not goals_present:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        for goal in goals_present:
            r_list = self._goal_rewards[goal]
            colour = GOAL_COLOURS.get(goal, "gray")
            ax.hist(r_list, bins=30, alpha=0.5, label=goal, color=colour, density=True)

        ax.set_xlabel("Episode reward")
        ax.set_ylabel("Density")
        ax.set_title("Reward Distribution by Goal")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.log_dir / "goal_reward_dist.png")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Text summary                                                         #
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        if not self.episodes:
            print("No episodes recorded yet.")
            return

        n = len(self.episodes)
        avg_r = sum(self.rewards) / n
        max_r = max(self.rewards)
        max_badges = max(self.badges)
        max_dex    = max(self.pokedex)
        avg_fps    = sum(self.fps_list) / max(n, 1)

        # Goal distribution
        goal_counts: dict[str, int] = defaultdict(int)
        for g in self.goals:
            goal_counts[g] += 1

        print("\n" + "=" * 60)
        print("  TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Episodes      : {n}")
        print(f"  Total steps   : {self._total_steps:,}")
        print(f"  Avg reward    : {avg_r:.3f}")
        print(f"  Best reward   : {max_r:.3f}")
        print(f"  Max badges    : {max_badges}/8")
        print(f"  Max Pokédex   : {max_dex}/151")
        print(f"  Avg FPS       : {avg_fps:.1f}")
        print("\n  Goal distribution:")
        for g, c in sorted(goal_counts.items(), key=lambda kv: -kv[1]):
            pct = c / n * 100
            bar = "█" * int(pct / 2)
            print(f"    {g:<20} {bar:<25} {pct:.1f}%")
        print("=" * 60)
