"""
heatmap.py
==========
Builds and saves an exploration heatmap that shows which tiles the agent
has visited and how frequently.

Each map ID gets its own heat layer; layers are composited into a single
colour image where hotter = visited more often.

Usage
-----
    heatmap = ExplorationHeatmap(map_size=(512, 512), log_dir="logs")
    # Record a visit
    heatmap.record(map_id=1, x=10, y=8)
    # Save PNG
    heatmap.save_image("heatmap.png")
    # Generate per-map breakdown
    heatmap.save_per_map_images("logs/maps/")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import matplotlib/PIL; fall back gracefully if not installed
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not installed – heatmap images will not be saved.")

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class ExplorationHeatmap:
    """
    Records (map_id, x, y) visit counts and renders them as heat images.

    Parameters
    ----------
    map_size : tuple[int, int]
        (width, height) of the output image in pixels.
        The tile coordinates are scaled to fit.
    log_dir : str
        Directory where images are saved.
    tile_scale : int
        Pixels per game tile in the output image.
    """

    # Game Boy screen is 20×18 tiles; maps vary up to ~100×100
    # We use a coordinate range and scale dynamically
    COORD_RANGE = 256   # max tile coordinate (0–255)

    def __init__(
        self,
        map_size: tuple[int, int] = (512, 512),
        log_dir: str = "logs",
        tile_scale: int = 4,
    ):
        self.map_size  = map_size
        self.log_dir   = Path(log_dir)
        self.tile_scale = tile_scale
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # visit_counts[map_id][(x, y)] = count
        self.visit_counts: dict[int, dict[tuple[int, int], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.total_visits = 0

    def record(self, map_id: int, x: int, y: int) -> None:
        """Record a single tile visit."""
        self.visit_counts[map_id][(x, y)] += 1
        self.total_visits += 1

    def get_visit_count(self, map_id: int, x: int, y: int) -> int:
        return self.visit_counts[map_id].get((x, y), 0)

    def unique_tiles(self) -> int:
        return sum(len(v) for v in self.visit_counts.values())

    def unique_maps(self) -> int:
        return len(self.visit_counts)

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def _build_grid(
        self,
        counts: dict[tuple[int, int], int],
        grid_w: int = 256,
        grid_h: int = 256,
    ) -> np.ndarray:
        """Build a 2-D visit-count grid (H, W)."""
        grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        for (x, y), count in counts.items():
            cx = min(x, grid_w - 1)
            cy = min(y, grid_h - 1)
            grid[cy, cx] += count
        return grid

    def save_image(
        self,
        filename: str = "exploration_heatmap.png",
        map_id: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Save a heatmap image.

        If map_id is given, renders only that map.
        Otherwise composites all maps together.
        """
        if not _HAS_MPL:
            logger.warning("matplotlib unavailable – skipping heatmap save.")
            return None

        if map_id is not None:
            counts = self.visit_counts.get(map_id, {})
            title  = f"Map {map_id} exploration"
        else:
            # Merge all maps into one grid
            counts = defaultdict(int)
            for mc in self.visit_counts.values():
                for coord, v in mc.items():
                    counts[coord] += v
            title = f"All maps – {self.unique_tiles()} unique tiles"

        grid = self._build_grid(counts)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        img = ax.imshow(
            grid,
            cmap="hot",
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=max(grid.max(), 1),
        )
        plt.colorbar(img, ax=ax, label="Visit count")
        ax.set_title(title)
        ax.set_xlabel("X tile")
        ax.set_ylabel("Y tile")

        out_path = self.log_dir / filename
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap saved → %s", out_path)
        return out_path

    def save_per_map_images(self, subdir: str = "maps") -> None:
        """Save one PNG per visited map."""
        if not _HAS_MPL:
            return
        out_dir = self.log_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for map_id in sorted(self.visit_counts.keys()):
            self.save_image(
                filename=f"map_{map_id:03d}.png",
                map_id=map_id,
            )

    def summary(self) -> str:
        lines = [
            f"Exploration summary:",
            f"  Total visits : {self.total_visits:,}",
            f"  Unique tiles : {self.unique_tiles():,}",
            f"  Unique maps  : {self.unique_maps()}",
        ]
        # Top 5 most-visited maps
        map_totals = {
            mid: sum(c.values())
            for mid, c in self.visit_counts.items()
        }
        top_maps = sorted(map_totals.items(), key=lambda kv: -kv[1])[:5]
        if top_maps:
            lines.append("  Most visited maps:")
            for mid, total in top_maps:
                from environment.memory_reader import MAP_NAMES
                name = MAP_NAMES.get(mid, f"Map {mid}")
                lines.append(f"    [{mid}] {name}: {total:,} visits")
        return "\n".join(lines)
