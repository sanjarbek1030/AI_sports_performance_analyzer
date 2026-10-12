"""Heatmaps of TIME SPENT per area (not just paths), in field coordinates.

Each observation adds the time until the next observation (capped at the max
gap) to the 1 m x 1 m cell it occupies; the grid is then Gaussian-smoothed.
Outputs PNGs: one per player, one per team, one for the whole match.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from ..config import Config  # noqa: E402
from ..field.football_field import FieldModel  # noqa: E402
from ..models import PlayerTrack, TEAM_REFEREE  # noqa: E402

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    def __init__(self, config: Config, field: FieldModel, effective_fps: float) -> None:
        self.cfg, self.field = config, field
        self.frame_dt = 1.0 / effective_fps
        self.nx = int(np.ceil(field.length / config.heatmap.bin_m))
        self.ny = int(np.ceil(field.width / config.heatmap.bin_m))

    def accumulate(self, track: PlayerTrack) -> np.ndarray:
        """Seconds per cell, shape (ny, nx)."""
        obs = [o for o in track.observations if o.smoothed_xy is not None and o.on_pitch]
        grid = np.zeros((self.ny, self.nx))
        if not obs:
            return grid
        t = np.array([o.timestamp for o in obs])
        xy = np.array([o.smoothed_xy for o in obs])
        dt = np.minimum(np.append(np.diff(t), self.frame_dt), max(self.cfg.movement.max_gap_s, self.frame_dt))
        ix = np.clip((xy[:, 0] / self.cfg.heatmap.bin_m).astype(int), 0, self.nx - 1)
        iy = np.clip((xy[:, 1] / self.cfg.heatmap.bin_m).astype(int), 0, self.ny - 1)
        np.add.at(grid, (iy, ix), dt)
        return grid

    def render(self, grid: np.ndarray, title: str, path: Path) -> None:
        smooth = gaussian_filter(grid, self.cfg.heatmap.sigma_bins)
        fig, ax = plt.subplots(figsize=(8, 5.4), facecolor="#1f3d1f")
        ax.set_facecolor("#2f6b2f")
        peak = smooth.max()
        image = ax.imshow(smooth / peak if peak > 0 else smooth, extent=(0, self.field.length, self.field.width, 0),
                          cmap=self.cfg.heatmap.colormap, alpha=0.75, vmin=0, vmax=1, interpolation="bilinear")
        self.field.draw_matplotlib(ax)
        ax.set_title(title, color="white")
        cbar = fig.colorbar(image, ax=ax, fraction=0.03)
        cbar.set_label("relative time spent", color="white")
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        fig.savefig(path, dpi=self.cfg.heatmap.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def generate_all(self, tracks: dict[int, PlayerTrack], out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        names = self.cfg.visualization.team_names
        written: list[Path] = []
        team_grids = {0: np.zeros((self.ny, self.nx)), 1: np.zeros((self.ny, self.nx))}
        for track in sorted(tracks.values(), key=lambda t: t.player_id):
            if track.team_id == TEAM_REFEREE:
                continue
            grid = self.accumulate(track)
            if track.team_id in team_grids:
                team_grids[track.team_id] += grid
            if grid.sum() >= self.cfg.heatmap.min_player_seconds:
                label = names[track.team_id] if track.team_id in (0, 1) else "unknown team"
                path = out_dir / f"player_{track.player_id}.png"
                self.render(grid, f"Player {track.player_id} ({label})", path)
                written.append(path)
        for team_id, grid in team_grids.items():
            if grid.sum() > 0:
                path = out_dir / f"team_{names[team_id].replace(' ', '_')}.png"
                self.render(grid, f"{names[team_id]} - time spent", path)
                written.append(path)
        combined = sum(team_grids.values())
        if combined.sum() > 0:
            path = out_dir / "match_combined.png"
            self.render(combined, "Both teams - combined", path)
            written.append(path)
        logger.info("Heatmaps written: %d files in %s", len(written), out_dir)
        return written
