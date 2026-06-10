"""
Top-down (XY) visualization of the robot exploration path in Habitat (RUB) coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

# Headless-safe backend (activesgm also calls configure_headless_env).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _poses_to_xy(poses_c2w: Sequence[np.ndarray]) -> np.ndarray:
    """(N, 2) camera centers in world XY."""
    pts = []
    for p in poses_c2w:
        p = np.asarray(p, dtype=np.float64)
        pts.append(p[:3, 3][:2])
    return np.stack(pts, axis=0) if pts else np.zeros((0, 2))


def _camera_forward_xy(c2w: np.ndarray) -> np.ndarray:
    """Unit forward direction projected onto the floor (XY). Camera looks along -Z."""
    R = np.asarray(c2w, dtype=np.float64)[:3, :3]
    fwd = -R[:, 2]
    xy = fwd[:2]
    n = float(np.linalg.norm(xy))
    if n < 1e-8:
        return np.array([1.0, 0.0])
    return xy / n


def save_exploration_path_topdown(
    poses_c2w: Sequence[np.ndarray],
    save_dir: Union[str, Path],
    *,
    bbox_xy: Optional[Sequence[Sequence[float]]] = None,
    scene_name: str = "",
    run_tag: str = "",
    arrow_every: int = 5,
    dpi: int = 150,
) -> Path:
    """
    Save ``exploration_path_topdown.png`` and ``exploration_path_poses.json``.

    Parameters
    ----------
    poses_c2w:
        Sequence of 4×4 camera-to-world poses in Habitat RUB coordinates.
    save_dir:
        Same directory as ``main_cfg.json`` (``dirs.result_dir``).
    bbox_xy:
        Optional ``[[x_min, x_max], [y_min, y_max]]`` for scene bounds overlay.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    poses_list = [np.asarray(p, dtype=np.float64).tolist() for p in poses_c2w]
    json_path = save_dir / "exploration_path_poses.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "coord_system": "habitat_RUB",
                "scene": scene_name,
                "run_tag": run_tag,
                "num_poses": len(poses_list),
                "poses_c2w": poses_list,
            },
            f,
            indent=2,
        )

    png_path = save_dir / "exploration_path_topdown.png"
    if len(poses_c2w) == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Exploration path (empty)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return png_path

    xy = _poses_to_xy(poses_c2w)
    steps = np.arange(len(xy))

    fig, ax = plt.subplots(figsize=(10, 10))

    if bbox_xy is not None and len(bbox_xy) >= 2:
        x0, x1 = float(bbox_xy[0][0]), float(bbox_xy[0][1])
        y0, y1 = float(bbox_xy[1][0]), float(bbox_xy[1][1])
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="gray",
                linestyle="--",
                linewidth=1.0,
                label="SLAM bbox",
            )
        )

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=steps,
        cmap="viridis",
        s=18,
        zorder=3,
        label="steps",
    )
    ax.plot(xy[:, 0], xy[:, 1], color="steelblue", linewidth=1.2, alpha=0.85, zorder=2)

    ax.scatter(xy[0, 0], xy[0, 1], s=120, c="limegreen", edgecolors="black", zorder=5, label="start")
    ax.scatter(xy[-1, 0], xy[-1, 1], s=120, c="red", edgecolors="black", zorder=5, label="end")

    if arrow_every > 0 and len(poses_c2w) > 1:
        scale = 0.25
        for idx in range(0, len(poses_c2w), arrow_every):
            p = np.asarray(poses_c2w[idx])
            fxy = _camera_forward_xy(p) * scale
            ax.arrow(
                xy[idx, 0],
                xy[idx, 1],
                fxy[0],
                fxy[1],
                head_width=0.08,
                head_length=0.06,
                fc="darkorange",
                ec="darkorange",
                alpha=0.75,
                zorder=4,
                length_includes_head=True,
            )

    title = "Exploration path (top-down, XY)"
    if scene_name:
        title += f" — {scene_name}"
    if run_tag:
        title += f" [{run_tag}]"
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("step index")
    ax.legend(loc="upper right", fontsize=9)

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path
