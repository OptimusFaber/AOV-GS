"""
Interactive CLIP open-vocabulary navigation validator.

Использует только сохранённые CLIP-эмбеддинги (clip_index.pt), без ключевых кадров
как изображений. Загружает index из result_dir/splatam/clip_index.pt (или --clip_index),
CLIP нужен только для кодирования текстового запроса.

  - Ранжирует ключевые кадры по косинусному сходству с запросом.
  - Карта: top-down X/Z, цвет точки = скор (зелёный → красный).
  - В терминал печатается топ-K с оценками и позициями.
  - Сетка картинок топ-K показывается только если есть keyframes/ (опционально).

Usage
-----
  # Предпочтительно: только эмбеддинги из clip_index.pt (без keyframe-изображений)
  python src/evaluation/validate_clip_nav.py --result_dir results/Replica/office0/ActiveSem-CLIP/run_bench_clip

  # Явный путь к .pt
  python src/evaluation/validate_clip_nav.py --clip_index results/.../splatam/clip_index.pt

  # Без GUI
  python src/evaluation/validate_clip_nav.py --result_dir ... --headless
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")       # headless: no window
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
from PIL import Image

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.semantic.clip_encoder import CLIPEncoder
from src.semantic.open_vocab_index import OpenVocabIndex


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive CLIP navigation validator (uses saved embeddings, no keyframe images).")
    p.add_argument(
        "--result_dir", type=str, default=None,
        help="Path to a completed ActiveSGM run. If set, clip_index is taken as result_dir/splatam/clip_index.pt unless --clip_index is given.",
    )
    p.add_argument(
        "--clip_index", type=str, default=None,
        help="Path to saved CLIP index .pt file (embeddings + poses). Overrides result_dir/splatam/clip_index.pt.",
    )
    p.add_argument("--clip_device",  type=str, default="cuda:0")
    p.add_argument("--model_name",   type=str, default="ViT-B-32")
    p.add_argument("--pretrained",   type=str, default="openai")
    p.add_argument("--top_k",        type=int, default=5)
    p.add_argument("--headless",     action="store_true", help="No GUI; only print query results to terminal")
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="[Only for legacy keyframe mode] params.npz for full trajectory.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_keyframe_images(keyframes_dir: str) -> dict[int, str]:
    """Return {kf_idx: image_path} sorted by kf index."""
    paths = sorted(glob.glob(os.path.join(keyframes_dir, "keyframe_*.jpg")))
    out = {}
    for p in paths:
        base = os.path.basename(p)          # keyframe_0042.jpg
        idx  = int(base.split("_")[1].split(".")[0])
        out[idx] = p
    return out


def load_camera_positions(params_path: str) -> np.ndarray:
    """Load (N, 3) camera translation vectors from params.npz.

    `cam_trans` in params.npz has shape [3, num_frames] or [1, 3, num_frames].
    Returns array of shape [num_frames, 3].
    """
    data = np.load(params_path, allow_pickle=True)
    cam_trans = np.asarray(data["cam_trans"])  # [3, N] or [1, 3, N]
    if cam_trans.ndim == 3:
        cam_trans = cam_trans.squeeze(0)
    return cam_trans.T  # [num_frames, 3]


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class CLIPNavigationVisualizer:
    """Matplotlib figure that updates on each text query."""

    CMAP = "RdYlGn_r"   # blue (low score) → red (high score) — reversed so red = best

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._fig  = None
        self._lock = threading.Lock()

    def init_figure(
        self,
        all_positions: np.ndarray,
        kf_indices:    list[int],
    ) -> None:
        """Create the window.  Call once before the query loop.

        Parameters
        ----------
        all_positions : [N, 3] array of ALL camera positions (full trajectory).
        kf_indices    : list of kf indices that have images (subset of trajectory).
        """
        self._all_positions = all_positions   # full trajectory
        self._kf_indices    = kf_indices      # subset with images

        plt.ion()
        self._fig = plt.figure(
            "CLIP Navigation Validator",
            figsize=(16, 8),
            constrained_layout=True,
        )
        gs = gridspec.GridSpec(
            2, self.top_k + 1,
            figure=self._fig,
            height_ratios=[2.5, 1],
        )

        self._ax_map   = self._fig.add_subplot(gs[:, 0])
        self._ax_imgs  = [self._fig.add_subplot(gs[1, k + 1]) for k in range(self.top_k)]
        self._ax_title = self._fig.add_subplot(gs[0, 1:])
        self._ax_title.axis("off")

        self._draw_base_map()
        self._title_text = self._ax_title.text(
            0.5, 0.5, "Type a query in the terminal…",
            ha="center", va="center",
            fontsize=14, color="#555555",
            transform=self._ax_title.transAxes,
        )
        plt.show(block=False)
        plt.pause(0.05)

    def _draw_base_map(self) -> None:
        ax = self._ax_map
        ax.clear()
        # Keyframe positions (X, Z for top-down)
        pos_kf = self._all_positions[self._kf_indices]
        ax.plot(pos_kf[:, 0], pos_kf[:, 2], color="#cccccc", lw=0.8, zorder=1, label="keyframes")
        ax.scatter(
            pos_kf[:, 0], pos_kf[:, 2],
            s=20, c="#aaaaaa", zorder=2, label="keyframe poses",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title("Top-down scene map")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
        self._scat_score   = None   # scatter coloured by score
        self._scat_best    = None   # star on best match
        self._cbar         = None

    def update(
        self,
        query:   str,
        results: list[tuple[float, int, torch.Tensor]],
    ) -> None:
        """Redraw the figure with new query results.

        Parameters
        ----------
        query   : text query string.
        results : list of (score, kf_id, c2w) from OpenVocabIndex.query().
        """
        with self._lock:
            self._draw_base_map()
            ax = self._ax_map

            scores  = np.array([r[0] for r in results])
            kf_ids  = [r[1]  for r in results]

            # --- colour all keyframes by score (grey if not in results) ---
            all_scores = np.zeros(len(self._kf_indices))
            kf_id_to_local = {kfid: i for i, kfid in enumerate(self._kf_indices)}
            for score, kfid, _ in results:
                if kfid in kf_id_to_local:
                    all_scores[kf_id_to_local[kfid]] = score

            norm  = Normalize(vmin=all_scores.min(), vmax=all_scores.max())
            cmap  = plt.get_cmap(self.CMAP)
            colors = cmap(norm(all_scores))

            pos_kf = self._all_positions[self._kf_indices]
            self._scat_score = ax.scatter(
                pos_kf[:, 0], pos_kf[:, 2],
                s=30, c=colors, zorder=3,
            )

            # colourbar
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            if self._cbar is not None:
                self._cbar.remove()
            self._cbar = self._fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
            self._cbar.set_label("CLIP similarity", fontsize=8)

            # --- star on the best match (position from c2w) ---
            best_score, best_kfid, best_c2w = results[0]
            bpos = best_c2w[:3, 3].cpu().numpy()
            ax.scatter(
                bpos[0], bpos[2],
                s=200, marker="*", c="gold", edgecolors="black",
                zorder=5, label=f"best (kf {best_kfid})",
            )
            ax.legend(fontsize=7, loc="upper right")

            # --- title text ---
            self._title_text.set_text(
                f'Query: "{query}"\n'
                f"Best match: kf {best_kfid} | score {best_score:.4f}"
            )
            self._title_text.set_color("#222222")

            # --- top-K images ---
            for k in range(self.top_k):
                ax_img = self._ax_imgs[k]
                ax_img.clear()
                ax_img.axis("off")
                if k < len(results):
                    score, kfid, _ = results[k]
                    img_path = self._kf_img_paths.get(kfid)
                    if img_path and os.path.exists(img_path):
                        img = np.array(Image.open(img_path))
                        ax_img.imshow(img)
                    border_color = "gold" if k == 0 else "#888888"
                    ax_img.set_title(
                        f"#{k+1}  kf {kfid}\n{score:.3f}",
                        fontsize=7, color=border_color if k == 0 else "black",
                    )
                    for spine in ax_img.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(2 if k == 0 else 0.5)

            self._fig.canvas.draw_idle()
            plt.pause(0.05)

    def set_kf_image_paths(self, kf_img_paths: dict[int, str]) -> None:
        self._kf_img_paths = kf_img_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _positions_from_index(index: OpenVocabIndex) -> tuple[np.ndarray, list[int]]:
    """Build all_positions (indexable by kf_id) and kf_indices from loaded index."""
    kf_ids_sorted = sorted(index._records.keys())
    positions = np.array([
        index._records[kid].c2w[:3, 3].cpu().numpy()
        for kid in kf_ids_sorted
    ])
    max_id = max(kf_ids_sorted)
    all_positions = np.zeros((max_id + 1, 3), dtype=np.float64)
    for kfid, pos in zip(kf_ids_sorted, positions):
        all_positions[kfid] = pos
    return all_positions, kf_ids_sorted


def main() -> None:
    args = parse_args()

    clip_index_path = args.clip_index
    if clip_index_path is None and args.result_dir:
        clip_index_path = os.path.join(args.result_dir, "splatam", "clip_index.pt")

    if clip_index_path is None or not os.path.isfile(clip_index_path):
        print(
            "[ERROR] No CLIP index found. Provide --result_dir (with splatam/clip_index.pt) or --clip_index PATH.\n"
            "Run training with CLIP enabled so that clip_index.pt is saved at validation steps."
        )
        sys.exit(1)

    print(f"Loading CLIP index from {clip_index_path} (no keyframe images).")
    encoder = CLIPEncoder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.clip_device,
    )
    index = OpenVocabIndex.load(
        clip_index_path,
        encoder,
        update_every=1,
        top_k=args.top_k,
    )
    all_positions, kf_indices = _positions_from_index(index)
    kf_img_paths = {}   # optional: thumbnails if result_dir/keyframes exists

    if args.result_dir:
        keyframes_dir = os.path.join(args.result_dir, "keyframes")
        if not os.path.isdir(keyframes_dir):
            keyframes_dir = os.path.join(args.result_dir, "splatam", "keyframes")
        if os.path.isdir(keyframes_dir):
            kf_img_paths = load_keyframe_images(keyframes_dir)
            # keep only indices that are in the index
            kf_img_paths = {k: v for k, v in kf_img_paths.items() if k in index._records}

    # ---- set up visualiser (skip figure in headless) ----
    vis = None
    if not args.headless:
        vis = CLIPNavigationVisualizer(top_k=args.top_k)
        vis.set_kf_image_paths(kf_img_paths)
        vis.init_figure(all_positions, kf_indices)
    else:
        print("(Headless mode: no plot window; results printed to terminal only)")

    print("\n" + "="*60)
    print("CLIP Navigation Validator — embedding-only mode")
    print("="*60)
    print("Type a text query and press Enter. 'quit' or Ctrl-C to exit.\n")

    try:
        while True:
            try:
                query = input("Query > ").strip()
            except EOFError:
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            results = index.query(query, top_k=args.top_k)
            if not results:
                print("  [!] Index returned no results.")
                continue

            print(f"\n  Results for: \"{query}\"")
            print(f"  {'Rank':<5}  {'kf_id':<8}  {'Score':<8}  Position (X, Y, Z)")
            print(f"  {'-'*55}")
            for rank, (score, kfid, c2w) in enumerate(results, 1):
                pos = c2w[:3, 3].cpu().numpy()
                print(
                    f"  {rank:<5}  {kfid:<8}  {score:<8.4f}  "
                    f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                )
            print()

            if vis is not None:
                vis.update(query, results)

    except KeyboardInterrupt:
        pass

    print("\nExiting.")
    if vis is not None:
        plt.close("all")


if __name__ == "__main__":
    main()
