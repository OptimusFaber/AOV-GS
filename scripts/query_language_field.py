"""
Open-vocabulary object localization in a trained Gaussian language field.

Pipeline
--------
1. Score every Gaussian: cosine_sim(lang_feat, CLIP_text_query_encoded_via_AE).
2. Take top-percentile Gaussians as candidates.
3. DBSCAN-cluster their 3D positions → find semantic clusters (optional ``--no_clusters``).
4. Rank clusters by mean/total/max score (``--cluster_rank_by``; default **mean** — не только «самый большой» кластер).
5. Pick a view from **either** ``--poses`` (JSON keyframes) **or** auto-sampled poses
   inside the Gaussian hull (``--poses`` omitted): **checkpoint is never used** for
   trajectory; with ``--pose_select relevancy`` the pose is scored by the relevancy map
   (default ``--pose_score_mode global_topmean``, не только патч у проекции центроида).
6. Render RGB + relevancy heatmap + overlay; SAM point from relevancy peak/COM by default.
7. Save 3D scatter map showing all clusters.

Usage
-----
python scripts/query_language_field.py \\
  --checkpoint  results/splatam/final/params.npz \\
  --lang_field  results/lang_field/lang_field.pt \\
  --text        "a sofa" \\
  --ae_ckpt     ckpt/room0/best_ckpt.pth \\
  --out         results/lang_field/query_sofa

  # Позы: не указывайте --poses → скрипт сам сэмплирует камеры внутри сцены.
  # Или: --poses results/keyframe_poses.json (как раньше).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.slam.langsplatam.langsplatam import LangSplatam


def infer_latent_dim(lang_field_pt: Path) -> int:
    """Read latent_dim from lang_field.pt (saved by train_language_field.py)."""
    ckpt = torch.load(str(lang_field_pt), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"{lang_field_pt} must contain a dict checkpoint.")
    if ckpt.get("format") == "langsplatv2" or "language_feature_logits" in ckpt:
        return int(
            ckpt.get("latent_dim")
            or ckpt["language_feature_logits"].shape[1]
        )
    if "latent_dim" in ckpt:
        return int(ckpt["latent_dim"])
    lf = ckpt.get("lang_feats")
    if lf is None:
        raise ValueError(f"{lang_field_pt} must contain 'lang_feats' or 'latent_dim'.")
    return int(lf.shape[1])


# ---------------------------------------------------------------------------
# Query encoding helpers
# ---------------------------------------------------------------------------

def _load_ae(ae_ckpt: Path, encoder_dims, decoder_dims, device: torch.device):
    from src.semantic.language_autoencoder import Autoencoder
    ae = Autoencoder(encoder_dims, decoder_dims).to(device)
    ae.load_state_dict(torch.load(str(ae_ckpt), map_location=device))
    ae.eval()
    return ae


def encode_query_clip(text: str, clip_model: str, clip_pretrained: str,
                      device: torch.device) -> torch.Tensor:
    """Return unit-norm 512-d CLIP text embedding."""
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device)
    clip.eval()
    tok = open_clip.get_tokenizer(clip_model)
    with torch.no_grad():
        return F.normalize(clip.encode_text(tok([text]).to(device))[0], dim=-1)


def encode_query(text: str, clip_model: str, clip_pretrained: str,
                 ae_ckpt: Path, device: torch.device,
                 encoder_dims: list = None,
                 decoder_dims: list = None) -> torch.Tensor:
    """Return unit-norm latent-dim query vector (AE.encode of CLIP text emb)."""
    emb = encode_query_clip(text, clip_model, clip_pretrained, device)
    ae  = _load_ae(ae_ckpt, encoder_dims, decoder_dims, device)
    with torch.no_grad():
        return ae.encode(emb.unsqueeze(0))[0]   # [latent_dim]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def find_clusters(means3d: np.ndarray,       # [K, 3]  top-% Gaussian positions
                  scores: np.ndarray,         # [K]     their cosine scores
                  eps: float = 0.15,
                  min_samples: int = 30,
                  ) -> list[dict]:
    """
    DBSCAN cluster the top-scoring Gaussians.
    Returns list of dicts sorted by total_score desc:
      { 'label', 'centroid':[3], 'total_score', 'size', 'means3d':[M,3], 'scores':[M] }
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(means3d)
    labels = db.labels_

    clusters = []
    for lbl in set(labels):
        if lbl == -1:
            continue   # noise
        mask = labels == lbl
        clusters.append({
            'label':       lbl,
            'centroid':    means3d[mask].mean(axis=0),
            'total_score': float(scores[mask].sum()),
            'size':        int(mask.sum()),
            'means3d':     means3d[mask],
            'scores':      scores[mask],
        })
    clusters.sort(key=lambda c: -c['total_score'])
    return clusters


def rank_clusters_by(clusters: list[dict], rank_by: str) -> None:
    """
    Re-order clusters after DBSCAN (or single-cluster mode).

    ``total`` — сумма cos×opacity по гауссианам (крупные объекты побеждают).
    ``mean`` — средний скор (лучше для «какой объект лучше всего подходит»).
    ``max`` — макс. скор в кластере (пик уверенности).
    """
    if rank_by == 'mean':
        clusters.sort(key=lambda c: -(c['total_score'] / max(1, c['size'])))
    elif rank_by == 'max':
        clusters.sort(key=lambda c: -float(np.max(c['scores'])))
    elif rank_by == 'total':
        clusters.sort(key=lambda c: -c['total_score'])
    else:
        raise ValueError(f'unknown cluster rank_by: {rank_by!r}')


# ---------------------------------------------------------------------------
# Best pose for a cluster centroid
# ---------------------------------------------------------------------------

def _cam_world_from_w2c(w2c) -> np.ndarray:
    """Camera centre in world coordinates from a 4×4 world-to-camera matrix.

    Uses ``c2w = inv(w2c)`` → ``t_world = c2w[:3, 3]`` so we match SplaTAM /
    OpenCV conventions without hand-derived ``-Rᵀt`` edge cases.
    """
    m = w2c.detach().cpu().numpy() if isinstance(w2c, torch.Tensor) else np.asarray(w2c, np.float64)
    c2w = np.linalg.inv(m)
    return c2w[:3, 3].astype(np.float64)


def _c2w_from_w2c(w2c) -> np.ndarray:
    m = w2c.detach().cpu().numpy() if isinstance(w2c, torch.Tensor) else np.asarray(w2c, np.float64)
    return np.linalg.inv(m)


def _print_pose_from_keyframe_json(
    poses_path: Path,
    poses_raw: dict,
    cluster_name: str,
    fid: int,
    w2c: torch.Tensor,
) -> None:
    """
    Prove the render pose is exactly the matrix stored under this frame_id
    in keyframe_poses.json (no checkpoint substitution).
    """
    key = str(fid)
    if key not in poses_raw:
        key = fid  # type: ignore[assignment]
    if key not in poses_raw:
        print(
            f'  [{cluster_name}] ERROR: frame_id={fid} not found in JSON keys. '
            f'Available sample: {list(poses_raw.keys())[:8]}...'
        )
        return

    raw = poses_raw[key]
    w2c_np = w2c.detach().cpu().numpy().reshape(4, 4)
    raw_np = np.asarray(raw, dtype=np.float64).reshape(4, 4)
    err = float(np.max(np.abs(w2c_np.astype(np.float64) - raw_np)))

    print(f'  --- {cluster_name}: camera pose from keyframe JSON ---')
    print(f'  File: {poses_path.resolve()}')
    print(f'  JSON key: {key!r}  |  max|w2c_tensor − JSON| = {err:.2e}  (expect 0)')
    print('  w2c (world → camera), same as in file:')
    for r in range(4):
        print('   ', ' '.join(f'{w2c_np[r, c]:14.8f}' for c in range(4)))
    print(f'  cam_center (world): {_cam_world_from_w2c(w2c)}')


def _print_auto_pose(cluster_name: str, fid: int, w2c: torch.Tensor) -> None:
    """Log synthetic / auto-sampled camera (no JSON file)."""
    w2c_np = w2c.detach().cpu().numpy().reshape(4, 4)
    print(f'  --- {cluster_name}: auto-sampled camera (no keyframes JSON) ---')
    print(f'  synthetic id={fid}  cam_center (world)={_cam_world_from_w2c(w2c)}')
    print('  w2c (world → camera):')
    for r in range(4):
        print('   ', ' '.join(f'{w2c_np[r, c]:14.8f}' for c in range(4)))


def _camera_gravity_alignment(w2c) -> float:
    """
    Y-up world, OpenCV camera: column 1 of c2w is camera +Y (image down) in world.
    Upright view: image-down ≈ −world_up ⇒ dot(cam_down, up) ≈ −1.
    Returns score in [-1, 1], **higher = more upright** (≈ +1), inverted ≈ −1.
    """
    c2w = _c2w_from_w2c(w2c)
    cam_down = c2w[:3, 1].astype(np.float64)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return float(-np.dot(cam_down, world_up))


def _cam_inside_aabb(cam_world: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return bool(np.all(cam_world >= lo) and np.all(cam_world <= hi))


def _gaussian_interior_aabb(
    means3d: np.ndarray,
    p_lo: float = 3.0,
    p_hi: float = 97.0,
    shrink_frac: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valid *interior* volume for cameras: bulk of reconstructed geometry (percentile
    hull), then shrunk **inward** so we do not pick views from the outer shell
    or outside the room-like extent of the Gaussians.

    Trajectory-only AABB is wrong here: SLAM can visit positions outside the
    furnished volume; Gaussian means approximate where the scene actually is.
    """
    lo = np.percentile(means3d, p_lo, axis=0)
    hi = np.percentile(means3d, p_hi, axis=0)
    span = np.maximum(hi - lo, 1e-4)
    lo_i = lo + shrink_frac * span
    hi_i = hi - shrink_frac * span
    if np.any(lo_i >= hi_i):
        # degenerate: fall back to slightly tighter percentiles, minimal shrink
        lo = np.percentile(means3d, 5.0, axis=0)
        hi = np.percentile(means3d, 95.0, axis=0)
        span = np.maximum(hi - lo, 1e-4)
        lo_i = lo + 0.03 * span
        hi_i = hi - 0.03 * span
    return lo_i.astype(np.float64), hi_i.astype(np.float64)


def _aabb_shrink_wall_margin(
    lo: np.ndarray,
    hi: np.ndarray,
    margin_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shrink an axis-aligned box **inward** by ``margin_m`` along each axis on
    both sides (minimum distance from any point inside to each of the six
    bounding planes of the **pre-shrink** box ≈ ``margin_m`` in metres).

    This approximates “at least ``margin_m`` from the walls” when the hull AABB
    bounds the room shell from Gaussian means.
    """
    if margin_m <= 0:
        return np.asarray(lo, dtype=np.float64).reshape(3), np.asarray(hi, dtype=np.float64).reshape(3)
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    m_req = float(margin_m)
    for frac in (1.0, 0.5, 0.25, 0.1, 0.05, 0.0):
        m = m_req * frac
        lo2 = lo + m
        hi2 = hi - m
        if np.all(lo2 < hi2):
            if frac < 1.0 and m > 0:
                print(
                    f'  [warn] wall margin reduced to {m:.3f} m per side '
                    f'(AABB too narrow for {m_req:.2f} m)'
                )
            elif m > 0:
                print(
                    f'  Wall margin: {m:.2f} m inward from Gaussian hull AABB '
                    f'(cameras stay ≥ this distance from hull “walls”)'
                )
            return lo2, hi2
    return lo.copy(), hi.copy()


def _scene_bounds_from_gaussians_only(
    means_all: np.ndarray,
    p_lo: float,
    p_hi: float,
    shrink_init: float,
    wall_margin_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interior AABB from Gaussian means only (no keyframe trajectory).
    Used when camera poses are auto-sampled instead of ``keyframes.json``.
    Finally applies ``wall_margin_m`` so samples are not flush on the hull.
    """
    for shrink in (shrink_init, 0.06, 0.03, 0.0):
        lo, hi = _gaussian_interior_aabb(
            means_all, p_lo=p_lo, p_hi=p_hi, shrink_frac=shrink,
        )
        span = hi - lo
        if np.all(span > 1e-4):
            print(
                f'  Camera interior (Gaussian-only p[{p_lo:g},{p_hi:g}], shrink={shrink}): '
                f'min={np.round(lo, 2)}  max={np.round(hi, 2)}'
            )
            return _aabb_shrink_wall_margin(lo, hi, wall_margin_m)

    lo, hi = _gaussian_interior_aabb(means_all, p_lo=0.5, p_hi=99.5, shrink_frac=0.0)
    print(
        f'  [warn] Gaussian-only bounds widened to p[0.5,99.5] shrink=0 → '
        f'min={np.round(lo, 2)}  max={np.round(hi, 2)}'
    )
    return _aabb_shrink_wall_margin(lo, hi, wall_margin_m)


def _w2c_look_at(
    eye: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """
    4×4 world-to-camera, OpenCV (X right, Y down, Z forward).

    ``eye``, ``target`` are length-3 world vectors (same units as the splat).
    """
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    z_axis = target - eye
    zn = float(np.linalg.norm(z_axis))
    if zn < 1e-5:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis = z_axis / zn
    x_axis = np.cross(world_up, z_axis)
    xn = float(np.linalg.norm(x_axis))
    if xn < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_axis = np.cross(world_up, z_axis)
        xn = float(np.linalg.norm(x_axis))
    x_axis = x_axis / max(xn, 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    yn = float(np.linalg.norm(y_axis))
    y_axis = y_axis / max(yn, 1e-12)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = eye
    return np.linalg.inv(c2w)


def _filter_camera_positions_dense_shell(
    means_all: np.ndarray,
    positions: np.ndarray,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None,
    rng: np.random.Generator,
    *,
    nn_k: int = 96,
    median_nn_schedule: tuple[float, ...] = (
        0.40, 0.48, 0.58, 0.72, 0.90, 1.15, 1.50, 2.0, 2.5, 3.0, 4.0,
    ),
) -> np.ndarray:
    """
    Keep only camera centres in ``scene_bounds`` (if given) with median k-NN
    distance to Gaussians below a metre-scale threshold (same idea as keyframes).
    """
    from scipy.spatial import cKDTree

    n_m = min(200_000, len(means_all))
    M = means_all[rng.choice(len(means_all), size=n_m, replace=False)]
    tree = cKDTree(M)
    k_query = min(nn_k, len(M))

    for thr in median_nn_schedule:
        ok_rows: list[int] = []
        for row in range(positions.shape[0]):
            cam = positions[row]
            if scene_bounds is not None:
                lo, hi = scene_bounds
                if np.any(cam < lo) or np.any(cam > hi):
                    continue
            dist, _ = tree.query(cam, k=k_query)
            if float(np.median(dist)) <= thr:
                ok_rows.append(row)
        if len(ok_rows) > 0:
            out = positions[np.array(ok_rows, dtype=np.int64)]
            print(
                f'  Auto pose filter: median kNN (k={k_query}) ≤ {thr:.2f} m '
                f'→ {len(out)}/{len(positions)} candidates (inside AABB + dense shell)'
            )
            return out

    only_aabb: list[int] = []
    for row in range(positions.shape[0]):
        cam = positions[row]
        if scene_bounds is None:
            only_aabb.append(row)
        else:
            lo, hi = scene_bounds
            if np.all(cam >= lo) and np.all(cam <= hi):
                only_aabb.append(row)
    if len(only_aabb) > 0:
        print('  [warn] auto poses: relaxed to AABB-only (all kNN thresholds failed)')
        return positions[np.array(only_aabb, dtype=np.int64)]
    return np.zeros((0, 3), dtype=np.float64)


def _sample_auto_camera_positions(
    means_all: np.ndarray,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None,
    rng: np.random.Generator,
    *,
    n_samples: int,
    nn_k: int,
    max_candidates: int,
    wall_margin_m: float = 0.0,
) -> np.ndarray:
    """
    Uniform samples in the interior AABB, then dense-shell filter and cap.

    If ``scene_bounds`` is None, builds hull from Gaussians then applies
    ``wall_margin_m``. If ``scene_bounds`` is already set (e.g. from
    ``_scene_bounds_from_gaussians_only``), pass ``wall_margin_m=0`` to avoid
    double shrinking.
    """
    if scene_bounds is None:
        lo, hi = _gaussian_interior_aabb(means3d=means_all, p_lo=3.0, p_hi=97.0, shrink_frac=0.12)
        lo, hi = _aabb_shrink_wall_margin(lo, hi, wall_margin_m)
    else:
        lo, hi = scene_bounds
    raw = rng.uniform(low=lo, high=hi, size=(n_samples, 3))
    filt = _filter_camera_positions_dense_shell(
        means_all, raw, scene_bounds, rng, nn_k=nn_k,
    )
    if filt.shape[0] == 0:
        return filt
    if filt.shape[0] > max_candidates:
        idx = rng.choice(filt.shape[0], size=max_candidates, replace=False)
        filt = filt[idx]
        print(f'  Auto poses: subsampled to {max_candidates} candidates (cap)')
    return filt.astype(np.float64)


def _synthetic_poses_look_at_centroid(
    positions: np.ndarray,
    centroid: np.ndarray,
    scene_bounds: tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """
    One w2c per row in ``positions``; each camera looks at ``centroid``, or at
    the AABB centre if the eye is too close to the centroid.
    """
    lo, hi = scene_bounds
    fallback = 0.5 * (lo + hi)
    c = np.asarray(centroid, dtype=np.float64).reshape(3)
    fb = np.asarray(fallback, dtype=np.float64).reshape(3)
    poses: dict[int, torch.Tensor] = {}
    pid = 0
    for row in range(positions.shape[0]):
        eye = positions[row]
        tgt = c
        if float(np.linalg.norm(eye - c)) < 0.04:
            tgt = fb
        w2c_np = _w2c_look_at(eye, tgt)
        poses[pid] = torch.tensor(w2c_np, dtype=torch.float32, device=device)
        pid += 1
    return poses


def _count_poses_inside(poses: dict, lo: np.ndarray, hi: np.ndarray) -> int:
    return sum(
        1 for w2c in poses.values()
        if _cam_inside_aabb(_cam_world_from_w2c(w2c), lo, hi)
    )


def _compute_allowed_pose_ids(
    poses: dict,
    means_all: np.ndarray,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None,
    rng: np.random.Generator,
    *,
    nn_k: int = 96,
    median_nn_schedule: tuple[float, ...] = (
        0.40, 0.48, 0.58, 0.72, 0.90, 1.15, 1.50, 2.0, 2.5, 3.0, 4.0,
    ),
) -> tuple[set[int], float]:
    """
    Keyframes whose camera centre lies in **dense geometry**, not empty space.

    A loose AABB around Gaussian percentiles still contains void outside the room;
    here we require the median distance to the ``nn_k`` nearest Gaussians (on a
    large subsample, KD-tree) to stay below a metre-scale threshold — cameras
    floating outside the splat shell get large median NN distances and are dropped.
    """
    from scipy.spatial import cKDTree

    n_m = min(200_000, len(means_all))
    M = means_all[rng.choice(len(means_all), size=n_m, replace=False)]
    tree = cKDTree(M)
    k_query = min(nn_k, len(M))

    def _cam_ok_for_threshold(max_median: float) -> set[int]:
        ok: set[int] = set()
        for fid, w2c in poses.items():
            cam = _cam_world_from_w2c(w2c)
            if scene_bounds is not None:
                lo, hi = scene_bounds
                if np.any(cam < lo) or np.any(cam > hi):
                    continue
            dist, _ = tree.query(cam, k=k_query)
            if float(np.median(dist)) <= max_median:
                ok.add(fid)
        return ok

    for thr in median_nn_schedule:
        s = _cam_ok_for_threshold(thr)
        if len(s) > 0:
            print(
                f'  Geometry pose filter: median kNN (k={k_query}) ≤ {thr:.2f} m '
                f'→ {len(s)}/{len(poses)} keyframes (inside AABB + dense shell)'
            )
            return s, thr

    # AABB only — still exclude cameras outside the box
    only_aabb: set[int] = set()
    for fid, w2c in poses.items():
        cam = _cam_world_from_w2c(w2c)
        if scene_bounds is None:
            only_aabb.add(fid)
        else:
            lo, hi = scene_bounds
            if np.all(cam >= lo) and np.all(cam <= hi):
                only_aabb.add(fid)
    if len(only_aabb) > 0:
        print('  [warn] relaxed pose filter to AABB-only (all kNN thresholds failed)')
        return only_aabb, float('nan')
    print(
        '  [error] no keyframe camera centre inside scene AABB — '
        'refusing to fall back to arbitrary poses (would pick views outside the scene).'
    )
    return set(), float('nan')


def _resolve_scene_interior_bounds(
    means_all: np.ndarray,
    poses: dict,
    p_lo: float,
    p_hi: float,
    shrink_init: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-based interior AABB with automatic relaxation until at least one
    keyframe camera lies inside (so ``best_pose_for_centroid`` can succeed).
    """
    for shrink in (shrink_init, 0.06, 0.03, 0.0):
        lo, hi = _gaussian_interior_aabb(
            means_all, p_lo=p_lo, p_hi=p_hi, shrink_frac=shrink,
        )
        n_in = _count_poses_inside(poses, lo, hi)
        if n_in > 0:
            print(
                f'  Camera interior (Gaussian p[{p_lo:g},{p_hi:g}], shrink={shrink}): '
                f'{n_in}/{len(poses)} keyframes inside  '
                f'min={np.round(lo, 2)}  max={np.round(hi, 2)}'
            )
            return lo, hi

    lo, hi = _gaussian_interior_aabb(means_all, p_lo=0.5, p_hi=99.5, shrink_frac=0.0)
    n_in = _count_poses_inside(poses, lo, hi)
    print(
        f'  [warn] widened to p[0.5,99.5] shrink=0 → {n_in}/{len(poses)} keyframes inside scene hull'
    )
    print(f'    AABB min={np.round(lo, 2)}  max={np.round(hi, 2)}')
    if n_in == 0:
        print(
            '  [error] No keyframe camera lies inside the Gaussian hull. '
            'Check that keyframe_poses.json matches this splat checkpoint / coordinate frame.'
        )
    return lo, hi


def best_pose_for_centroid(
    centroid: np.ndarray,          # [3] world
    poses: dict[int, torch.Tensor],
    intrinsics: torch.Tensor,      # [3, 3]
    H: int,
    W: int,
    device: torch.device,
    *,
    margin_px: float = 12.0,
    z_min: float = 0.2,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    require_cam_in_scene: bool = True,
    allowed_pose_ids: set[int] | None = None,
    exclude_fids: set[int] | None = None,
    min_gravity: float | None = None,
) -> tuple[int | None, torch.Tensor | None]:
    """
    Pick a keyframe where the cluster centroid is visible and the view is usable.

    When ``require_cam_in_scene`` and ``scene_bounds`` are set (default), **only**
    poses whose camera centre lies inside ``scene_bounds`` are considered — no
    fallback to cameras outside the volume (that was causing views from outside
    the reconstructed scene).

    ``allowed_pose_ids`` (optional) further restricts to keyframes that passed the
    KD-tree median kNN test — cameras in empty space outside the splat are removed.
    An **empty** set means no pose is acceptable (e.g. failed geometry filter).

    Sub-stages inside that constraint: margin frustum → loose frustum → any
    in-front view with centroid closest to image centre.

    ``exclude_fids``: do not reuse these frame ids (other clusters).
    ``min_gravity``: if set, drop poses with ``_camera_gravity_alignment`` below
    this (filters upside-down / rolled views in Y-up scenes).
    """
    if allowed_pose_ids is not None and len(allowed_pose_ids) == 0:
        return None, None

    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()
    cx_img = intrinsics[0, 2].item()
    cy_img = intrinsics[1, 2].item()

    pt = torch.tensor([*centroid, 1.0], dtype=torch.float32, device=device)  # [4]
    excl = exclude_fids or set()

    def try_stage(
        use_aabb: bool,
        in_frustum: str,
        gravity_min: float | None,
    ) -> tuple[int | None, torch.Tensor | None, float]:
        """in_frustum: 'margin' | 'loose' | 'none'."""
        best_fid, best_w2c, best_dist2 = None, None, float('inf')
        for fid, w2c in poses.items():
            if fid in excl:
                continue
            if allowed_pose_ids is not None and fid not in allowed_pose_ids:
                continue
            if gravity_min is not None:
                if _camera_gravity_alignment(w2c) < gravity_min:
                    continue
            if use_aabb and scene_bounds is not None:
                cw = _cam_world_from_w2c(w2c)
                lo, hi = scene_bounds
                if np.any(cw < lo) or np.any(cw > hi):
                    continue

            cam = w2c @ pt
            z = cam[2].item()
            if z < z_min:
                continue
            x_proj = cam[0].item() / z * fx + cx_img
            y_proj = cam[1].item() / z * fy + cy_img

            if in_frustum == 'margin':
                m = margin_px
                if not (m <= x_proj < W - m and m <= y_proj < H - m):
                    continue
            elif in_frustum == 'loose':
                if not (0.0 < x_proj < W - 1.0 and 0.0 < y_proj < H - 1.0):
                    continue
            # 'none': only z check

            dist2 = (x_proj - cx_img) ** 2 + (y_proj - cy_img) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_fid = fid
                best_w2c = w2c
        return best_fid, best_w2c, best_dist2

    use_aabb = bool(require_cam_in_scene and scene_bounds is not None)

    def _run_all_stages(gmin: float | None) -> tuple[int | None, torch.Tensor | None]:
        if use_aabb:
            fid, w2c, _ = try_stage(True, 'margin', gmin)
            if fid is not None:
                return fid, w2c
            fid, w2c, _ = try_stage(True, 'loose', gmin)
            if fid is not None:
                return fid, w2c
            fid, w2c, _ = try_stage(True, 'none', gmin)
            return fid, w2c
        fid, w2c, _ = try_stage(False, 'margin', gmin)
        if fid is not None:
            return fid, w2c
        fid, w2c, _ = try_stage(False, 'loose', gmin)
        if fid is not None:
            return fid, w2c
        fid, w2c, _ = try_stage(False, 'none', gmin)
        return fid, w2c

    fid, w2c = _run_all_stages(min_gravity)
    if fid is not None:
        return fid, w2c
    if min_gravity is not None:
        fid, w2c = _run_all_stages(None)
    return fid, w2c


def _render_raw_cosine_map(
    model: LangSplatam,
    clip_query: torch.Tensor,
    ae: "Autoencoder | None",
    w2c: torch.Tensor,
    H: int,
    W: int,
    device: torch.device,
    blur_sigma: float,
    *,
    use_v2: bool = False,
) -> np.ndarray:
    """Per-pixel raw cosine(decoded latent, CLIP query); used to rank camera poses by query."""
    batch = 4096
    q = clip_query.to(device).to(torch.float32)
    with torch.no_grad():
        if use_v2:
            rendered512 = model.render_v2_clip_feature_map(w2c, H, W)
            rendered512 = F.normalize(rendered512, p=2, dim=0)
            r_flat = rendered512.permute(1, 2, 0).reshape(-1, 512)
        else:
            assert ae is not None
            rendered = model.render_lang(w2c, H, W)
            D = rendered.shape[0]
            r_flat = rendered.permute(1, 2, 0).reshape(-1, D)
        sim_flat = torch.empty((r_flat.shape[0],), device=r_flat.device, dtype=torch.float32)
        if use_v2:
            for i in range(0, r_flat.shape[0], batch):
                sim_flat[i : i + batch] = r_flat[i : i + batch] @ q
        else:
            assert ae is not None
            for i in range(0, r_flat.shape[0], batch):
                c = ae.decode(r_flat[i : i + batch])
                c = F.normalize(c, p=2, dim=-1).to(torch.float32)  # [B,512]
                sim_flat[i : i + c.shape[0]] = (c @ q)
    sim = sim_flat.reshape(H, W)
    sim_np = sim.cpu().float().numpy()
    if blur_sigma > 0.0:
        ksize = int(blur_sigma * 6) | 1
        sim_np = cv2.GaussianBlur(sim_np, (ksize, ksize), blur_sigma)
    return sim_np


def _patch_mean(sim_np: np.ndarray, xi: int, yi: int, patch_radius: int) -> float:
    H, W = sim_np.shape
    y0, y1 = max(0, yi - patch_radius), min(H, yi + patch_radius + 1)
    x0, x1 = max(0, xi - patch_radius), min(W, xi + patch_radius + 1)
    return float(np.mean(sim_np[y0:y1, x0:x1]))


def _score_sim_map(
    sim_np: np.ndarray,
    score_mode: str,
    *,
    xi: int,
    yi: int,
    patch_radius: int,
    top_frac: float,
) -> float:
    """
    Скаляр для ранжирования позы по карте сырого cos (после blur как в рендере).

    ``centroid_patch`` — среднее в окне у проекции 3D-центроида (старое поведение).
    ``global_mean`` — среднее по всему кадру.
    ``global_topmean`` — среднее по верхнему ``top_frac`` пикселей (устойчиво к фону).
    """
    if score_mode == 'centroid_patch':
        return _patch_mean(sim_np, xi, yi, patch_radius)
    if score_mode == 'global_mean':
        return float(np.mean(sim_np))
    if score_mode == 'global_topmean':
        flat = sim_np.astype(np.float64, copy=False).ravel()
        k = max(1, int(round(float(top_frac) * flat.size)))
        top = np.partition(flat, -k)[-k:]
        return float(np.mean(top))
    raise ValueError(f'unknown pose_score_mode: {score_mode!r}')


def _centroid_in_front_only(
    centroid: np.ndarray,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    device: torch.device,
    z_min: float,
) -> tuple[bool, float, int, int]:
    """Центроид перед камерой (z>=z_min); xi,yi — проекция (могут быть вне кадра)."""
    pt = torch.tensor([*centroid, 1.0], dtype=torch.float32, device=device)
    cam = w2c @ pt
    z = cam[2].item()
    if z < z_min:
        return False, 0.0, 0, 0
    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()
    cx_img = intrinsics[0, 2].item()
    cy_img = intrinsics[1, 2].item()
    x_proj = cam[0].item() / z * fx + cx_img
    y_proj = cam[1].item() / z * fy + cy_img
    dist2 = (x_proj - cx_img) ** 2 + (y_proj - cy_img) ** 2
    xi = int(round(x_proj))
    yi = int(round(y_proj))
    return True, float(dist2), xi, yi


def best_pose_by_query_relevancy(
    centroid: np.ndarray,
    poses: dict[int, torch.Tensor],
    intrinsics: torch.Tensor,
    H: int,
    W: int,
    device: torch.device,
    model: LangSplatam,
    ae: "Autoencoder | None",
    clip_query: torch.Tensor,
    *,
    use_v2: bool = False,
    margin_px: float,
    z_min: float,
    scene_bounds: tuple[np.ndarray, np.ndarray] | None,
    require_cam_in_scene: bool,
    allowed_pose_ids: set[int] | None,
    blur_sigma: float,
    patch_radius: int,
    score_mode: str = 'global_topmean',
    top_frac: float = 0.05,
    exclude_fids: set[int] | None = None,
    min_gravity: float | None = 0.12,
) -> tuple[int | None, torch.Tensor | None]:
    """
    Среди поз, где центроид кластера перед камерой (и для ``centroid_patch`` ещё и
    внутри поля зрения с отступом), выбираем кадр с максимальным скором по карте
    ``cos(decoded latent, CLIP text)`` (см. ``score_mode``).

    По умолчанию ``global_topmean``: сравниваем верхнюю долю пикселей кадра — так
    выбор устойчивее, чем одно окно у проекции центроида, если 3D-центроид промахивается.

    ``exclude_fids``: не переиспользовать эти frame_id (другие кластеры).

    ``min_gravity``: отсекаем перевёрнутые камеры (Y-up); при отсутствии кандидатов ослабляем.
    """
    if allowed_pose_ids is not None and len(allowed_pose_ids) == 0:
        return None, None

    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()
    cx_img = intrinsics[0, 2].item()
    cy_img = intrinsics[1, 2].item()
    pt = torch.tensor([*centroid, 1.0], dtype=torch.float32, device=device)
    excl = exclude_fids or set()

    def _geo_ok(w2c: torch.Tensor) -> tuple[bool, float, float, float, int, int]:
        """Return (ok, x_proj, y_proj, dist2, xi, yi)."""
        cam = w2c @ pt
        z = cam[2].item()
        if z < z_min:
            return False, 0.0, 0.0, 0.0, 0, 0
        x_proj = cam[0].item() / z * fx + cx_img
        y_proj = cam[1].item() / z * fy + cy_img
        m = margin_px
        if not (m <= x_proj < W - m and m <= y_proj < H - m):
            return False, 0.0, 0.0, 0.0, 0, 0
        dist2 = (x_proj - cx_img) ** 2 + (y_proj - cy_img) ** 2
        xi = int(round(x_proj))
        yi = int(round(y_proj))
        return True, x_proj, y_proj, dist2, xi, yi

    sim_cache: dict[int, np.ndarray] = {}

    def _collect_scored(
        use_exclude: bool,
        gravity_floor: float | None,
    ) -> list[tuple[float, float, float, int, torch.Tensor]]:
        """List of (sc, grav, dist2, fid, w2c)."""
        out: list[tuple[float, float, float, int, torch.Tensor]] = []
        xex = excl if use_exclude else set()
        for fid, w2c in poses.items():
            if allowed_pose_ids is not None and fid not in allowed_pose_ids:
                continue
            if fid in xex:
                continue
            if require_cam_in_scene and scene_bounds is not None:
                cw = _cam_world_from_w2c(w2c)
                lo, hi = scene_bounds
                if np.any(cw < lo) or np.any(cw > hi):
                    continue

            grav = _camera_gravity_alignment(w2c)
            if gravity_floor is not None and grav < gravity_floor:
                continue

            if score_mode == 'centroid_patch':
                ok, _xp, _yp, dist2, xi, yi = _geo_ok(w2c)
                if not ok:
                    continue
            else:
                ok_front, dist2, xi, yi = _centroid_in_front_only(
                    centroid, w2c, intrinsics, device, z_min,
                )
                if not ok_front:
                    continue

            if fid not in sim_cache:
                sim_cache[fid] = _render_raw_cosine_map(
                    model, clip_query, ae, w2c, H, W, device, blur_sigma,
                    use_v2=use_v2,
                )
            sim_np = sim_cache[fid]
            sc = _score_sim_map(
                sim_np, score_mode,
                xi=xi, yi=yi, patch_radius=patch_radius, top_frac=top_frac,
            )
            out.append((sc, grav, dist2, fid, w2c))
        return out

    # Relaxation: (use_exclude_fids, gravity_floor). Prefer other clusters' frames + upright first.
    relax_plan: list[tuple[bool, float | None]] = [
        (True, min_gravity),
        (True, None),
        (False, min_gravity),
        (False, None),
    ]

    for use_excl, gfloor in relax_plan:
        scored = _collect_scored(use_excl, gfloor)
        if not scored:
            continue
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        _sc, _gv, _d2, best_fid, best_w2c = scored[0]
        return best_fid, best_w2c

    return best_pose_for_centroid(
        centroid, poses, intrinsics, H, W, device,
        margin_px=margin_px,
        z_min=z_min,
        scene_bounds=scene_bounds,
        require_cam_in_scene=require_cam_in_scene,
        allowed_pose_ids=allowed_pose_ids,
        exclude_fids=exclude_fids,
        min_gravity=min_gravity,
    )


# ---------------------------------------------------------------------------
# Rendering (identical to SplaTAM eval)
# ---------------------------------------------------------------------------

def render_rgb(model: LangSplatam, w2c: torch.Tensor, H: int, W: int) -> np.ndarray:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_parties', 'splatam'))
    from utils.slam_helpers import transformed_params2rendervar
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer

    cam = model._setup_camera(H, W, w2c)
    tr  = model._transform_gaussians(w2c)
    rv  = transformed_params2rendervar(model.params, tr)
    with torch.no_grad():
        rgb, _, _ = Renderer(raster_settings=cam)(**rv)
    img = rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _normalize_relevancy_map(
    sim_np: np.ndarray,
    mode: str,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    """
    Map raw cosine similarities to [0, 1] for visualization.

    ``percentile`` (default): stretch only the top part of the distribution
    (p_low … p_high); avoids turning global noise into full-range red/yellow.
    """
    if mode == 'minmax':
        lo, hi = float(sim_np.min()), float(sim_np.max())
        return np.clip((sim_np - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    lo = np.percentile(sim_np, p_low)
    hi = np.percentile(sim_np, p_high)
    if hi <= lo + 1e-8:
        return np.zeros_like(sim_np)
    return np.clip((sim_np - lo) / (hi - lo), 0.0, 1.0)


def render_relevancy_map(
    model: LangSplatam,
    clip_query: torch.Tensor,    # [512] unit-norm CLIP text emb
    ae: "Autoencoder | None",
    w2c: torch.Tensor,
    H: int,
    W: int,
    *,
    use_v2: bool = False,
    heatmap_norm: str = 'percentile',
    heatmap_p_low: float = 8.0,
    heatmap_p_high: float = 98.0,
    blur_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render D-dim latent field, decode each pixel to 512d, compute cosine
    with CLIP text embedding.  Comparing in 512d (not latent) is more
    discriminative because the AE may distort relative distances.

    ``blur_sigma`` applies Gaussian blur to the raw similarity map before
    normalization — this suppresses isolated speckle from alpha-composited
    multi-Gaussian pixels while keeping spatially-coherent object blobs.

    Returns (jet heatmap BGR, float32 [H,W] normalized to [0,1] for viz,
    float32 [H,W] raw cosine after blur — для SAM/пика relevancy).
    """
    batch = 4096
    with torch.no_grad():
        if use_v2:
            rendered512 = model.render_v2_clip_feature_map(w2c, H, W)
            rendered512 = F.normalize(rendered512, p=2, dim=0)
            r_flat = rendered512.permute(1, 2, 0).reshape(-1, 512)
        else:
            assert ae is not None
            rendered = model.render_lang(w2c, H, W)          # [D, H, W]
            D = rendered.shape[0]
            r_flat = rendered.permute(1, 2, 0).reshape(-1, D)    # [H*W, D]

        q = clip_query.to(r_flat.device).to(torch.float32)
        sim_flat = torch.empty((r_flat.shape[0],), device=r_flat.device, dtype=torch.float32)
        if use_v2:
            for i in range(0, r_flat.shape[0], batch):
                sim_flat[i : i + batch] = r_flat[i : i + batch] @ q
        else:
            assert ae is not None
            for i in range(0, r_flat.shape[0], batch):
                c = ae.decode(r_flat[i : i + batch])
                c = F.normalize(c, p=2, dim=-1).to(torch.float32)  # [B,512]
                sim_flat[i : i + c.shape[0]] = (c @ q)
    sim = sim_flat.reshape(H, W)                         # [H, W]
    sim_np = sim.cpu().float().numpy()

    # Spatial blur to suppress single-Gaussian speckle
    if blur_sigma > 0.0:
        ksize = int(blur_sigma * 6) | 1   # always odd
        sim_np = cv2.GaussianBlur(sim_np, (ksize, ksize), blur_sigma)

    norm = _normalize_relevancy_map(sim_np, heatmap_norm, heatmap_p_low, heatmap_p_high)
    jet = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return jet, norm, sim_np.astype(np.float32)


# ---------------------------------------------------------------------------
# Cluster mask on RGB (semantic)
# ---------------------------------------------------------------------------

def project_centroid(centroid: np.ndarray,    # [3] world
                     w2c: torch.Tensor,        # [4,4]
                     intrinsics: torch.Tensor, # [3,3]
                     device: torch.device,
                     ) -> tuple[int, int] | None:
    """Project 3-D centroid → (x_px, y_px) or None if behind camera."""
    pt = torch.tensor([*centroid, 1.0], dtype=torch.float32, device=device)
    cam = w2c @ pt
    z = cam[2].item()
    if z < 0.2:
        return None
    fx = intrinsics[0, 0].item()
    fy = intrinsics[1, 1].item()
    cx = intrinsics[0, 2].item()
    cy = intrinsics[1, 2].item()
    x = int(round(cam[0].item() / z * fx + cx))
    y = int(round(cam[1].item() / z * fy + cy))
    return x, y


def sam_prompt_xy_from_relevancy(
    sim_raw: np.ndarray,
    mode: str,
    centroid_xy: tuple[int, int] | None,
    H: int,
    W: int,
    *,
    com_top_frac: float = 0.12,
) -> tuple[int, int] | None:
    """
    Точка для SAM: проекция 3D-центроида или пик / центр масс по сырой карте cos
    (после blur, как в ``render_relevancy_map``).
    """
    if mode == 'centroid':
        if centroid_xy is None:
            mode = 'relevancy_com'
        else:
            x, y = centroid_xy
            return max(0, min(W - 1, int(x))), max(0, min(H - 1, int(y)))
    if mode == 'relevancy_peak':
        yi, xi = np.unravel_index(np.argmax(sim_raw), sim_raw.shape)
        return max(0, min(W - 1, int(xi))), max(0, min(H - 1, int(yi)))
    if mode == 'relevancy_com':
        flat = sim_raw.ravel()
        k = max(1, int(round(float(com_top_frac) * flat.size)))
        thr = np.partition(flat.astype(np.float64, copy=False), -k)[-k]
        mask = sim_raw >= thr
        ys, xs = np.where(mask)
        if len(ys) == 0:
            yi, xi = np.unravel_index(np.argmax(sim_raw), sim_raw.shape)
        else:
            w = sim_raw[ys, xs].astype(np.float64)
            xi = int(np.round(np.average(xs, weights=w)))
            yi = int(np.round(np.average(ys, weights=w)))
        return max(0, min(W - 1, int(xi))), max(0, min(H - 1, int(yi)))
    raise ValueError(f'unknown sam_prompt_from: {mode!r}')


# SAM predictor is heavy — load it once and reuse
_sam_predictor = None

def _get_sam_predictor(sam_ckpt: str, device: torch.device):
    global _sam_predictor
    if _sam_predictor is None:
        from segment_anything import sam_model_registry, SamPredictor
        name = Path(sam_ckpt).name
        if 'vit_h' in name:
            model_type = 'vit_h'
        elif 'vit_l' in name:
            model_type = 'vit_l'
        else:
            model_type = 'vit_b'
        sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
        sam.to(device)
        _sam_predictor = SamPredictor(sam)
        print(f'SAM loaded: {model_type} from {sam_ckpt}')
    return _sam_predictor


def sam_all_masks(rgb_bgr: np.ndarray,   # H×W×3 uint8 BGR
                  sam_ckpt: str,
                  device: torch.device,
                  ) -> list[np.ndarray]:
    """
    Run SAM automatic mask generator on the full image.
    Returns list of binary uint8 masks [H, W] sorted by area (largest first).
    """
    from segment_anything import SamAutomaticMaskGenerator
    predictor = _get_sam_predictor(sam_ckpt, device)
    generator = SamAutomaticMaskGenerator(predictor.model)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    results = generator.generate(rgb)
    results.sort(key=lambda r: -r['area'])
    return [(r['segmentation'].astype(np.uint8) * 255) for r in results]


def sam_mask_from_point(rgb_bgr: np.ndarray,    # H×W×3 uint8 BGR
                        point_xy: tuple[int, int],
                        sam_ckpt: str,
                        device: torch.device,
                        ) -> np.ndarray:
    """
    Use SAM with a single point prompt to segment the object under `point_xy`.
    Returns binary mask uint8 [H, W] (255 = object).
    """
    predictor = _get_sam_predictor(sam_ckpt, device)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    pts   = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    lbls  = np.array([1], dtype=np.int32)   # 1 = foreground
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbls,
        multimask_output=True,
    )
    # Pick the mask with the highest score
    best = masks[scores.argmax()]          # bool [H, W]
    return (best.astype(np.uint8) * 255)


def semantic_overlay(rgb_bgr: np.ndarray, mask: np.ndarray,
                     point_xy: tuple[int, int] | None = None,
                     color_bgr=(0, 255, 0)) -> np.ndarray:
    """Alpha-blend SAM mask onto RGB and draw contour + prompt point."""
    out = rgb_bgr.copy()
    color = np.zeros_like(out, dtype=np.uint8)
    color[:] = np.array(color_bgr, dtype=np.uint8)

    alpha = (mask > 0).astype(np.float32)[:, :, None]
    out_f = out.astype(np.float32)
    out_f = out_f * (1.0 - 0.5 * alpha) + color.astype(np.float32) * (0.5 * alpha)
    out = out_f.clip(0, 255).astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 2)

    if point_xy is not None:
        cv2.circle(out, point_xy, 8, (0, 0, 255), -1)   # red dot = SAM prompt
        cv2.circle(out, point_xy, 8, (255, 255, 255), 2)

    return out


# ---------------------------------------------------------------------------
# 3-D visualisation
# ---------------------------------------------------------------------------

def save_cluster_map(all_means: np.ndarray, clusters: list[dict],
                     cam_positions: np.ndarray, best_cam_pos: np.ndarray,
                     out_path: Path, text: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Clusters for query: "{text}"', fontsize=13)

    colors_map = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))

    for ax, (ix, iy, xl, yl, title) in zip(axes, [
        (0, 2, 'X', 'Z (forward)', 'Top-down  X–Z'),
        (0, 1, 'X', 'Y (down)',    'Front     X–Y'),
    ]):
        # scene background
        idx = np.random.choice(len(all_means), size=min(30_000, len(all_means)), replace=False)
        ax.scatter(all_means[idx, ix], all_means[idx, iy],
                   s=0.2, c='lightgrey', alpha=0.25, zorder=1)

        # clusters
        for ci, cl in enumerate(clusters[:8]):
            c = colors_map[ci]
            ax.scatter(cl['means3d'][:, ix], cl['means3d'][:, iy],
                       s=6, color=c, alpha=0.7, zorder=3,
                       label=f'cluster {ci+1} (n={cl["size"]} score={cl["total_score"]:.0f})')
            ax.scatter([cl['centroid'][ix]], [cl['centroid'][iy]],
                       s=200, color=c, marker='X', zorder=5, edgecolors='black', linewidths=0.5)

        # cameras
        ax.scatter(cam_positions[:, ix], cam_positions[:, iy],
                   s=15, c='green', marker='^', alpha=0.4, zorder=4, label='cameras')
        ax.scatter([best_cam_pos[ix]], [best_cam_pos[iy]],
                   s=250, c='red', marker='*', zorder=6, label='best camera')

        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        ax.legend(loc='upper right', fontsize=6)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Cluster map → {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',        required=True)
    p.add_argument('--lang_field',        required=True)
    p.add_argument('--latent_dim',        type=int, default=None,
                   help='Размер латента (должен совпадать с обучением). '
                        'По умолчанию читается из lang_field.pt.')
    p.add_argument(
        '--poses',
        default=None,
        help='JSON {frame_id: [[4×4]]} w2c — как раньше. '
             'Если **не указать**, позы сэмплируются внутри объёма сцены по гауссианам '
             '(без keyframes); камера смотрит на центроид кластера.',
    )
    p.add_argument(
        '--traj_txt',
        default=None,
        help='Путь к traj.txt (каждая строка = 16 чисел, 4x4). '
             'Если задано, используется ТОЛЬКО этот набор поз (игнорирует --poses и автопозы).',
    )
    p.add_argument(
        '--traj_format',
        choices=('c2w', 'w2c'),
        default='c2w',
        help='Как интерпретировать матрицы в traj.txt. По умолчанию c2w (инвертируется в w2c).',
    )
    p.add_argument(
        '--auto_pose_samples',
        type=int,
        default=4096,
        help='Сколько случайных точек в AABB испытать до фильтра плотности (только без --poses).',
    )
    p.add_argument(
        '--auto_pose_max_candidates',
        type=int,
        default=96,
        help='Максимум поз-кандидатов после фильтра (меньше = быстрее --pose_select relevancy).',
    )
    p.add_argument(
        '--auto_pose_seed',
        type=int,
        default=0,
        help='RNG для автопоз.',
    )
    p.add_argument('--text',              required=True)
    p.add_argument(
        '--ae_ckpt',
        default=None,
        help='Путь к чекпойнту автоэнкодера (legacy lang_field). '
             'Для LangSplatV2 (format=langsplatv2) не нужен — можно опустить.',
    )
    p.add_argument('--clip_model',        default='ViT-B-16')
    p.add_argument('--clip_pretrained',   default='laion2b_s34b_b88k')
    p.add_argument('--device',            default='cuda:0')
    p.add_argument('--top_percentile',    type=float, default=2.0,
                   help='Top-%%  of Gaussians by score to cluster (default 2%%)')
    p.add_argument('--dbscan_eps',        type=float, default=0.15,
                   help='DBSCAN eps in scene units (default 0.15 m)')
    p.add_argument('--dbscan_min',        type=int,   default=30,
                   help='DBSCAN min_samples (default 30)')
    p.add_argument('--no_clusters', action='store_true',
                   help='Отключить DBSCAN и использовать один кластер из всего top-% набора.')
    p.add_argument(
        '--cluster_rank_by',
        choices=('total', 'mean', 'max'),
        default='mean',
        help='Как упорядочить кластеры после DBSCAN: total — сумма скоров (крупные объекты), '
             'mean — средний скор (лучше для «самого подходящего» объекта), '
             'max — пиковый скор в кластере.',
    )
    p.add_argument(
        '--gaussian_score_no_opacity',
        action='store_true',
        help='Ранжировать гауссианы по сырому cos(CLIP) без умножения на opacity '
             '(иначе крупные полупрозрачные облака могут доминировать в top-%%).',
    )
    p.add_argument('--top_k_views',       type=int,   default=3,
                   help='Save renders for top-K clusters (default 3)')
    p.add_argument('--sam_ckpt',          default='ckpts/sam_vit_b_01ec64.pth',
                   help='SAM checkpoint for semantic mask (default: vit_b)')
    p.add_argument(
        '--sam_prompt_from',
        choices=('centroid', 'relevancy_peak', 'relevancy_com'),
        default='relevancy_com',
        help='Точка для SAM: проекция 3D-центроида или пик / центр масс по карте relevancy '
             '(устойчивее, чем одна проекция центроида).',
    )
    p.add_argument(
        '--sam_com_top_frac',
        type=float,
        default=0.12,
        help='Для --sam_prompt_from relevancy_com: верхняя доля пикселей по cos для центра масс.',
    )
    p.add_argument('--encoder_dims', nargs='+', type=int, default=None,
                   help='Размерности энкодера AE (должны совпадать с train_language_autoencoder.py). '
                        'None = использовать дефолты класса Autoencoder (64d).')
    p.add_argument('--decoder_dims', nargs='+', type=int, default=None,
                   help='Размерности декодера AE. None = использовать дефолты (64d).')
    p.add_argument('--pose_margin_px', type=float, default=12.0,
                   help='Центроид должен проецироваться не ближе этого отступа от края кадра.')
    p.add_argument('--pose_z_min', type=float, default=0.2,
                   help='Мин. глубина центроида в координатах камеры.')
    p.add_argument('--no_pose_scene_bounds', action='store_true',
                   help='Отключить ограничение «камера внутри объёма сцены» (только отладка).')
    p.add_argument('--pose_interior_shrink', type=float, default=0.12,
                   help='На сколько сузить перцентильный hull гауссианов с каждой стороны (0…0.25).')
    p.add_argument('--pose_percentile_lo', type=float, default=3.0,
                   help='Нижний перцентиль по means3D для объёма сцены.')
    p.add_argument('--pose_percentile_hi', type=float, default=97.0,
                   help='Верхний перцентиль по means3D для объёма сцены.')
    p.add_argument('--pose_nn_k', type=int, default=96,
                   help='Сколько ближайших гауссиан для медианного расстояния (KD-tree).')
    p.add_argument(
        '--pose_wall_margin_m',
        type=float,
        default=0.5,
        help='Для AABB по гауссианам: дополнительное сужение внутрь на столько метров '
             'с каждой стороны коробки (мин. расстояние до «стен» hull). По умолчанию 0.5 м.',
    )
    p.add_argument(
        '--pose_select',
        choices=('relevancy', 'centroid'),
        default='relevancy',
        help='Как выбрать позу для рендера кластера: relevancy — по карте cos (см. '
             '--pose_score_mode); centroid — только геометрия (центроид ближе к центру кадра).',
    )
    p.add_argument(
        '--pose_score_mode',
        choices=('centroid_patch', 'global_mean', 'global_topmean'),
        default='global_topmean',
        help='При --pose_select relevancy: как агрегировать карту cos по кадру. '
             'global_topmean (по умолчанию) — среднее по верхнему %% пикселей (устойчиво); '
             'centroid_patch — окно у проекции 3D-центроида (старое поведение).',
    )
    p.add_argument(
        '--pose_score_top_frac',
        type=float,
        default=0.05,
        help='Доля пикселей для global_topmean (верхняя часть распределения cos).',
    )
    p.add_argument(
        '--pose_patch_radius',
        type=int,
        default=10,
        help='Полуразмер окна (px) для centroid_patch при --pose_select relevancy.',
    )
    p.add_argument(
        '--pose_reuse_across_clusters',
        action='store_true',
        help='Разрешить один и тот же keyframe для нескольких кластеров '
             '(по умолчанию — разные frame_id, если хватает поз).',
    )
    p.add_argument(
        '--pose_min_gravity',
        type=float,
        default=0.12,
        help='Мин. «upright» по Y-up (отсекает перевёрнутые кадры). 0 = без порога по ориентации.',
    )
    p.add_argument(
        '--no_pose_gravity_filter',
        action='store_true',
        help='Не фильтровать по ориентации камеры (если сцена не Y-up).',
    )
    p.add_argument('--heatmap_norm', choices=('percentile', 'minmax'), default='percentile',
                   help='Нормализация тепловой карты: percentile уменьшает «пёстрость» от min–max.')
    p.add_argument('--heatmap_p_low', type=float, default=8.0,
                   help='Нижний перцентиль сырого cos-сходства (только для percentile).')
    p.add_argument('--heatmap_p_high', type=float, default=98.0,
                   help='Верхний перцентиль (только для percentile).')
    p.add_argument('--heatmap_blur', type=float, default=3.0,
                   help='Sigma Gaussian blur на карте схожести (px). 0 = без blur.')
    p.add_argument('--out',               required=True,
                   help='Output prefix.  Produces <out>_cluster<N>_rgb/overlay/3d.png')
    return p.parse_args()


def _load_traj_txt(path: Path, *, fmt: str, device: torch.device) -> dict[int, torch.Tensor]:
    arr = np.loadtxt(str(path), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 16:
        raise ValueError(f"{path} must have 16 floats per line (got {arr.shape[1]}).")
    mats = arr.reshape(-1, 4, 4)
    out: dict[int, torch.Tensor] = {}
    for i in range(mats.shape[0]):
        m = torch.tensor(mats[i], dtype=torch.float32, device=device)
        if fmt == 'c2w':
            m = torch.inverse(m)
        out[i] = m
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    latent_dim = args.latent_dim if args.latent_dim is not None else infer_latent_dim(
        Path(args.lang_field)
    )

    # Load model (latent_dim must match lang_field.pt — 3, 64, etc.)
    model = LangSplatam(checkpoint_path=args.checkpoint, latent_dim=latent_dim, device=args.device)
    model.load_lang_field(Path(args.lang_field))
    H = int(model.params['org_height'])
    W = int(model.params['org_width'])
    N = int(model.params['means3D'].shape[0])
    print(f'Gaussians: {N}   latent_dim={latent_dim}   Resolution: {H}×{W}')
    use_v2 = getattr(model, "model_format", "legacy") == "langsplatv2"
    if use_v2:
        print('Language field: LangSplatV2 (codebook + sparse coefficients)')
    else:
        print('Language field: legacy (AE latent)')

    means_all = model.params['means3D'].detach().cpu().numpy()

    use_traj = args.traj_txt is not None and str(args.traj_txt).strip() != ''
    poses_synthetic = (not use_traj) and (args.poses is None or str(args.poses).strip() == '')
    poses_path: Path | None
    poses_raw: dict
    poses_from_file: dict[int, torch.Tensor] | None = None
    auto_positions: np.ndarray | None = None
    scene_bounds_for_syn: tuple[np.ndarray, np.ndarray]

    if use_traj:
        poses_path = Path(args.traj_txt).expanduser()
        if not poses_path.is_file():
            raise FileNotFoundError(f"traj.txt not found: {poses_path.resolve()}")
        poses_from_file = _load_traj_txt(
            poses_path,
            fmt=args.traj_format,
            device=device,
        )
        poses_raw = {str(k): v.detach().cpu().numpy().tolist() for k, v in poses_from_file.items()}
        print(
            f'Camera poses: ONLY traj.txt ({len(poses_from_file)} poses) → {poses_path.resolve()} '
            f'(format={args.traj_format}).'
        )

        allowed_pose_ids_file = set(poses_from_file.keys())
        if args.no_pose_scene_bounds:
            scene_bounds = None
        else:
            scene_bounds = _resolve_scene_interior_bounds(
                means_all,
                poses_from_file,
                p_lo=args.pose_percentile_lo,
                p_hi=args.pose_percentile_hi,
                shrink_init=args.pose_interior_shrink,
            )

        lo_fb, hi_fb = _gaussian_interior_aabb(means_all, p_lo=3.0, p_hi=97.0, shrink_frac=0.12)
        scene_bounds_for_syn = _aabb_shrink_wall_margin(
            lo_fb, hi_fb, args.pose_wall_margin_m,
        )
    elif poses_synthetic:
        poses_path = None
        poses_raw = {}
        print(
            'Camera poses: **auto** (uniform in Gaussian interior AABB + dense-shell '
            'filter; look-at each cluster centroid). No keyframes JSON. '
            'Checkpoint is not used for camera selection.'
        )
        if args.no_pose_scene_bounds:
            scene_bounds = None
            lo_fb, hi_fb = _gaussian_interior_aabb(
                means_all, p_lo=args.pose_percentile_lo, p_hi=args.pose_percentile_hi,
                shrink_frac=args.pose_interior_shrink,
            )
            scene_bounds_for_syn = _aabb_shrink_wall_margin(
                lo_fb, hi_fb, args.pose_wall_margin_m,
            )
        else:
            scene_bounds = _scene_bounds_from_gaussians_only(
                means_all,
                p_lo=args.pose_percentile_lo,
                p_hi=args.pose_percentile_hi,
                shrink_init=args.pose_interior_shrink,
                wall_margin_m=args.pose_wall_margin_m,
            )
            scene_bounds_for_syn = scene_bounds
        rng_auto = np.random.default_rng(args.auto_pose_seed)
        # Sampling box: same hull + wall margin as ``scene_bounds_for_syn`` (margin applied once).
        sample_bounds = scene_bounds if scene_bounds is not None else scene_bounds_for_syn
        auto_positions = _sample_auto_camera_positions(
            means_all,
            sample_bounds,
            rng_auto,
            n_samples=args.auto_pose_samples,
            nn_k=args.pose_nn_k,
            max_candidates=args.auto_pose_max_candidates,
            wall_margin_m=0.0,
        )
        if auto_positions is None or auto_positions.shape[0] == 0:
            print(
                '[error] Auto pose sampling found no camera positions inside the dense '
                'Gaussian shell. Try --auto_pose_samples 8000, loosen '
                '--pose_percentile_lo / --pose_percentile_hi, or '
                '--no_pose_scene_bounds (debug only).'
            )
            sys.exit(1)
        allowed_pose_ids_file: set[int] | None = None
    else:
        poses_path = Path(args.poses).expanduser()
        if not poses_path.is_file():
            raise FileNotFoundError(
                f'Нет файла с keyframe-позами: {poses_path.resolve()}. '
                f'Оставьте --poses пустым для автопоз, или укажите корректный путь.'
            )
        with open(poses_path) as f:
            poses_raw = json.load(f)
        poses_from_file = {int(k): torch.tensor(v, dtype=torch.float32, device=device)
                           for k, v in poses_raw.items()}
        _ids = sorted(poses_from_file.keys())
        _preview = _ids[:20]
        _more = f' … (+{len(_ids) - 20} ids)' if len(_ids) > 20 else ''
        print(
            f'Camera poses: ONLY {poses_path.resolve()}  ({len(poses_from_file)} keyframes). '
            f'Checkpoint is not used for camera selection.'
        )
        print(f'  frame_id keys in JSON (sample): {_preview}{_more}')

        scene_bounds: tuple[np.ndarray, np.ndarray] | None
        if args.no_pose_scene_bounds:
            scene_bounds = None
        else:
            scene_bounds = _resolve_scene_interior_bounds(
                means_all,
                poses_from_file,
                p_lo=args.pose_percentile_lo,
                p_hi=args.pose_percentile_hi,
                shrink_init=args.pose_interior_shrink,
            )

        rng_pose = np.random.default_rng(0)
        allowed_pose_ids_file = None
        if not args.no_pose_scene_bounds:
            allowed_pose_ids_file, _thr_nn = _compute_allowed_pose_ids(
                poses_from_file,
                means_all,
                scene_bounds,
                rng_pose,
                nn_k=args.pose_nn_k,
            )
            if len(allowed_pose_ids_file) == 0:
                print(
                    '[error] Нет ни одной допустимой позы: центр камеры не попадает '
                    'внутрь hull сцены по гауссианам (или keyframe_poses не в той же '
                    'системе координат, что checkpoint). '
                    'Подбор поз вне сцены отключён. Проверьте poses/checkpoint или '
                    'ослабьте --pose_percentile_lo / --pose_percentile_hi / '
                    '--pose_interior_shrink. Для отладки без ограничения: '
                    '--no_pose_scene_bounds. Либо запустите без --poses для автопоз.'
                )
                sys.exit(1)
        lo_fb, hi_fb = _gaussian_interior_aabb(means_all, p_lo=3.0, p_hi=97.0, shrink_frac=0.12)
        scene_bounds_for_syn = _aabb_shrink_wall_margin(
            lo_fb, hi_fb, args.pose_wall_margin_m,
        )

    # Load AE for decoding (legacy only)
    ae = None
    if use_v2:
        if args.ae_ckpt:
            print('[warn] --ae_ckpt ignored for LangSplatV2 checkpoints.')
    else:
        if not args.ae_ckpt:
            sys.exit(
                'Для legacy lang_field.pt укажите --ae_ckpt (чекпойнт автоэнкодера).'
            )
        ae = _load_ae(Path(args.ae_ckpt), args.encoder_dims, args.decoder_dims, device)

    # CLIP text embedding in 512d (the actual semantic space)
    clip_query = encode_query_clip(
        args.text, args.clip_model, args.clip_pretrained, device)  # [512]
    print(f'Query: "{args.text}"')
    print(
        f'Pose selection: {args.pose_select}  '
        f'(pose_score_mode={args.pose_score_mode}, top_frac={args.pose_score_top_frac}, '
        f'patch_radius={args.pose_patch_radius})'
    )
    print(
        f'Cluster rank: {args.cluster_rank_by}  |  '
        f'Gaussian score: {"raw cos (no opacity)" if args.gaussian_score_no_opacity else "cos × opacity"}'
    )

    # Score all Gaussians in 512d: decode latent → CLIP space, then cosine.
    # По умолчанию умножаем на sigmoid(opacity), чтобы почти прозрачные выбросы
    # не забирали top-percentile; опционально — сырой cos для более «семантического» ранжирования.
    with torch.no_grad():
        opacity = torch.sigmoid(
            model.params['logit_opacities'].detach().squeeze(-1)   # [N]
        )
        # IMPORTANT: stream scoring to avoid holding [N,512] on GPU.
        batch = 65536
        dev = torch.device(args.device)
        q = clip_query.to(dev).to(torch.float32)            # [512]
        scores = np.empty((N,), dtype=np.float32)
        for i in range(0, N, batch):
            if use_v2:
                chunk_512 = model.decode_gaussian_clip_v2(i, i + batch)
            else:
                assert ae is not None
                chunk_lat = model.lang_feats.detach()[i : i + batch]
                chunk_512 = ae.decode(chunk_lat)
            chunk_512 = F.normalize(chunk_512, dim=-1).to(torch.float32)
            cos = (chunk_512 @ q).clamp(min=0.0)                    # [B]
            op = opacity[i : i + cos.shape[0]].to(torch.float32)
            if args.gaussian_score_no_opacity:
                sc = cos.cpu().numpy()
            else:
                sc = (cos * op).cpu().numpy()
            scores[i : i + sc.shape[0]] = sc

    # Take top-percentile
    threshold = np.percentile(scores, 100.0 - args.top_percentile)
    mask = scores >= threshold
    top_means  = model.params['means3D'].detach().cpu().numpy()[mask]  # [K, 3]
    top_scores = scores[mask]                                           # [K]
    print(f'Top-{args.top_percentile}%: {mask.sum()} Gaussians  '
          f'score range [{top_scores.min():.4f}, {top_scores.max():.4f}]')

    # Clustering (DBSCAN by default; optional single-cluster mode).
    if args.no_clusters:
        centroid = top_means.mean(axis=0)
        if top_scores.sum() > 1e-12:
            # Weighted centroid is usually more stable for semantic peaks.
            centroid = (top_means * top_scores[:, None]).sum(axis=0) / (top_scores.sum() + 1e-12)
        clusters = [{
            'label': 0,
            'centroid': centroid,
            'total_score': float(top_scores.sum()),
            'size': int(top_means.shape[0]),
            'means3d': top_means,
            'scores': top_scores,
        }]
        print('Clustering: disabled (--no_clusters), using single aggregated cluster.')
    else:
        clusters = find_clusters(top_means, top_scores,
                                 eps=args.dbscan_eps, min_samples=args.dbscan_min)
        if not clusters:
            print('No clusters found. Try --dbscan_eps larger or --top_percentile larger.')
            return

    rank_clusters_by(clusters, args.cluster_rank_by)
    print(f'Found {len(clusters)} cluster(s), ordered by {args.cluster_rank_by}:')
    for i, cl in enumerate(clusters[:6]):
        _mean = float(cl['total_score']) / max(1, cl['size'])
        print(f'  cluster {i+1}: size={cl["size"]}  '
              f'total={cl["total_score"]:.1f}  mean={_mean:.4f}  '
              f'centroid={cl["centroid"].round(3)}')

    # Camera world positions for visualisation (traj/keyframes or auto-sampled centres)
    if poses_synthetic:
        assert auto_positions is not None
        cam_positions = auto_positions.copy()
    else:
        if poses_from_file is None:
            cam_positions = np.zeros((0, 3), dtype=np.float32)
        else:
            cam_positions = np.array(
                [
                    _cam_world_from_w2c(v.detach().cpu().numpy().astype(np.float64))
                    for v in poses_from_file.values()
                ]
            )

    rmap_kw = dict(
        heatmap_norm=args.heatmap_norm,
        heatmap_p_low=args.heatmap_p_low,
        heatmap_p_high=args.heatmap_p_high,
        blur_sigma=args.heatmap_blur,
        use_v2=use_v2,
    )

    # Colour palette: index=0 is reserved for the TARGET cluster (green).
    # Other indices are used for the remaining clusters in their natural order.
    GREEN_BGR = (0, 220, 0)
    OTHER_COLORS_BGR = [
        (0, 60, 255),    # red
        (0, 165, 255),   # orange
        (0, 230, 255),   # yellow
        (255, 80, 0),    # azure/blue
        (180, 0, 255),   # violet/magenta
        (0, 255, 180),   # cyan-lime
        (128, 0, 128),   # purple
    ]

    def _color_for(ci_target, ci_draw):
        """Return BGR colour for cluster ci_draw when target is ci_target."""
        if ci_draw == ci_target:
            return GREEN_BGR
        # assign OTHER_COLORS_BGR in order, skipping the target index slot
        other_idx = ci_draw if ci_draw < ci_target else ci_draw - 1
        return OTHER_COLORS_BGR[other_idx % len(OTHER_COLORS_BGR)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_save = min(args.top_k_views, len(clusters))

    # Pre-compute SAM masks & best poses for ALL clusters we will draw.
    # We do this once so each per-cluster image can reuse them.
    used_fids: set[int] = set()
    min_grav = None if args.no_pose_gravity_filter else (
        None if args.pose_min_gravity <= 0.0 else args.pose_min_gravity
    )

    cluster_data = []   # list of dicts per cluster
    for ci, cl in enumerate(clusters[:n_save]):
        excl = set() if args.pose_reuse_across_clusters else used_fids
        if poses_synthetic:
            assert auto_positions is not None
            poses = _synthetic_poses_look_at_centroid(
                auto_positions, cl['centroid'], scene_bounds_for_syn, device,
            )
            allowed_pose_ids = set(poses.keys())
        else:
            assert poses_from_file is not None
            poses = poses_from_file
            allowed_pose_ids = allowed_pose_ids_file

        if args.pose_select == 'relevancy':
            fid, w2c = best_pose_by_query_relevancy(
                cl['centroid'], poses, model.intrinsics, H, W, device,
                model, ae, clip_query,
                use_v2=use_v2,
                margin_px=args.pose_margin_px,
                z_min=args.pose_z_min,
                scene_bounds=scene_bounds,
                require_cam_in_scene=not args.no_pose_scene_bounds,
                allowed_pose_ids=allowed_pose_ids,
                blur_sigma=args.heatmap_blur,
                patch_radius=args.pose_patch_radius,
                score_mode=args.pose_score_mode,
                top_frac=args.pose_score_top_frac,
                exclude_fids=excl,
                min_gravity=min_grav,
            )
        else:
            fid, w2c = best_pose_for_centroid(
                cl['centroid'], poses, model.intrinsics, H, W, device,
                margin_px=args.pose_margin_px,
                z_min=args.pose_z_min,
                scene_bounds=scene_bounds,
                require_cam_in_scene=not args.no_pose_scene_bounds,
                allowed_pose_ids=allowed_pose_ids,
                exclude_fids=excl,
                min_gravity=min_grav,
            )
        if fid is not None:
            used_fids.add(int(fid))
        if fid is None:
            print(f'  Warning: no valid pose for cluster {ci + 1} centroid.')
        _gv = _camera_gravity_alignment(w2c) if w2c is not None else float('nan')
        cluster_data.append({'cl': cl, 'fid': fid, 'w2c': w2c, 'grav': _gv})

    # Write a lightweight metrics summary for batch evaluation (so we don't need to parse stdout).
    try:
        metrics = {
            "text": args.text,
            "lang_field": str(Path(args.lang_field).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "ae_ckpt": str(Path(args.ae_ckpt).resolve()) if args.ae_ckpt else "",
            "langsplat_v2": bool(use_v2),
            "latent_dim": int(latent_dim),
            "top_percentile": float(args.top_percentile),
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_min": int(args.dbscan_min),
            "pose_select": str(args.pose_select),
            "pose_score_mode": str(args.pose_score_mode),
            "pose_score_top_frac": float(args.pose_score_top_frac),
            "cluster_rank_by": str(args.cluster_rank_by),
            "gaussian_score_no_opacity": bool(args.gaussian_score_no_opacity),
            "sam_prompt_from": str(args.sam_prompt_from),
            "clusters_found": int(len(clusters)),
            "clusters_saved": int(n_save),
            "clusters": [
                {
                    "rank": int(i + 1),
                    "size": int(cd["cl"]["size"]),
                    "total_score": float(cd["cl"]["total_score"]),
                    "centroid": [float(x) for x in cd["cl"]["centroid"]],
                    "frame_id": None if cd["fid"] is None else int(cd["fid"]),
                    "gravity": float(cd.get("grav", float("nan"))),
                }
                for i, cd in enumerate(cluster_data)
            ],
        }
        mpath = out.parent / f"{out.name}_metrics.json"
        mpath.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Metrics JSON → {mpath}")
    except Exception as _e:
        print(f"[warn] Could not write metrics JSON: {_e}")

    # Colour palette for SAM auto masks (no green — reserved for query cluster)
    SAM_PALETTE_BGR = [
        (0,    60, 255),   # red
        (0,   165, 255),   # orange
        (0,   230, 255),   # yellow
        (255,  80,   0),   # blue
        (180,   0, 255),   # violet
        (0,   255, 180),   # cyan
        (128,   0, 128),   # purple
        (255, 255,   0),   # aqua
        (0,   128, 255),   # deep-orange
        (200, 200,   0),   # teal
    ]
    GREEN_BGR = (0, 220, 0)

    def _apply_mask(img: np.ndarray, mask2d: np.ndarray,
                    color_bgr: tuple, alpha: float = 0.45) -> np.ndarray:
        a = (mask2d > 0).astype(np.float32)[:, :, None]
        c = np.full_like(img, np.array(color_bgr, dtype=np.uint8))
        blended = (img.astype(np.float32) * (1 - alpha * a)
                   + c.astype(np.float32) * (alpha * a)).clip(0, 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, cnts, -1, color_bgr, 1)
        return blended

    best_cam_pos = None
    rng = np.random.default_rng(42)

    for ci_target, cd_target in enumerate(cluster_data):
        w2c = cd_target['w2c']
        if w2c is None:
            print(f'  Cluster {ci_target+1}: no valid pose, skipping.')
            continue

        cp = _cam_world_from_w2c(w2c)
        if best_cam_pos is None:
            best_cam_pos = cp
        _g = cd_target.get('grav', float('nan'))
        _gstr = f'{_g:.2f}' if np.isfinite(_g) else 'nan'
        if poses_synthetic:
            _print_auto_pose(
                f'cluster {ci_target + 1}',
                int(cd_target['fid']),
                w2c,
            )
        else:
            assert poses_path is not None
            _print_pose_from_keyframe_json(
                poses_path,
                poses_raw,
                f'cluster {ci_target + 1}',
                int(cd_target['fid']),
                w2c,
            )
        print(f'  upright (Y-up heuristic)={_gstr}  (if always wrong, try --no_pose_gravity_filter)')

        rgb     = render_rgb(model, w2c, H, W)
        jet, _norm, sim_raw = render_relevancy_map(
            model, clip_query, ae, w2c, H, W, **rmap_kw)
        overlay = cv2.addWeighted(rgb, 0.55, jet, 0.45, 0)

        # --- _semantic: SAM auto masks + query cluster on top in green ---
        print(f'    Running SAM auto mask generator for cluster {ci_target+1}...')
        auto_masks = sam_all_masks(rgb, args.sam_ckpt, device)
        print(f'    SAM found {len(auto_masks)} masks')

        sem = rgb.copy()

        # 1) All SAM masks in non-green colours
        for i, mask2d in enumerate(auto_masks):
            color_bgr = SAM_PALETTE_BGR[i % len(SAM_PALETTE_BGR)]
            sem = _apply_mask(sem, mask2d, color_bgr, alpha=0.35)

        # 2) Query cluster on top — green, via point prompt (пик relevancy или центроид)
        centroid_xy = project_centroid(
            cd_target['cl']['centroid'], w2c, model.intrinsics, device)
        pt_xy = sam_prompt_xy_from_relevancy(
            sim_raw,
            args.sam_prompt_from,
            centroid_xy,
            H,
            W,
            com_top_frac=args.sam_com_top_frac,
        )
        if pt_xy is not None:
            query_mask = sam_mask_from_point(rgb, pt_xy, args.sam_ckpt, device)
        else:
            query_mask = np.zeros((H, W), dtype=np.uint8)
            pt_xy = None
        sem = _apply_mask(sem, query_mask, GREEN_BGR, alpha=0.55)

        # Dot + label for the query cluster
        if pt_xy is not None:
            cv2.circle(sem, pt_xy, 9, GREEN_BGR, -1)
            cv2.circle(sem, pt_xy, 9, (255, 255, 255), 2)
            label = f'#{ci_target+1} << query'
            cv2.putText(sem, label, (pt_xy[0] + 13, pt_xy[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(sem, label, (pt_xy[0] + 13, pt_xy[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        sfx = f'_cluster{ci_target+1}'
        cv2.imwrite(str(out) + sfx + '_rgb.png',       rgb)
        cv2.imwrite(str(out) + sfx + '_relevancy.png',  jet)
        cv2.imwrite(str(out) + sfx + '_overlay.png',   overlay)
        cv2.imwrite(str(out) + sfx + '_semantic.png',  sem)
        print(f'    saved {out}{sfx}_{{rgb,relevancy,overlay,semantic}}.png')

    # Global _semantic.png: SAM auto on best-cluster pose, no query highlight
    if best_cam_pos is not None:
        best_w2c = cluster_data[0]['w2c'] if cluster_data[0]['w2c'] is not None else next(
            cd['w2c'] for cd in cluster_data if cd['w2c'] is not None)
        rgb_best = render_rgb(model, best_w2c, H, W)
        jet_best, _nb, _sr = render_relevancy_map(
            model, clip_query, ae, best_w2c, H, W, **rmap_kw)
        auto_best = sam_all_masks(rgb_best, args.sam_ckpt, device)
        sem_best = rgb_best.copy()
        for i, mask2d in enumerate(auto_best):
            color_bgr = SAM_PALETTE_BGR[i % len(SAM_PALETTE_BGR)]
            sem_best = _apply_mask(sem_best, mask2d, color_bgr, alpha=0.35)
        cv2.imwrite(str(out) + '_rgb.png',       rgb_best)
        cv2.imwrite(str(out) + '_relevancy.png',  jet_best)
        cv2.imwrite(str(out) + '_overlay.png',
                    cv2.addWeighted(rgb_best, 0.55, jet_best, 0.45, 0))
        cv2.imwrite(str(out) + '_semantic.png',  sem_best)
        _gf = cluster_data[0]['fid']
        if poses_synthetic:
            print(
                f'Saved global: {out}_{{rgb,relevancy,overlay,semantic}}.png  '
                f'(w2c = auto-sampled pose id {_gf!r})'
            )
        else:
            assert poses_path is not None
            print(
                f'Saved global: {out}_{{rgb,relevancy,overlay,semantic}}.png  '
                f'(w2c = JSON key {_gf!r} from {poses_path.name})'
            )

    # 3-D cluster map
    if best_cam_pos is None:
        best_cam_pos = cam_positions[0]
    save_cluster_map(
        model.params['means3D'].detach().cpu().numpy(),
        clusters, cam_positions, best_cam_pos,
        Path(str(out) + '_clusters_3d.png'), args.text,
    )


if __name__ == '__main__':
    main()
