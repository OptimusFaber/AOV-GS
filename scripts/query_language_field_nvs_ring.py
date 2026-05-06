#!/usr/bin/env python3
"""
Запуск query_language_field.py с наборами поз **как в replica_sim_nvs**: орбита камер
вокруг точки ``center`` на радиусе ``radius``, каждая камера смотрит на ``center``.

Опорная **якорная** поза берётся из ``keyframe_poses.json`` (как при NVS берут первый
кадр траектории). По умолчанию ``center`` — точка впереди камеры на ``--look_ahead_m``
(вдоль оптической оси в мировых координатах), чтобы кольцо обходило область перед
объективом, а не случайный AABB.

Использование (все аргументы для query_language_field.py передаются после ring-опций):

  cd AOV-GS
  python scripts/query_language_field_nvs_ring.py \\
    --poses_json results/Replica/office0/ActiveOpenSem/run_0/keyframe_poses.json \\
    --anchor_frame 9 \\
    --orbit_radius 0.45 \\
    --num_ring 36 \\
    --checkpoint results/.../splatam/final/params.npz \\
    --lang_field results/.../lang_field_m64/lang_field.pt \\
    --text "a sofa" \\
    --ae_ckpt ckpt/office0/64/best_ckpt.pth \\
    --out results/.../query_sofa_nvs_ring \\
    --no_clusters

Опционально: ``--ring_poses_out path.json`` сохранить сгенерированные w2c;
``--dry_run`` только записать JSON и выйти.

Кольцо поз заимствует геометрию из ``generate_round_trajectory`` (Replica NVS).

**Только overlay (12 кадров, не подряд):** строится полная орбита из
``--orbit_total_frames`` поз (по умолчанию **180**, как плотность NVS), затем
берутся кадры с шагом ``--overlay_stride`` (по умолчанию **15**): в нумерации с 1 это
**1, 16, 31, 46, …** (в коде индексы **0, 15, 30, …**), всего ``--overlay_num_samples``
кадров (по умолчанию **12**).

  python scripts/query_language_field_nvs_ring.py --overlay_only \\
    --poses_json .../keyframe_poses.json --anchor_frame 9 \\
    --orbit_radius 0.45 \\
    --checkpoint .../params.npz --lang_field .../lang_field.pt \\
    --text "a sofa" --ae_ckpt ckpt/office0/64/best_ckpt.pth \\
    --out results/.../nvs_ring_sofa

  Пишет ``{out}_000_overlay.png`` … ``{out}_011_overlay.png`` (только overlay).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# --- Replica NVS: generate_round_trajectory (same as src/data/generate_Replica_NVS_data.py) ---


def generate_round_trajectory(
    center: np.ndarray,
    radius: float,
    up_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    num_points: int = 36,
) -> list[np.ndarray]:
    """Камеры на окружности вокруг ``center``, ось ``up_axis``, взгляд на ``center``."""
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-12)
    poses: list[np.ndarray] = []

    right_axis = np.cross(up_axis, np.array([0.0, 0.0, -1.0]))
    if np.linalg.norm(right_axis) == 0:
        right_axis = np.array([1.0, 0.0, 0.0])
    right_axis = right_axis / (np.linalg.norm(right_axis) + 1e-12)

    center = np.asarray(center, dtype=np.float64).reshape(3)

    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        position = center + radius * (
            np.cos(angle) * right_axis + np.sin(angle) * np.cross(up_axis, right_axis)
        )

        forward = center - position
        zn = float(np.linalg.norm(forward))
        if zn < 1e-6:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            forward = forward / zn

        up = up_axis - np.dot(up_axis, forward) * forward
        un = float(np.linalg.norm(up))
        if un < 1e-8:
            up = np.cross(forward, right_axis)
            un = float(np.linalg.norm(up))
        up = up / (un + 1e-12)

        right = np.cross(forward, up)

        rotation_matrix = np.vstack((right, -up, forward)).T
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = position.astype(np.float32)
        poses.append(pose)

    return poses


def _load_w2c(poses_json: Path, anchor_frame: str) -> np.ndarray:
    with open(poses_json) as f:
        raw = json.load(f)
    key = str(anchor_frame)
    if key not in raw:
        raise KeyError(f"frame id {key!r} not in {poses_json}")
    m = np.asarray(raw[key], dtype=np.float64).reshape(4, 4)
    return m


def _c2w_from_w2c(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(w2c)


def _parse_center_mode(
    args: argparse.Namespace,
    c2w_anchor: np.ndarray,
) -> np.ndarray:
    cam = c2w_anchor[:3, 3].astype(np.float64)
    if args.center_mode == "manual":
        c = np.array([args.cx, args.cy, args.cz], dtype=np.float64)
        return c.reshape(3)
    if args.center_mode == "camera":
        return cam.copy()
    # in_front
    fwd = c2w_anchor[:3, 2].astype(np.float64)
    fn = float(np.linalg.norm(fwd))
    if fn < 1e-8:
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        fwd = fwd / fn
    return cam + float(args.look_ahead_m) * fwd


def subsample_orbit_indices(
    orbit_total: int,
    num_samples: int,
    stride: int,
) -> list[int]:
    """
    Индексы 0-based: ``0, stride, 2*stride, …`` — как кадры 1, 1+stride, … в нумерации NVS.
    Последний индекс: ``stride * (num_samples - 1)`` (должен быть < orbit_total).
    """
    idx = [stride * k for k in range(num_samples)]
    mx = idx[-1] if idx else -1
    if mx >= orbit_total:
        raise ValueError(
            f"orbit_total_frames={orbit_total} слишком мало для "
            f"num_samples={num_samples}, stride={stride}: нужен индекс <= {mx}"
        )
    return idx


def _build_ring_json(
    c2w_list: list[np.ndarray],
) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for i, c2w in enumerate(c2w_list):
        w2c = np.linalg.inv(c2w.astype(np.float64))
        out[str(i)] = w2c.astype(np.float64).tolist()
    return out


def _load_query_language_field_module(proj: Path):
    path = proj / "scripts" / "query_language_field.py"
    spec = importlib.util.spec_from_file_location("query_language_field_nvs_aux", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_overlay_only_frames(
    proj: Path,
    ring_c2w: list[np.ndarray],
    *,
    checkpoint: str,
    lang_field: str,
    text: str,
    ae_ckpt: str,
    out: str,
    device: str = "cuda:0",
    latent_dim: int | None = None,
    clip_model: str = "ViT-B-16",
    clip_pretrained: str = "laion2b_s34b_b88k",
    encoder_dims: list[int] | None = None,
    decoder_dims: list[int] | None = None,
    heatmap_norm: str = "percentile",
    heatmap_p_low: float = 8.0,
    heatmap_p_high: float = 98.0,
    heatmap_blur: float = 3.0,
) -> None:
    qlf = _load_query_language_field_module(proj)
    import cv2
    import torch

    device_t = torch.device(device)
    lf = Path(lang_field)
    latent_dim = latent_dim if latent_dim is not None else qlf.infer_latent_dim(lf)
    model = qlf.LangSplatam(
        checkpoint_path=checkpoint, latent_dim=latent_dim, device=device
    )
    model.load_lang_field(lf)
    H = int(model.params["org_height"])
    W = int(model.params["org_width"])
    ae = qlf._load_ae(Path(ae_ckpt), encoder_dims, decoder_dims, device_t)
    clip_query = qlf.encode_query_clip(text, clip_model, clip_pretrained, device_t)
    rmap_kw = dict(
        heatmap_norm=heatmap_norm,
        heatmap_p_low=heatmap_p_low,
        heatmap_p_high=heatmap_p_high,
        blur_sigma=heatmap_blur,
    )
    out_prefix = Path(out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for i, c2w in enumerate(ring_c2w):
        w2c_np = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
        w2c = torch.tensor(w2c_np, dtype=torch.float32, device=device_t)
        rgb = qlf.render_rgb(model, w2c, H, W)
        jet, _norm, _sim = qlf.render_relevancy_map(
            model, clip_query, ae, w2c, H, W, **rmap_kw
        )
        overlay = cv2.addWeighted(rgb, 0.55, jet, 0.45, 0)
        fp = out_prefix.parent / f"{out_prefix.name}_{i:03d}_overlay.png"
        cv2.imwrite(str(fp), overlay)
        print(f"[nvs_ring overlay] {fp}")


def _add_ring_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--poses_json", type=Path, required=True, help="keyframe_poses.json (w2c)")
    p.add_argument("--anchor_frame", type=str, required=True, help="Ключ кадра, напр. 9")
    p.add_argument("--center_mode", choices=("in_front", "camera", "manual"), default="in_front")
    p.add_argument("--cx", type=float, default=0.0)
    p.add_argument("--cy", type=float, default=0.0)
    p.add_argument("--cz", type=float, default=0.0)
    p.add_argument(
        "--look_ahead_m",
        type=float,
        default=1.0,
        help="Для in_front: центр орбиты = камера + это * forward (м).",
    )
    p.add_argument("--orbit_radius", type=float, default=0.5, help="Радиус кольца (м), как sim.radius")
    p.add_argument("--num_ring", type=int, default=36, help="Число поз на кольце (режим query)")
    p.add_argument(
        "--up",
        nargs=3,
        type=float,
        default=[0.0, 1.0, 0.0],
        metavar=("UX", "UY", "UZ"),
        help="Up для generate_round_trajectory (по умолчанию Y-up)",
    )
    p.add_argument(
        "--no_include_anchor",
        action="store_true",
        help="Не добавлять исходную якорную w2c как кадр 0 (по умолчанию якорь — кадр 0)",
    )
    p.add_argument("--ring_poses_out", type=Path, default=None, help="Куда сохранить ring JSON")
    p.add_argument("--dry_run", action="store_true", help="Только JSON / метаданные, без query/рендера")
    p.add_argument(
        "--keep_temp",
        action="store_true",
        help="Не удалять временный JSON после запуска (если ring_poses_out не задан)",
    )


def main() -> int:
    proj = Path(__file__).resolve().parents[1]
    argv = sys.argv[1:]

    if "--overlay_only" in argv:
        # 12 кадров по орбите NVS, только overlay PNG
        p = argparse.ArgumentParser(
            description="NVS orbit → только overlay (без полного query_language_field).",
        )
        p.add_argument(
            "--overlay_only",
            action="store_true",
            help="Только overlay: полная орбита orbit_total_frames, затем субсэмпл с overlay_stride",
        )
        _add_ring_args(p)
        p.add_argument(
            "--orbit_total_frames",
            type=int,
            default=180,
            help="Полная орбита: столько же поз, сколько плотных кадров NVS (по умолчанию 180).",
        )
        p.add_argument(
            "--overlay_num_samples",
            type=int,
            default=12,
            help="Сколько кадров сохранить (равномерная выборка с шагом overlay_stride).",
        )
        p.add_argument(
            "--overlay_stride",
            type=int,
            default=15,
            help="Шаг по полной орбите: кадры 1, 1+stride, … (1-based) → 0, stride, … (0-based).",
        )
        p.add_argument("--checkpoint", type=str, required=True)
        p.add_argument("--lang_field", type=str, required=True)
        p.add_argument("--text", type=str, required=True)
        p.add_argument("--ae_ckpt", type=str, required=True)
        p.add_argument("--out", type=str, required=True, help="Префикс: out_000_overlay.png …")
        p.add_argument("--device", type=str, default="cuda:0")
        p.add_argument("--latent_dim", type=int, default=None)
        p.add_argument("--clip_model", type=str, default="ViT-B-16")
        p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
        p.add_argument("--encoder_dims", nargs="+", type=int, default=None)
        p.add_argument("--decoder_dims", nargs="+", type=int, default=None)
        p.add_argument("--heatmap_norm", choices=("percentile", "minmax"), default="percentile")
        p.add_argument("--heatmap_p_low", type=float, default=8.0)
        p.add_argument("--heatmap_p_high", type=float, default=98.0)
        p.add_argument("--heatmap_blur", type=float, default=3.0)

        args = p.parse_args(argv)

        w2c_anchor = _load_w2c(args.poses_json, args.anchor_frame)
        c2w_anchor = _c2w_from_w2c(w2c_anchor)
        center = _parse_center_mode(args, c2w_anchor)
        up_axis = np.array(args.up, dtype=np.float64)

        orbit_total = int(args.orbit_total_frames)
        ring_full = generate_round_trajectory(
            center,
            float(args.orbit_radius),
            up_axis,
            orbit_total,
        )
        idx_list = subsample_orbit_indices(
            orbit_total,
            int(args.overlay_num_samples),
            int(args.overlay_stride),
        )
        ring_c2w = [ring_full[i] for i in idx_list]
        print(
            f"[nvs_ring overlay] anchor_frame={args.anchor_frame}  center_mode={args.center_mode}  "
            f"center={np.round(center, 4)}  r={args.orbit_radius}  "
            f"orbit_total={orbit_total}  indices(0-based)={idx_list}  "
            f"→ frames(1-based)={[i + 1 for i in idx_list]}"
        )

        if args.dry_run:
            print("[nvs_ring overlay] dry_run: no render")
            return 0

        run_overlay_only_frames(
            proj,
            ring_c2w,
            checkpoint=args.checkpoint,
            lang_field=args.lang_field,
            text=args.text,
            ae_ckpt=args.ae_ckpt,
            out=args.out,
            device=args.device,
            latent_dim=args.latent_dim,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            encoder_dims=args.encoder_dims,
            decoder_dims=args.decoder_dims,
            heatmap_norm=args.heatmap_norm,
            heatmap_p_low=args.heatmap_p_low,
            heatmap_p_high=args.heatmap_p_high,
            heatmap_blur=args.heatmap_blur,
        )
        return 0

    p = argparse.ArgumentParser(
        description="NVS-style ring poses → query_language_field.py",
    )
    _add_ring_args(p)
    args, rest = p.parse_known_args(argv)
    if not rest:
        print(
            "ERROR: после ring-опций нужны аргументы для scripts/query_language_field.py "
            "(--checkpoint, --lang_field, --text, --ae_ckpt, --out, ...). "
            "Или используйте --overlay_only для 12 overlay без полного query.",
            file=sys.stderr,
        )
        return 2

    w2c_anchor = _load_w2c(args.poses_json, args.anchor_frame)
    c2w_anchor = _c2w_from_w2c(w2c_anchor)

    center = _parse_center_mode(args, c2w_anchor)
    up_axis = np.array(args.up, dtype=np.float64)

    ring_c2w = generate_round_trajectory(
        center,
        float(args.orbit_radius),
        up_axis,
        int(args.num_ring),
    )
    if not args.no_include_anchor:
        ring_c2w = [c2w_anchor.astype(np.float32)] + ring_c2w

    ring_obj = _build_ring_json(ring_c2w)

    tmp_path: Path | None = None
    if args.ring_poses_out is not None:
        ring_path = args.ring_poses_out.expanduser().resolve()
        ring_path.parent.mkdir(parents=True, exist_ok=True)
        ring_path.write_text(json.dumps(ring_obj), encoding="utf-8")
    else:
        fd, tmp_name = tempfile.mkstemp(suffix="_nvs_ring_poses.json", text=True)
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ring_obj, f)
        ring_path = tmp_path

    print(
        f"[nvs_ring] anchor_frame={args.anchor_frame}  center_mode={args.center_mode}  "
        f"center={np.round(center, 4)}  r={args.orbit_radius}  N={len(ring_obj)}  → {ring_path}"
    )

    if args.dry_run:
        print("[nvs_ring] dry_run: skipping query_language_field.py (JSON оставлен на диске)")
        return 0

    q_script = proj / "scripts" / "query_language_field.py"
    cmd = [sys.executable, str(q_script), "--poses", str(ring_path)] + rest
    print("[nvs_ring] ", " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=str(proj))
        return int(r.returncode)
    finally:
        if tmp_path is not None and not args.keep_temp:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
