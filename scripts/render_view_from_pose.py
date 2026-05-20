#!/usr/bin/env python3
"""
Render a 2D RGB view of the trained Gaussian splat from a **robot / camera pose**.

RU
--
Использует тот же пайплайн, что валидация / query-скрипты AOV-GS-V2: загрузка ``params.npz`` через
``LangSplatam`` и ``render_rgb`` (``diff_gaussian_rasterization``, как в ``query_language_field.py``).

Поза для рендера — матрица **world → camera** (``w2c``) в системе координат **обучающего** SplaTAM
(тот же мир, что ``data/Replica/<scene>/traj.txt``). Для абсолютных **c2w** из ``replica_sim_nvs``
(как в ``eval_helper.py`` при ``align_eval_world``): ``w2c = inv(c2w_nvs) @ c2w_train0``, где
``c2w_train0`` — первая строка **Replica**, не обязательно первая строка NVS-файла.

**Режим ``--from_nvs_eval``:** как ``scripts/run_nvs_validation`` — загрузка cfg → SLAM →
``load_params``, поза из ``dataset_eval`` с той же формулой ``gt_w2c``, что ``eval_result``.
Рендер **как в** ``src/slam/splatam/eval_helper.eval`` (``transform_to_frame`` +
``setup_camera`` с **первым кадром**), а не ``LangSplatam.render_rgb`` — иначе вид не совпадает с
``PR/frame_*.png`` / ``rendered_rgb/gs_*.png`` из eval.
Флаги ``--width`` / ``--height`` при необходимости только масштабируют результат; для пиксель-в-пиксель
с eval их лучше не задавать.

**Почему ``--traj .../replica_sim_nvs/...`` ломался:** якорь ``inv(c2w_i)@c2w_traj[0]`` верен только если
первая NVS-поза совпадает с первой Replica-позой; иначе кадр 0 может казаться верным, а следующие — нет.
Частый случай: при генерации ``results_habitat`` кадры с ``too_close`` **пропускаются**
(``generate_Replica_NVS_data.py``), и первая строка ``traj.txt`` — это первый **сохранённый** кадр, не
обязательно совпадающий с Replica[0]. По умолчанию скрипт выравнивает через
``data/Replica/<scene>/traj.txt`` (см. ``--traj_align``).

EN
--
Expected pose is **w2c** in the **training / checkpoint** world frame. For absolute **c2w** from
``replica_sim_nvs``, the same alignment as ``eval_helper.py`` (cross-sequence eval) is
``w2c = inv(c2w_nvs) @ c2w_train0`` where ``c2w_train0`` is the **first pose of**
``data/Replica/<scene>/traj.txt`` — not necessarily row 0 of the NVS traj file.

**``--from_nvs_eval`` mode:** same loading as ``run_nvs_validation``; same ``gt_w2c`` as ``eval_result``.
Rendering matches ``eval_helper.eval`` (first-frame ``setup_camera`` + per-frame ``transform_to_frame``),
**not** ``LangSplatam.render_rgb`` — otherwise the view diverges from ``PR/frame_*.png`` /
``rendered_rgb/gs_*.png``. ``--width`` / ``--height`` only rescale output; omit them for pixel-exact agreement with eval.

**Why ``--traj`` with anchor row 0 failed:** ``inv(c2w_i)@c2w_nvs[0]`` only matches SplaTAM if
NVS row 0 equals Replica train row 0; otherwise later frames drift. **Common case:** while
building ``results_habitat``, frames are skipped when ``too_close`` (see
``generate_Replica_NVS_data.py``), so **NVS traj row 0 is the first saved frame**, not necessarily
Replica row 0. Default is Replica train alignment when that file exists (see ``--traj_align``).

Examples
--------
  # Явная w2c (16 чисел) уже в системе чекпойнта
  python scripts/render_view_from_pose.py \\
    --checkpoint results/splatam/final/params.npz \\
    --w2c_flat $(cat w2c_row.txt)

  # replica_sim_nvs: по умолчанию якорь — data/Replica/<scene>/traj.txt (как validate_lang_field_traj)
  python scripts/render_view_from_pose.py \\
    --checkpoint .../params.npz \\
    --traj data/replica_sim_nvs/office0/traj.txt --frame 42

  # Старый режим только по NVS: первая строка того же traj как якорь
  python scripts/render_view_from_pose.py ... --traj ... --frame 1 --traj_align traj_first

  # Replica: c2w из траекта + первая поза Replica как якорь обучения
  python scripts/render_view_from_pose.py \\
    --checkpoint .../params.npz \\
    --replica_c2w_traj path/to/replica_sim_nvs/traj.txt --frame 10 \\
    --train_traj0 data/Replica/room0/traj.txt

  # JSON: ключ "w2c" или корень = список/строка 4x4
  python scripts/render_view_from_pose.py --checkpoint ... --w2c_json keyframe_poses.json --frame_id 6

  # Тот же w2c, что ``run_nvs_validation`` / ``eval_result`` (dataset_eval + align_eval_world),
  # один кадр — без полного eval:
  python scripts/render_view_from_pose.py --from_nvs_eval --time_idx 4 \\
    --cfg configs/Replica/office0/ActiveOpenSem.py \\
    --result_dir results/Replica/office0/ActiveOpenSem/run_0 \\
    --stage eval_exploration_stage_1 \\
    --out single.png --save_with_gt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.query_language_field import (  # noqa: E402
    render_rgb,
    w2c_gaussian_frame_from_replica_c2w,
)
from src.slam.langsplatam.langsplatam import LangSplatam  # noqa: E402


def _eval_clean_suffix(stage: str) -> str:
    return stage[5:] if isinstance(stage, str) and stage.startswith("eval_") else stage


def _resolve_params_npz(config, stage: str) -> str:
    """Same resolution as ``SplatamOurs.load_params`` (checkpoint_time_idx == 0)."""
    base = os.path.join(config["workdir"], config["run_name"])
    stage_dirs = [stage]
    if isinstance(stage, str) and stage.startswith("eval_"):
        stage_dirs.append(stage[5:])
    for d in stage_dirs:
        p = os.path.join(base, d, "params.npz")
        if os.path.isfile(p):
            return p
    tried = "\n  ".join(os.path.join(base, d, "params.npz") for d in stage_dirs)
    raise FileNotFoundError(f"params.npz not found. Tried:\n  {tried}")


def _tensor_rgb_to_bgr_u8(color: torch.Tensor) -> np.ndarray:
    """Dataset color tensor → BGR uint8 for OpenCV (handles HWC / 0..1 / 0..255)."""
    x = color.detach().cpu().float()
    if x.ndim == 3 and x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    arr = x.numpy()
    if arr.max() <= 1.0 + 1e-3:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _render_bgr_like_eval_result(
    slam,
    time_idx: int,
    final_params: dict,
) -> tuple[np.ndarray, np.ndarray, str, torch.Tensor]:
    """Delegate to ``eval_helper.render_rgb_eval_pose`` (must stay in sync with ``eval()``)."""
    from src.slam.splatam.eval_helper import render_rgb_eval_pose

    _align = slam._eval_alignment_kwargs()
    return render_rgb_eval_pose(
        slam.dataset_eval,
        final_params,
        int(time_idx),
        train_first_abs_c2w=_align.get("train_first_abs_c2w"),
        align_eval_world=bool(_align.get("align_eval_world", False)),
    )


def _read_traj(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path.expanduser()), dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, 4, 4)
    return arr.reshape(-1, 4, 4)


def _parse_16_floats(s: str) -> np.ndarray:
    parts = str(s).replace(",", " ").split()
    if len(parts) != 16:
        raise argparse.ArgumentTypeError(f"expected 16 floats, got {len(parts)}")
    return np.array([float(x) for x in parts], dtype=np.float64).reshape(4, 4)


def _infer_replica_train_traj_from_nvs_path(traj_path: Path) -> Path | None:
    """If traj lives under .../replica_sim_nvs/<scene>/, return data/Replica/<scene>/traj.txt when present."""
    try:
        parts = traj_path.resolve().parts
    except Exception:
        parts = traj_path.parts
    for i, p in enumerate(parts):
        if p == "replica_sim_nvs" and i + 1 < len(parts):
            scene = parts[i + 1]
            cand = ROOT / "data" / "Replica" / scene / "traj.txt"
            if cand.is_file():
                return cand
    return None


def _resolve_train_traj0_path(explicit: str | None, nvs_traj: Path) -> tuple[Path | None, str]:
    """Pick Replica train traj for alignment; return (path_or_none, provenance string)."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--train_traj0 not found: {p}")
        return p, f"explicit --train_traj0 {p}"
    inf = _infer_replica_train_traj_from_nvs_path(nvs_traj)
    if inf is not None:
        return inf, f"inferred Replica train traj {inf}"
    return None, "no Replica traj inferred (not under replica_sim_nvs/<scene>/ or file missing)"


def _w2c_from_traj_row(traj: np.ndarray, frame: int, anchor_frame: int) -> np.ndarray:
    if frame < 0 or frame >= traj.shape[0]:
        raise ValueError(f"frame {frame} out of range [0, {traj.shape[0] - 1}]")
    if anchor_frame < 0 or anchor_frame >= traj.shape[0]:
        raise ValueError(f"anchor_frame {anchor_frame} out of range [0, {traj.shape[0] - 1}]")
    c2w_i = traj[frame].astype(np.float64)
    c2w_a = traj[anchor_frame].astype(np.float64)
    return (np.linalg.inv(c2w_i) @ c2w_a).astype(np.float64)


def _load_w2c_json(path: Path, frame_id: int) -> np.ndarray:
    raw = json.loads(path.read_text())
    key = str(frame_id)
    if key not in raw:
        if frame_id in raw:  # type: ignore[operator]
            key = frame_id  # type: ignore[assignment]
        else:
            raise KeyError(
                f'frame_id={frame_id!r} not in JSON. Sample keys: {list(raw.keys())[:12]}...'
            )
    mat = np.asarray(raw[key], dtype=np.float64).reshape(4, 4)
    return mat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render RGB from a pose (same GS render path as query_language_field / suite tests)."
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="SplaTAM params.npz (not needed with --from_nvs_eval).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: render_view.png next to cwd)",
    )
    p.add_argument("--device", default="cuda:0", help="Torch device (falls back to CPU if no CUDA).")

    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override render height (default: org_height from checkpoint)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override render width (default: org_width from checkpoint)",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--w2c_flat",
        type=str,
        default=None,
        help="16 floats (space/comma separated): w2c already in checkpoint frame",
    )
    src.add_argument(
        "--w2c_txt",
        type=str,
        default=None,
        help="Text file with 16 floats (one line): w2c in checkpoint frame",
    )
    src.add_argument(
        "--traj",
        type=str,
        default=None,
        help="traj.txt with c2w rows (same file convention as Replica / NVS exports)",
    )
    src.add_argument(
        "--replica_c2w_traj",
        type=str,
        default=None,
        help="Alias of --traj with traj_align=replica_train0 (--train_traj0 optional if path infers).",
    )
    src.add_argument(
        "--w2c_json",
        type=str,
        default=None,
        help='JSON like keyframe_poses.json: {"0": [[...]], ...} — w2c in checkpoint frame',
    )

    p.add_argument(
        "--frame_id",
        type=int,
        default=None,
        help="Key index for --w2c_json (defaults to --frame if omitted).",
    )

    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Row index for --traj / --replica_c2w_traj, or default JSON key for --w2c_json",
    )
    p.add_argument(
        "--anchor_frame",
        type=int,
        default=0,
        help="With --traj_align traj_first: w2c = inv(c2w[frame]) @ c2w[anchor_frame] (default: 0)",
    )
    p.add_argument(
        "--train_traj0",
        type=str,
        default=None,
        help="Optional explicit data/Replica/<scene>/traj.txt for train frame-0 (c2w_train0). "
        "If omitted and --traj is under replica_sim_nvs/<scene>/, the script infers "
        "data/Replica/<scene>/traj.txt when it exists.",
    )
    p.add_argument(
        "--traj_align",
        type=str,
        choices=("auto", "replica_train0", "traj_first"),
        default="auto",
        help="How to map absolute c2w rows to SplaTAM checkpoint frame: "
        "replica_train0 = inv(c2w)@Replica train first pose (same as eval_helper / validate_lang_field_traj); "
        "traj_first = inv(c2w)@c2w_traj[anchor] (old behavior); "
        "auto = replica_train0 when Replica train traj is found, else traj_first. "
        "Ignored when --replica_c2w_traj is used (always replica_train0).",
    )
    p.add_argument(
        "--pose_diagnostics",
        action="store_true",
        help="Print |NVS[0]-Replica[0]| and distance from computed w2c to checkpoint gt_w2c_all_frames.",
    )
    p.add_argument(
        "--beside_habitat_rgb",
        type=str,
        default=None,
        help="Optional GT image path (e.g. results_habitat/frame000001.jpg): save "
        "horizontal concat [GS render | GT] to --out for viewpoint sanity check.",
    )
    p.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Replica scene id (e.g. office0): sets data/Replica/<scene>/traj.txt as "
        "train anchor when --train_traj0 is omitted.",
    )
    p.add_argument(
        "--align_gs_train_frame",
        action="store_true",
        help="Alias for --traj_align replica_train0 (w2c = inv(c2w_nvs) @ c2w_Replica_train0). "
        "Requires Replica train traj (--scene or --train_traj0).",
    )
    return p.parse_args()


def _apply_pose_compat_flags(args: argparse.Namespace) -> None:
    """Map README-friendly flags to traj_align / train_traj0."""
    if args.scene and not args.train_traj0:
        cand = ROOT / "data" / "Replica" / args.scene / "traj.txt"
        if cand.is_file():
            args.train_traj0 = str(cand)
    if args.align_gs_train_frame:
        args.traj_align = "replica_train0"
        if not args.train_traj0:
            traj_arg = args.traj or args.replica_c2w_traj
            if traj_arg:
                train_path, _ = _resolve_train_traj0_path(None, Path(traj_arg))
                if train_path is not None:
                    args.train_traj0 = str(train_path)
            if not args.train_traj0:
                raise SystemExit(
                    "--align_gs_train_frame: provide --scene, --train_traj0, or a traj path "
                    "under replica_sim_nvs/<scene>/."
                )


def _resolve_w2c(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    """Return (w2c_np, description)."""
    if args.w2c_flat:
        m = _parse_16_floats(args.w2c_flat)
        return m, "CLI w2c (checkpoint frame)"

    if args.w2c_txt:
        text = Path(args.w2c_txt).expanduser().read_text().strip().splitlines()
        if not text:
            raise ValueError(f"empty file: {args.w2c_txt}")
        m = _parse_16_floats(text[0])
        return m, f"w2c from file {args.w2c_txt}"

    traj_arg = args.traj or args.replica_c2w_traj
    if traj_arg:
        traj_path = Path(traj_arg).expanduser()
        traj = _read_traj(traj_path)
        fr = int(args.frame)
        if fr < 0 or fr >= traj.shape[0]:
            raise ValueError(f"frame {fr} out of range [0, {traj.shape[0] - 1}]")
        c2w_i = traj[fr].astype(np.float64)

        align = "replica_train0" if args.replica_c2w_traj else str(args.traj_align)

        if align == "traj_first":
            m = _w2c_from_traj_row(traj, fr, int(args.anchor_frame))
            return (
                m,
                f"traj {traj_path} frame={fr} anchor={args.anchor_frame}  "
                f"w2c=inv(c2w)@c2w_traj[anchor] (traj_align=traj_first)",
            )

        if align == "replica_train0":
            train_path, train_note = _resolve_train_traj0_path(args.train_traj0, traj_path)
            if train_path is None:
                raise FileNotFoundError(
                    "traj_align=replica_train0 needs data/Replica/<scene>/traj.txt — "
                    "pass --train_traj0, or put --traj under .../replica_sim_nvs/<scene>/ "
                    "with that Replica file present."
                )
            c2w_train0 = _read_traj(train_path)[0].astype(np.float64)
            m = w2c_gaussian_frame_from_replica_c2w(c2w_i, c2w_train0)
            return m, (
                f"traj {traj_path} frame={fr}  w2c=inv(c2w_nvs)@c2w_train0  ({train_note})"
            )

        # auto
        train_path, train_note = _resolve_train_traj0_path(args.train_traj0, traj_path)
        if train_path is not None:
            c2w_train0 = _read_traj(train_path)[0].astype(np.float64)
            m = w2c_gaussian_frame_from_replica_c2w(c2w_i, c2w_train0)
            return m, (
                f"traj {traj_path} frame={fr}  w2c=inv(c2w_nvs)@c2w_train0  ({train_note})"
            )
        m = _w2c_from_traj_row(traj, fr, int(args.anchor_frame))
        return m, (
            f"traj {traj_path} frame={fr} anchor={args.anchor_frame}  "
            f"w2c=inv(c2w)@c2w_traj[anchor]  (auto: {train_note})"
        )

    if args.w2c_json:
        fid = args.frame_id if args.frame_id is not None else int(args.frame)
        m = _load_w2c_json(Path(args.w2c_json).expanduser(), int(fid))
        return m, f"w2c_json {args.w2c_json} frame_id={fid}"

    raise RuntimeError("internal: no pose source matched")


def _maybe_print_traj_train0_note(args: argparse.Namespace) -> None:
    traj_arg = args.traj or args.replica_c2w_traj
    if not traj_arg:
        return
    traj_path = Path(traj_arg).expanduser()
    train_path, _ = _resolve_train_traj0_path(args.train_traj0, traj_path)
    if train_path is None:
        return
    traj = _read_traj(traj_path)
    rep0 = _read_traj(train_path)[0]
    d0 = float(np.max(np.abs(traj[0].astype(np.float64) - rep0.astype(np.float64))))
    print(
        f"[pose] max|NVS traj row0 - Replica train0| = {d0:.3e}  -> if ~0, --train_traj0 "
        "barely changes w2c (expected)."
    )


def _maybe_print_pose_diagnostics(args: argparse.Namespace, w2c_np: np.ndarray) -> None:
    if not args.pose_diagnostics:
        return
    if not getattr(args, "checkpoint", None):
        return
    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        return
    z = np.load(ckpt_path, allow_pickle=True)
    if "gt_w2c_all_frames" not in z.files:
        print("[pose] checkpoint has no gt_w2c_all_frames — skip SLAM distance diagnostic.")
        return
    gt = z["gt_w2c_all_frames"]
    if not isinstance(gt, np.ndarray) or gt.ndim != 3:
        return
    w = np.asarray(w2c_np, dtype=np.float64).reshape(4, 4)
    errs = [float(np.linalg.norm(w.ravel() - gt[i].ravel())) for i in range(len(gt))]
    j = int(np.argmin(errs))
    print(
        f"[pose] min ‖w2c − gt_w2c_all_frames[k]‖_F over k∈[0,{len(gt)-1}] = {errs[j]:.4f} at k={j}. "
        "Орбита NVS (replica_sim_nvs) обычно не совпадает с траекторией SLAM при обучении — "
        "рендер GS по позе из NVS ≠ Habitat пиксель в пиксель даже при верной математике w2c."
    )


def main_from_nvs_eval(nv: argparse.Namespace) -> None:
    """One-frame NVS render: same ``dataset_eval`` / ``gt_w2c`` as ``run_nvs_validation`` (``eval_result``)."""
    if nv.time_idx is None:
        raise SystemExit("error: --from_nvs_eval requires --time_idx N (dataset_eval index).")

    os.chdir(ROOT)
    sys.path.insert(0, os.getcwd())

    from tensorboardX import SummaryWriter

    from src.naruto.cfg_loader import argument_parsing, load_cfg
    from src.slam import init_SLAM_model
    from src.utils.general_utils import InfoPrinter, fix_random_seed

    info_printer = InfoPrinter("render_view_from_pose_nvs")
    info_printer("Parsing arguments (NVS eval / single frame)...", 0, "Initialization")
    args = argument_parsing()
    main_cfg = load_cfg(args)

    fix_random_seed(getattr(main_cfg.general, "seed", 0))
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(log_savedir)

    info_printer("Initializing SLAM (loads dataset_eval)...", 0, "Initialization")
    slam = init_SLAM_model(main_cfg, info_printer, logger)
    slam.load_params(stage=args.stage)

    bgr, w2c_np, align_note, color = _render_bgr_like_eval_result(
        slam, int(nv.time_idx), slam.params
    )
    ckpt = _resolve_params_npz(slam.config, args.stage)
    suff = _eval_clean_suffix(args.stage)

    if nv.height is not None or nv.width is not None:
        h0, w0 = bgr.shape[:2]
        W = int(nv.width) if nv.width is not None else w0
        H = int(nv.height) if nv.height is not None else h0
        if (H, W) != (h0, w0):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
            print(
                f"[from_nvs_eval] resized render to {W}x{H} (eval/gs_*.png use dataset resolution "
                f"{w0}x{h0}; omit --width/--height for a pixel match)"
            )

    H, W = bgr.shape[:2]

    to_write = bgr
    if nv.save_with_gt:
        gt_bgr = _tensor_rgb_to_bgr_u8(color)
        if gt_bgr.shape[:2] != (H, W):
            gt_bgr = cv2.resize(gt_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        to_write = cv2.hconcat([bgr, gt_bgr])
        print("[out] side-by-side [GS | dataset_eval RGB] (same pairing as eval_result / run_nvs_validation)")

    out_path = Path(nv.out).expanduser() if nv.out else Path.cwd() / "render_view.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), to_write)

    result_dir = os.path.normpath(main_cfg.dirs.result_dir)
    print(f"[from_nvs_eval] time_idx={int(nv.time_idx)}  stage={args.stage}  eval_dir_suffix={suff}")
    print(f"[from_nvs_eval] checkpoint: {ckpt}")
    print(f"[from_nvs_eval] {align_note}")
    np.set_printoptions(precision=6, suppress=True)
    print("w2c (4×4, checkpoint frame):\n", np.asarray(w2c_np, dtype=np.float64))
    print(f"Rendered {W}x{H} → {out_path.resolve()}")
    print(f"[from_nvs_eval] SplaTAM eval folder (full NVS run): {result_dir}/splatam/eval_{suff}/")


def main() -> None:
    if "--from_nvs_eval" in sys.argv:
        nv_p = argparse.ArgumentParser(add_help=False)
        nv_p.add_argument("--from_nvs_eval", action="store_true")
        nv_p.add_argument("--time_idx", type=int, default=None)
        nv_p.add_argument("--save_with_gt", action="store_true")
        nv_p.add_argument("--out", type=str, default=None)
        nv_p.add_argument("--height", type=int, default=None)
        nv_p.add_argument("--width", type=int, default=None)
        nv_p.add_argument("--device", default="cuda:0")
        nv, rest = nv_p.parse_known_args(sys.argv[1:])
        if not nv.from_nvs_eval:
            raise RuntimeError("internal: expected --from_nvs_eval")
        sys.argv = [sys.argv[0]] + rest
        main_from_nvs_eval(nv)
        return

    args = parse_args()
    _apply_pose_compat_flags(args)
    if not args.checkpoint:
        raise SystemExit(
            "error: specify --checkpoint, or use --from_nvs_eval with --time_idx and the usual "
            "--cfg / --result_dir / --stage (same as run_nvs_validation)."
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    w2c_np, pose_desc = _resolve_w2c(args)
    w2c = torch.tensor(w2c_np, dtype=torch.float32, device=device)

    _maybe_print_traj_train0_note(args)

    model = LangSplatam(
        checkpoint_path=str(Path(args.checkpoint).expanduser()),
        latent_dim=3,
        device=str(device),
    )
    H0 = int(model.params["org_height"])
    W0 = int(model.params["org_width"])
    H = int(args.height) if args.height is not None else H0
    W = int(args.width) if args.width is not None else W0

    bgr = render_rgb(model, w2c, H, W)

    to_write = bgr
    if args.beside_habitat_rgb:
        gt_path = Path(args.beside_habitat_rgb).expanduser()
        gt_bgr = cv2.imread(str(gt_path))
        if gt_bgr is None:
            raise FileNotFoundError(f"cannot read --beside_habitat_rgb: {gt_path}")
        if gt_bgr.shape[:2] != (H, W):
            gt_bgr = cv2.resize(gt_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        to_write = cv2.hconcat([bgr, gt_bgr])
        print(f"[out] wrote side-by-side [GS | {gt_path.name}]")

    out_path = Path(args.out).expanduser() if args.out else Path.cwd() / "render_view.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), to_write)

    print(f"Pose source: {pose_desc}")
    np.set_printoptions(precision=6, suppress=True)
    print("w2c (4×4, checkpoint frame):\n", np.asarray(w2c_np, dtype=np.float64))
    print(f"Rendered {W}x{H} → {out_path.resolve()}")
    _maybe_print_pose_diagnostics(args, w2c_np)


if __name__ == "__main__":
    main()