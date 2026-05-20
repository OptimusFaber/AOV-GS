#!/usr/bin/env python3
"""
Run NVS (replica_sim_nvs) validation on a trained SplaTAM checkpoint — same pipeline as
``src/evaluation/eval_nvs_result.py`` (load cfg → init SLAM → ``load_params`` → ``eval_result``).

Writes **GT** and **predicted RGB** into a user-chosen directory and prints aggregate metrics
(PSNR, depth RMSE/L1, MS-SSIM, LPIPS) to the terminal (also saved under the SplaTAM eval folder).

Example
-------
  cd AOV-GS-V2

  python scripts/run_nvs_validation.py \\
    --export_dir results/nvs_val_run0 \\
    --cfg configs/Replica/office0/ActiveOpenSem.py \\
    --result_dir results/Replica/office0/ActiveOpenSem/run_0 \\
    --stage eval_exploration_stage_1

Output layout (``--export_dir``)::

  <export_dir>/
    GT/    frame_XXXX.png   # ground-truth RGB from dataset
    PR/    frame_XXXX.png   # rendered RGB (Gaussian splat)
    metrics_summary.txt    # copy of render_result.txt from eval if present

Intermediate / full eval artifacts remain under::

  <result_dir>/splatam/eval_<stage>/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def _patch_argv_for_export_flags() -> tuple[str | None, int | None, bool]:
    """Parse --export_dir / --max_frames / --eval_first_frame before naruto's parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="Create GT/ and PR/ RGB folders here (optional).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Cap number of dataset frames (passed to slam.eval_result).",
    )
    parser.add_argument(
        "--eval_first_frame",
        action="store_true",
        help="Include time index 0 in metrics (default: ignore first frame, like eval_nvs_result.py).",
    )
    known, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    ignore_first = not known.eval_first_frame
    return known.export_dir, known.max_frames, ignore_first


def _eval_clean_suffix(stage: str) -> str:
    return stage[5:] if isinstance(stage, str) and stage.startswith("eval_") else stage


def _export_gt_pr_rgb(export_root: str, result_dir: str, stage: str) -> None:
    """Rename-copy ``rgb/gt_*`` → ``GT/frame_*`` and ``rendered_rgb/gs_*`` → ``PR/frame_*``.

    Predicted PNGs come **verbatim** from ``eval_helper.eval(save_frames=True)`` (same
    pipeline as ``rendered_rgb/gs_%04d.png``), **not** from ``LangSplatam`` / ``render_rgb``.
    """
    suff = _eval_clean_suffix(stage)
    eval_dir = os.path.join(result_dir, "splatam", f"eval_{suff}")
    rgb_dir = os.path.join(eval_dir, "rgb")
    rend_dir = os.path.join(eval_dir, "rendered_rgb")
    gt_out = os.path.join(export_root, "GT")
    pr_out = os.path.join(export_root, "PR")
    os.makedirs(gt_out, exist_ok=True)
    os.makedirs(pr_out, exist_ok=True)

    if not os.path.isdir(rgb_dir):
        print(f"[run_nvs_validation] Warning: missing GT rgb dir: {rgb_dir}")
        return
    n = 0
    for fn in sorted(os.listdir(rgb_dir)):
        if not (fn.startswith("gt_") and fn.endswith(".png")):
            continue
        mid = fn[3:-4]
        s_gt = os.path.join(rgb_dir, fn)
        s_pr = os.path.join(rend_dir, f"gs_{mid}.png")
        d_gt = os.path.join(gt_out, f"frame_{mid}.png")
        d_pr = os.path.join(pr_out, f"frame_{mid}.png")
        shutil.copy2(s_gt, d_gt)
        if os.path.isfile(s_pr):
            shutil.copy2(s_pr, d_pr)
        else:
            print(f"[run_nvs_validation] Warning: no predicted frame for index {mid}: {s_pr}")
        n += 1
    print(f"[run_nvs_validation] Exported {n} RGB pairs → {export_root}/{{GT,PR}}")

    rr = os.path.join(eval_dir, "render_result.txt")
    if os.path.isfile(rr):
        dst = os.path.join(export_root, "metrics_summary.txt")
        shutil.copy2(rr, dst)
        print(f"[run_nvs_validation] Copied metrics → {dst}")


def _print_metrics_tail(result_dir: str, stage: str) -> None:
    suff = _eval_clean_suffix(stage)
    rr = os.path.join(result_dir, "splatam", f"eval_{suff}", "render_result.txt")
    if os.path.isfile(rr):
        print()
        print("========== metrics_summary (render_result.txt) ==========")
        with open(rr, "r", encoding="utf-8", errors="replace") as f:
            print(f.read().rstrip())
        print("===========================================================")
        print()


def main() -> None:
    export_dir, max_frames, ignore_first_frame = _patch_argv_for_export_flags()

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.getcwd())

    from tensorboardX import SummaryWriter

    from src.naruto.cfg_loader import argument_parsing, load_cfg, save_cfg_to_json
    from src.slam import init_SLAM_model
    from src.utils.general_utils import InfoPrinter, fix_random_seed

    info_printer = InfoPrinter("run_nvs_validation")

    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)
    os.makedirs(main_cfg.dirs.result_dir, exist_ok=True)
    save_cfg_to_json(main_cfg, os.path.join(main_cfg.dirs.result_dir, "main_cfg.json"))
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(getattr(main_cfg.general, "seed", 0))

    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(log_savedir)

    info_printer("Initializing SLAM...", 0, "Initialization")
    slam = init_SLAM_model(main_cfg, info_printer, logger)
    slam.load_params(stage=args.stage)

    suffix = _eval_clean_suffix(args.stage)
    if args.stage.startswith("eval_") and suffix != args.stage:
        print(
            f"[run_nvs_validation] eval_dir_suffix: '{suffix}' "
            f"(from stage '{args.stage}', same rule as eval_nvs_result.py)"
        )

    info_printer("Running eval_result (NVS dataset, save_frames=True)...", 0, "Eval")
    slam.eval_result(
        eval_dir_suffix=suffix,
        ignore_first_frame=ignore_first_frame,
        save_frames=True,
        max_frames=max_frames,
    )

    result_dir = os.path.normpath(main_cfg.dirs.result_dir)
    if export_dir:
        export_dir = os.path.abspath(os.path.expanduser(export_dir))
        os.makedirs(export_dir, exist_ok=True)
        _export_gt_pr_rgb(export_dir, result_dir, args.stage)

    _print_metrics_tail(result_dir, args.stage)
    print(f"[run_nvs_validation] Done. SplaTAM eval dir: {result_dir}/splatam/eval_{suffix}/")


if __name__ == "__main__":
    main()
