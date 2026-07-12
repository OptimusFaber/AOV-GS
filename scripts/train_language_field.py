"""
Train LangSplatV2-style language field on frozen SplaTAM Gaussians.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.slam.langsplatam.langsplatam import LangSplatam

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Language-field fine-tuning on top of a frozen SplaTAM checkpoint."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--features_dir", required=True)
    p.add_argument("--features_dim3", help="legacy alias for --features_dir")
    p.add_argument("--level", default="s", choices=["default", "s", "m", "l"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--vq_layer_num", type=int, default=1)
    p.add_argument("--codebook_size", type=int, default=64)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--max_init_features", type=int, default=200000)
    p.add_argument("--train_downscale", type=float, default=1.0)
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--num_iters", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_cos", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument(
        "--render_checkpoint",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
    )
    return p.parse_args()


def main():
    args = parse_args()

    features_dir = args.features_dir or args.features_dim3
    if features_dir is None:
        raise ValueError("Specify --features_dir")

    logger.info("Loading LangSplatam from: %s", args.checkpoint)
    model = LangSplatam(
        checkpoint_path=args.checkpoint,
        latent_dim=args.latent_dim,
        device=args.device,
        render_checkpoint=args.render_checkpoint,
        vq_layer_num=args.vq_layer_num,
        codebook_size=args.codebook_size,
        topk=args.topk,
    )

    logger.info(
        "Training language field (mode=%s, level=%s, iters=%d)",
        "legacy" if args.legacy else "LangSplatV2",
        args.level,
        args.num_iters,
    )
    model.train_language_field(
        language_features_dir=Path(features_dir),
        poses_file=Path(args.poses),
        level=args.level,
        num_iters=args.num_iters,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        lambda_cos=args.lambda_cos,
        output_dir=Path(args.output_dir),
        log_every=args.log_every,
        use_langsplat_v2=not args.legacy,
        max_init_features=args.max_init_features,
        train_downscale=args.train_downscale,
    )

    logger.info("Done: %s", args.output_dir)


if __name__ == "__main__":
    main()
