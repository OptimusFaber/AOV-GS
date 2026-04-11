"""
Обучение языкового поля на замороженных гауссианах (LangSplat-совместимый).

Предварительные условия:
  1. Запущен ActiveSGM → results/{scene}/params*.npz + keyframe_poses.json
  2. Запущен SAMCLIPExtractor → results/{scene}/language_features/*_s.npy, *_f.npy
  3. Запущен train_language_autoencoder.py → results/{scene}/language_features_dim{D}/

Usage (64d, дефолт)
-------------------
    python scripts/train_language_field.py \\
        --checkpoint    results/room0/splatam/final/params0.npz \\
        --poses         results/room0/keyframe_poses.json \\
        --features_dir  results/room0/language_features_dim64 \\
        --level         s \\
        --output_dir    results/room0/lang_field_s \\
        --num_iters     30000 \\
        --latent_dim    64 \\
        --device        cuda:0
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Language-field fine-tuning on top of a frozen SplaTAM checkpoint."
    )
    p.add_argument("--checkpoint", required=True,
                   help="params*.npz от ActiveSGM.")
    p.add_argument("--poses", required=True,
                   help="keyframe_poses.json (frame_id → 4×4 w2c).")
    p.add_argument("--features_dir", required=True,
                   help="Папка с сжатыми фичами (после train_language_autoencoder.py), "
                        "напр. language_features_dim64/. "
                        "Для обратной совместимости принимается и --features_dim3.")
    p.add_argument("--features_dim3",
                   help="(устарело) Псевдоним для --features_dir.")
    p.add_argument("--level", default="s", choices=["default", "s", "m", "l"],
                   help="Уровень SAM (default=0, s=1, m=2, l=3). По умолчанию s.")
    p.add_argument("--output_dir", required=True,
                   help="Куда сохранить lang_field.pt.")
    p.add_argument("--latent_dim", type=int, default=64,
                   help="Размер латентного вектора (должен совпадать с AE). "
                        "64 для 64d режима, 3 для LangSplat-совместимого.")
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
        help="Как рендерить multi-pass языковую карту (D>3): "
             "'off' — одна большая склейка графа (все проходы в памяти, быстро, много VRAM); "
             "'on' — gradient checkpoint на каждый 3-канальный проход (мало VRAM, медленнее); "
             "'auto' — checkpoint только при latent_dim==64 (как раньше).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --features_dim3 is a legacy alias for --features_dir
    features_dir = args.features_dir or args.features_dim3
    if features_dir is None:
        raise ValueError("Укажите --features_dir (или устаревший --features_dim3).")

    logger.info("Загружаем LangSplatam из: %s", args.checkpoint)
    model = LangSplatam(
        checkpoint_path=args.checkpoint,
        latent_dim=args.latent_dim,
        device=args.device,
        render_checkpoint=args.render_checkpoint,
    )

    logger.info("Запускаем обучение language field (level=%s, iters=%d, latent_dim=%d)...",
                args.level, args.num_iters, args.latent_dim)
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
    )

    logger.info("Готово. Language field сохранён в: %s", args.output_dir)


if __name__ == "__main__":
    main()
