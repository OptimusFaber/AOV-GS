"""
Train LangSplatV2-style language field on frozen SplaTAM Gaussians.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from lang_pipeline_utils import resolve_use_langsplat_v2  # noqa: E402
from src.slam.langsplatam.langsplatam import LangSplatam

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _warn_nonzero_cuda_device(device: str) -> None:
    """Rasterizer extension is only reliable on logical cuda:0."""
    import re

    m = re.match(r"cuda:(\d+)$", device)
    if m and int(m.group(1)) != 0:
        idx = m.group(1)
        logger.warning(
            "device=%s: diff_gaussian_rasterization often crashes on cuda:N (N>0). "
            "Use the shell script (auto-remaps) or run manually:\n"
            "  CUDA_VISIBLE_DEVICES=%s python %s ... --device cuda:0",
            device,
            idx,
            Path(__file__).name,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Language-field fine-tuning on top of a frozen SplaTAM checkpoint."
    )
    p.add_argument("--checkpoint", required=True,
                   help="params*.npz от ActiveSGM.")
    p.add_argument("--poses", required=True,
                   help="keyframe_poses.json (frame_id → 4×4 w2c).")
    p.add_argument("--features_dir", required=True,
                   help="Папка c raw SAM+CLIP фичами, напр. language_features/.")
    p.add_argument("--features_dim3",
                   help="(устарело) Псевдоним для --features_dir.")
    p.add_argument("--level", default="s", choices=["default", "s", "m", "l"],
                   help="Уровень SAM (default=0, s=1, m=2, l=3). По умолчанию s.")
    p.add_argument("--output_dir", required=True,
                   help="Куда сохранить lang_field.pt.")
    p.add_argument("--latent_dim", type=int, default=64,
                   help="(legacy) Для обратной совместимости. "
                        "В режиме V2 не используется для декодера.")
    p.add_argument("--vq_layer_num", type=int, default=1,
                   help="Число уровней residual VQ (L).")
    p.add_argument("--codebook_size", type=int, default=64,
                   help="Размер codebook на уровень (K).")
    p.add_argument("--topk", type=int, default=4,
                   help="Сколько коэффициентов top-k оставлять на уровень.")
    p.add_argument("--max_init_features", type=int, default=200000,
                   help="Макс. число CLIP-векторов для KMeans инициализации.")
    p.add_argument("--train_downscale", type=float, default=1.0,
                   help="Масштаб рендера при обучении (0,1]. "
                        "Напр. 0.5 сильно снижает VRAM.")
    p.add_argument("--legacy", action="store_true",
                   help="(устар.) То же, что --lang_mode langsplat.")
    p.add_argument(
        "--lang_mode",
        choices=("langsplatv2", "langsplat", "auto"),
        default="langsplatv2",
        help="Сценарий после сбора масок: langsplatv2 (codebook, default) или "
             "langsplat (legacy AE + language_features_dim*).",
    )
    p.add_argument("--num_iters", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_cos", type=float, default=1.0)
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for lang-field training (default cuda:0). "
        "Rasterizer requires logical cuda:0; use CUDA_VISIBLE_DEVICES to pick a physical GPU.",
    )
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
    _warn_nonzero_cuda_device(args.device)

    # --features_dim3 is a legacy alias for --features_dir
    features_dir = args.features_dir or args.features_dim3
    if features_dir is None:
        raise ValueError("Укажите --features_dir (или устаревший --features_dim3).")

    use_v2 = resolve_use_langsplat_v2(args.lang_mode, legacy_flag=bool(args.legacy))

    logger.info("Загружаем LangSplatam из: %s", args.checkpoint)
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
        "Запускаем обучение language field (mode=%s, level=%s, iters=%d, L=%d, K=%d, topk=%d)...",
        "LangSplatV2" if use_v2 else "LangSplat",
        args.level,
        args.num_iters,
        args.vq_layer_num,
        args.codebook_size,
        args.topk,
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
        use_langsplat_v2=use_v2,
        max_init_features=args.max_init_features,
        train_downscale=args.train_downscale,
    )

    logger.info("Готово. Language field сохранён в: %s", args.output_dir)


if __name__ == "__main__":
    main()
