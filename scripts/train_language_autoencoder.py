"""
Обучение автоэнкодера языкового поля (LangSplat-совместимый).

Соответствует LangSplat/autoencoder/train.py + test.py:

Шаг 1 — train:
  Читает все *_f.npy из language_features/
  Обучает Autoencoder (512 → 3 → 512)
  Сохраняет лучший чекпоинт best_ckpt.pth

Шаг 2 — test (применение энкодера):
  Прогоняет все *_f.npy через encoder → (N, 3)
  Копирует *_s.npy без изменений
  Сохраняет в language_features_dim3/

Usage
-----
    python scripts/train_language_autoencoder.py \\
        --dataset_path  results/room0 \\
        --dataset_name  room0 \\
        --num_epochs    100 \\
        --lr            1e-4 \\
        --device        cuda:0
"""

import argparse
import glob
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.semantic.language_autoencoder import Autoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset — аналог LangSplat/autoencoder/dataset.py
# ---------------------------------------------------------------------------

class LanguageFeatureDataset(Dataset):
    """
    Читает все *_f.npy из data_dir, конкатенирует в один большой массив.
    Запоминает количество масок на файл (data_dic) — нужно для test.py.
    """
    def __init__(self, data_dir: str) -> None:
        data_names = sorted(glob.glob(os.path.join(data_dir, '*_f.npy')))
        if not data_names:
            raise FileNotFoundError(f"Нет *_f.npy файлов в {data_dir}")

        self.data_dic = {}  # {stem: n_masks}
        arrays = []
        for path in data_names:
            feats = np.load(path).astype(np.float32)  # (N, 512)
            stem = os.path.basename(path).split('.')[0]
            self.data_dic[stem] = feats.shape[0]
            if feats.shape[0] > 0:
                arrays.append(feats)

        self.data = np.concatenate(arrays, axis=0) if arrays else np.zeros((0, 512), dtype=np.float32)
        logger.info("Загружено %d масок из %d файлов.", self.data.shape[0], len(data_names))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.data[index])


# ---------------------------------------------------------------------------
# Шаг 1: Обучение — аналог LangSplat/autoencoder/train.py
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    data_dir = os.path.join(args.dataset_path, 'language_features')
    latent_dim = args.encoder_dims[-1]
    ckpt_dir = os.path.join('ckpt', args.dataset_name, str(latent_dim))
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = LanguageFeatureDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=False)
    test_loader  = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, drop_last=False)

    model = Autoencoder(args.encoder_dims, args.decoder_dims).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 100.0
    best_epoch = 0

    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        model.train()
        for feat in train_loader:
            data = feat.to(args.device)
            z = model.encode(data)
            out = model.decode(z)
            loss = Autoencoder.l2_loss(out, data) + 0.001 * Autoencoder.cos_loss(out, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch > args.num_epochs - 5:
            model.eval()
            eval_loss = 0.0
            for feat in test_loader:
                data = feat.to(args.device)
                with torch.no_grad():
                    out = model(data)
                eval_loss += (Autoencoder.l2_loss(out, data) + Autoencoder.cos_loss(out, data)).item() * len(feat)
            eval_loss /= len(dataset)
            logger.info("Epoch %d  eval_loss=%.8f", epoch, eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_ckpt.pth'))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{epoch}_ckpt.pth'))

    logger.info("Лучший epoch: %d  loss=%.8f", best_epoch, best_loss)
    logger.info("Чекпоинт сохранён → %s", os.path.join(ckpt_dir, 'best_ckpt.pth'))


# ---------------------------------------------------------------------------
# Шаг 2: Применение энкодера — аналог LangSplat/autoencoder/test.py
# ---------------------------------------------------------------------------

def test(args: argparse.Namespace) -> None:
    data_dir = os.path.join(args.dataset_path, 'language_features')
    latent_dim = args.encoder_dims[-1]
    output_dir = os.path.join(args.dataset_path, f'language_features_dim{latent_dim}')
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join('ckpt', args.dataset_name, str(latent_dim), 'best_ckpt.pth')

    # Копируем *_s.npy без изменений (как в LangSplat test.py)
    for fname in os.listdir(data_dir):
        if fname.endswith('_s.npy'):
            shutil.copy(os.path.join(data_dir, fname), os.path.join(output_dir, fname))

    dataset = LanguageFeatureDataset(data_dir)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, drop_last=False)

    model = Autoencoder(args.encoder_dims, args.decoder_dims).to(args.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    model.eval()

    all_encoded = []
    for feat in tqdm(loader, desc="Encoding"):
        data = feat.to(args.device)
        with torch.no_grad():
            z = model.encode(data).cpu().numpy()
        all_encoded.append(z)

    features_encoded = np.concatenate(all_encoded, axis=0)  # (N_total, latent_dim)

    # Сохраняем по файлам — восстанавливаем разбиение как в test.py
    start = 0
    for stem, n in dataset.data_dic.items():
        path = os.path.join(output_dir, stem + '.npy')
        np.save(path, features_encoded[start:start + n])
        start += n

    logger.info("Сжатые фичи (dim=%d) сохранены в %s", latent_dim, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LangSplat-compatible language autoencoder.")
    p.add_argument('--dataset_path', required=True,
                   help="Корневая папка датасета (там должна быть language_features/).")
    p.add_argument('--dataset_name', required=True,
                   help="Имя сцены, используется для папки ckpt/{dataset_name}/.")
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--encoder_dims', nargs='+', type=int, default=[256, 128, 64],
                   help="Размерности слоёв энкодера. Последний = latent_dim. "
                        "По умолчанию [256,128,64] → 64d. "
                        "Для LangSplat-совместимого 3d: 256 128 64 32 3")
    p.add_argument('--decoder_dims', nargs='+', type=int, default=[128, 256, 512],
                   help="Размерности слоёв декодера. Последний должен быть 512. "
                        "По умолчанию [128,256,512]. "
                        "Для LangSplat-совместимого 3d: 16 32 64 128 256 256 512")
    p.add_argument(
        '--device',
        default='cuda:0',
        help='Torch device for AE training (default cuda:0).',
    )
    p.add_argument('--skip_train', action='store_true',
                   help="Пропустить обучение, только применить уже обученный энкодер.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_train:
        logger.info("=== Шаг 1: Обучение автоэнкодера ===")
        train(args)
    logger.info("=== Шаг 2: Применение энкодера (→ language_features_dim3/) ===")
    test(args)
    logger.info("Готово.")


if __name__ == '__main__':
    main()
