"""
Train language-field autoencoder (LangSplat-compatible).

Matches LangSplat/autoencoder/train.py + test.py:

Step 1 — train:
  Reads all *_f.npy from language_features/
  Trains Autoencoder (512 → 3 → 512)
  Saves best checkpoint best_ckpt.pth

Step 2 — test (apply encoder):
  Runs all *_f.npy through encoder → (N, 3)
  Copies *_s.npy unchanged
  Saves to language_features_dim3/

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
# Dataset — analogue of LangSplat/autoencoder/dataset.py
# ---------------------------------------------------------------------------

class LanguageFeatureDataset(Dataset):
    """
    Reads all *_f.npy from data_dir, concatenates into one large array.
    Remembers mask count per file (data_dic) — needed for test.py.
    """
    def __init__(self, data_dir: str) -> None:
        data_names = sorted(glob.glob(os.path.join(data_dir, '*_f.npy')))
        if not data_names:
            raise FileNotFoundError(f"No *_f.npy files in {data_dir}")

        self.data_dic = {}  # {stem: n_masks}
        arrays = []
        for path in data_names:
            feats = np.load(path).astype(np.float32)  # (N, 512)
            stem = os.path.basename(path).split('.')[0]
            self.data_dic[stem] = feats.shape[0]
            if feats.shape[0] > 0:
                arrays.append(feats)

        self.data = np.concatenate(arrays, axis=0) if arrays else np.zeros((0, 512), dtype=np.float32)
        logger.info("Loaded %d masks from %d files.", self.data.shape[0], len(data_names))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.data[index])


# ---------------------------------------------------------------------------
# Step 1: Training — analogue of LangSplat/autoencoder/train.py
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

    logger.info("Best epoch: %d  loss=%.8f", best_epoch, best_loss)
    logger.info("Checkpoint saved → %s", os.path.join(ckpt_dir, 'best_ckpt.pth'))


# ---------------------------------------------------------------------------
# Step 2: Apply encoder — analogue of LangSplat/autoencoder/test.py
# ---------------------------------------------------------------------------

def test(args: argparse.Namespace) -> None:
    data_dir = os.path.join(args.dataset_path, 'language_features')
    latent_dim = args.encoder_dims[-1]
    output_dir = os.path.join(args.dataset_path, f'language_features_dim{latent_dim}')
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join('ckpt', args.dataset_name, str(latent_dim), 'best_ckpt.pth')

    # Copy *_s.npy unchanged (as in LangSplat test.py)
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

    # Save per file — restore split as in test.py
    start = 0
    for stem, n in dataset.data_dic.items():
        path = os.path.join(output_dir, stem + '.npy')
        np.save(path, features_encoded[start:start + n])
        start += n

    logger.info("Compressed features (dim=%d) saved to %s", latent_dim, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LangSplat-compatible language autoencoder.")
    p.add_argument('--dataset_path', required=True,
                   help="Dataset root folder (must contain language_features/).")
    p.add_argument('--dataset_name', required=True,
                   help="Scene name, used for ckpt/{dataset_name}/ folder.")
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--encoder_dims', nargs='+', type=int, default=[256, 128, 64],
                   help="Encoder layer dims. Last = latent_dim. "
                        "Default [256,128,64] → 64d. "
                        "For LangSplat-compatible 3d: 256 128 64 32 3")
    p.add_argument('--decoder_dims', nargs='+', type=int, default=[128, 256, 512],
                   help="Decoder layer dims. Last must be 512. "
                        "Default [128,256,512]. "
                        "For LangSplat-compatible 3d: 16 32 64 128 256 256 512")
    p.add_argument(
        '--device',
        default='cuda:0',
        help='Torch device for AE training (default cuda:0).',
    )
    p.add_argument('--skip_train', action='store_true',
                   help="Skip training; only apply an already trained encoder.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_train:
        logger.info("=== Step 1: Train autoencoder ===")
        train(args)
    logger.info("=== Step 2: Apply encoder (→ language_features_dim3/) ===")
    test(args)
    logger.info("Done.")


if __name__ == '__main__':
    main()
