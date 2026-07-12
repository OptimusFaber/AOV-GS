"""
Evaluate language autoencoder quality via semantic segmentation metrics.

What is measured
-----------
1. **Reconstruction cosine** — mean cos(original_512, decode(encode(original_512)))
   over all masks of all frames. Shows how well the AE preserves information.

1b. **Comparison to original 512** (same masks):
   - **MSE / RMSE** on vector differences (after L2-norm, as at runtime);
   - **mean L2** — mean Euclidean error ||orig − dec||₂;
   - **MAE** over coordinates; **median / p5 / p95 cosine**.

2. **Retrieval mAP** (mean Average Precision) — for each text query:
   - rank masks by cosine to original_512  → "ground-truth" relevance
   - rank masks by cosine to decoded_512   → predicted relevance
   - compare AP@top_k to measure how well the AE preserves ranking.

3. **Rank correlation** (Spearman ρ) — over all masks and all queries:
   rank by original vs rank by decoded. Shows preservation of semantic order.

4. **Pixel segmentation IoU** — for each frame, for each query:
   - build binary mask: top-T% pixels by cos to original → GT mask
   - build binary mask: top-T% pixels by cos to decoded → pred mask
   - IoU(pred, GT)

Usage
-----
python scripts/eval_language_autoencoder.py \\
    --features_orig  results/language_features \\
    --ae_ckpt        ckpt/room0/best_ckpt.pth \\
    # optional for IoU: --features_enc results/language_features_dim64 \\

python scripts/eval_language_autoencoder.py \\
    --features_orig  results/language_features \\
    --features_enc   results/language_features_dim64 \\
    --ae_ckpt        ckpt/room0/best_ckpt.pth \\
    --queries        "a sofa" "a table" "a plant" "the floor" \\
    --feature_level  1 \\
    --top_pct        10 \\
    --device         cuda:0
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.semantic.language_autoencoder import Autoencoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frame(orig_dir: str, frame_id: str, feature_level: int):
    """
    Returns (features_512, pixel_features_512) for one frame.

    features_512     : (N_masks, 512) — one CLIP vector per mask
    pixel_features   : (H*W, 512)    — pixel-wise (invalid pixels = zero)
    valid_pixels     : (H*W,) bool
    """
    s_path = os.path.join(orig_dir, f"{frame_id}_s.npy")
    f_path = os.path.join(orig_dir, f"{frame_id}_f.npy")
    seg   = np.load(s_path)                     # (4, H, W)  int32
    feats = np.load(f_path).astype(np.float32)  # (N, 512)
    if feats.shape[0] == 0:
        return None, None, None

    H, W = seg.shape[1], seg.shape[2]
    seg_level = seg[feature_level]              # (H, W)
    valid = seg_level >= 0                      # (H, W)

    pix_feats = np.zeros((H * W, feats.shape[1]), dtype=np.float32)
    pix_feats[valid.ravel()] = feats[seg_level[valid]]

    return feats, pix_feats, valid.ravel()


def encode_texts(queries: List[str], clip_model: str, clip_pretrained: str,
                 device: torch.device) -> torch.Tensor:
    """Returns (Q, 512) unit-norm text embeddings."""
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device)
    clip.eval()
    tok = open_clip.get_tokenizer(clip_model)
    with torch.no_grad():
        embs = F.normalize(clip.encode_text(tok(queries).to(device)), dim=-1)
    return embs.cpu()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def eval_ae(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # Same defaults as src.semantic.language_autoencoder.Autoencoder
    encoder_dims = args.encoder_dims if args.encoder_dims is not None else [256, 128, 64]
    decoder_dims = args.decoder_dims if args.decoder_dims is not None else [128, 256, 512]

    ae = Autoencoder(encoder_dims, decoder_dims).to(device)
    state = torch.load(args.ae_ckpt, map_location=device)
    ae.load_state_dict(state)
    ae.eval()
    latent_dim = encoder_dims[-1]
    print(f"AE loaded: 512 → {latent_dim} → 512  from {args.ae_ckpt}")

    # Encode text queries
    print(f"Encoding {len(args.queries)} text queries via CLIP …")
    text_embs = encode_texts(
        args.queries, args.clip_model, args.clip_pretrained, device
    )  # (Q, 512)

    # Collect all masks: original 512d, decoded 512d
    orig_dir = os.path.abspath(args.features_orig)
    frames = sorted(glob.glob(os.path.join(orig_dir, "*_f.npy")))
    frame_ids = [os.path.basename(f).replace("_f.npy", "") for f in frames]

    if len(frame_ids) == 0:
        print(
            f"\n[error] No *_f.npy files under:\n  {orig_dir}\n\n"
            "  Expected: **raw 512-d CLIP mask embeddings** in LangSplat format:\n"
            "    {frame_id:06d}_f.npy  and  {frame_id:06d}_s.npy\n"
            "  Written by **SAMCLIPExtractor** during AOV-GS / ActiveOpenSem if the config enables\n"
            "  language feature saving (e.g. slam.save_clip_features / open-vocab section).\n"
            "  `train_language_autoencoder.py` **reads** this folder and creates language_features_dim*/\n"
            "  — but does **not** generate raw *_f.npy itself.\n\n"
            "  What to do:\n"
            "  • Run feature extraction for the scene (as when training the AE), or\n"
            "  • Point --features_orig at a directory that already has *_f.npy (other disk/backup).\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Evaluating {len(frame_ids)} frames …")

    # ---- metrics accumulators ----
    recon_cos_all: List[float] = []          # per-mask reconstruction cosine
    mse_vec_all: List[float] = []            # per-mask MSE over 512 coords (mean dim)
    rmse_vec_all: List[float] = []           # sqrt(MSE) per mask
    l2_err_all: List[float] = []             # ||orig - dec||_2 per mask
    mae_vec_all: List[float] = []            # mean |orig - dec| per mask
    rank_corr_all: List[float] = []          # per-frame Spearman ρ averaged over queries
    iou_all: List[float] = []               # per-frame per-query IoU
    pearson_o_chunks: List[np.ndarray] = []
    pearson_d_chunks: List[np.ndarray] = []
    _pearson_flat_max = 50_000

    for fid in tqdm(frame_ids):
        mask_feats_512, pix_feats_512, valid = load_frame(
            args.features_orig, fid, args.feature_level
        )
        if mask_feats_512 is None:
            continue

        # Encode & decode masks
        with torch.no_grad():
            t = torch.from_numpy(mask_feats_512).to(device)
            z = ae.encode(t)
            dec = ae.decode(z)              # (N, 512) unit-norm

        ori_t = F.normalize(torch.from_numpy(mask_feats_512).to(device), dim=-1)
        dec_t = F.normalize(dec, dim=-1)

        # 1) Reconstruction cosine (per mask)
        cos = F.cosine_similarity(ori_t, dec_t, dim=-1)  # (N,)
        recon_cos_all.extend(cos.cpu().tolist())

        diff = ori_t - dec_t
        mse_v = (diff ** 2).mean(dim=-1)
        mse_vec_all.extend(mse_v.cpu().tolist())
        rmse_vec_all.extend(torch.sqrt(mse_v + 1e-12).cpu().tolist())
        l2_err_all.extend(diff.norm(dim=-1).cpu().tolist())
        mae_vec_all.extend(diff.abs().mean(dim=-1).cpu().tolist())

        n_acc = sum(len(x) for x in pearson_o_chunks)
        if n_acc < _pearson_flat_max:
            rem = _pearson_flat_max - n_acc
            o_flat = ori_t.cpu().numpy().ravel()
            d_flat = dec_t.cpu().numpy().ravel()
            take = min(rem, o_flat.size)
            pearson_o_chunks.append(o_flat[:take])
            pearson_d_chunks.append(d_flat[:take])

        # 2) Rank correlation per query over masks
        ori_scores = (ori_t @ text_embs.T.to(device)).cpu().numpy()  # (N, Q)
        dec_scores = (dec_t @ text_embs.T.to(device)).cpu().numpy()  # (N, Q)
        if ori_scores.shape[0] < 3:
            continue

        frame_rhos = []
        for qi in range(len(args.queries)):
            rho, _ = spearmanr(ori_scores[:, qi], dec_scores[:, qi])
            if not np.isnan(rho):
                frame_rhos.append(rho)
        if frame_rhos:
            rank_corr_all.append(np.mean(frame_rhos))

        # 3) Pixel IoU: build pixel maps from decoded _f.npy
        if args.features_enc is None:
            continue
        enc_f_path = os.path.join(args.features_enc, f"{fid}_f.npy")
        enc_s_path = os.path.join(args.features_enc, f"{fid}_s.npy")
        if not os.path.exists(enc_f_path):
            continue
        enc_feats_latent = np.load(enc_f_path).astype(np.float32)  # (N, D)
        seg = np.load(enc_s_path)
        H, W = seg.shape[1], seg.shape[2]
        seg_level = seg[args.feature_level]
        valid_pix = seg_level >= 0

        # Decode enc features to 512d
        with torch.no_grad():
            el = torch.from_numpy(enc_feats_latent).to(device)
            dec_enc = ae.decode(el)  # (N, 512) — decode of encoded features
        dec_enc = F.normalize(dec_enc, dim=-1).cpu().numpy()

        # Build pixel maps
        pix_ori = np.zeros((H * W, 512), dtype=np.float32)
        pix_dec = np.zeros((H * W, 512), dtype=np.float32)
        flat_valid = valid_pix.ravel()
        # Some frames/levels can have no valid pixels (all -1), especially for
        # certain SAM levels. In that case IoU for this frame is undefined.
        if not np.any(flat_valid):
            continue
        seg_flat = seg_level.ravel()
        pix_ori[flat_valid] = mask_feats_512[seg_flat[flat_valid]]
        # normalise original
        norms = np.linalg.norm(pix_ori[flat_valid], axis=-1, keepdims=True)
        pix_ori[flat_valid] /= (norms + 1e-8)
        pix_dec[flat_valid] = dec_enc[seg_flat[flat_valid]]

        t_text = text_embs.numpy()  # (Q, 512)
        thr = 100.0 - args.top_pct
        for qi in range(len(args.queries)):
            q = t_text[qi]
            score_ori = pix_ori @ q        # (H*W,)
            score_dec = pix_dec @ q

            # Guard against empty valid set (should be covered above, but keep safe).
            score_ori_v = score_ori[flat_valid]
            score_dec_v = score_dec[flat_valid]
            if score_ori_v.size == 0 or score_dec_v.size == 0:
                continue
            thr_ori = np.percentile(score_ori_v, thr)
            thr_dec = np.percentile(score_dec_v, thr)
            mask_ori = (score_ori >= thr_ori) & flat_valid
            mask_dec = (score_dec >= thr_dec) & flat_valid

            inter = (mask_ori & mask_dec).sum()
            union = (mask_ori | mask_dec).sum()
            iou = inter / (union + 1e-8)
            iou_all.append(float(iou))

    if len(recon_cos_all) == 0:
        print(
            "\n[error] No mask vectors were evaluated (all *_f.npy empty or missing "
            "matching *_s.npy / invalid segments). Check --features_orig and "
            "--feature_level.",
            file=sys.stderr,
        )
        sys.exit(1)

    cos_arr = np.asarray(recon_cos_all, dtype=np.float64)
    p5, p50, p95 = np.percentile(cos_arr, [5, 50, 95])
    pearson_r = float("nan")
    if pearson_o_chunks:
        o = np.concatenate(pearson_o_chunks)
        d = np.concatenate(pearson_d_chunks)
        rng = np.random.default_rng(0)
        if len(o) > 512:
            idx = rng.choice(len(o), size=min(50_000, len(o)), replace=False)
            pr, _ = pearsonr(o[idx], d[idx])
            pearson_r = float(pr)

    # ---- Report ----
    print("\n" + "=" * 60)
    print(f"{'AUTOENCODER EVALUATION':^60}")
    print("=" * 60)
    print("  --- Encoder vs original 512-d (per mask) ---")
    print(f"  Cosine similarity  (mean ± std):     "
          f"{np.mean(recon_cos_all):.4f} ± {np.std(recon_cos_all):.4f}")
    print(f"  Cosine  p5 / median / p95:           {p5:.4f} / {p50:.4f} / {p95:.4f}")
    print(f"  MSE  (mean ± std, over 512 dims):     "
          f"{np.mean(mse_vec_all):.6f} ± {np.std(mse_vec_all):.6f}")
    print(f"  RMSE (mean ± std):                   "
          f"{np.mean(rmse_vec_all):.6f} ± {np.std(rmse_vec_all):.6f}")
    print(f"  L2 error ||o−d||₂  (mean ± std):     "
          f"{np.mean(l2_err_all):.6f} ± {np.std(l2_err_all):.6f}")
    print(f"  MAE  (mean ± std, over coords):      "
          f"{np.mean(mae_vec_all):.6f} ± {np.std(mae_vec_all):.6f}")
    if np.isfinite(pearson_r):
        print(f"  Pearson r (flattened o vs d, subsample): {pearson_r:.4f}")
    print(f"  (min cosine: {np.min(recon_cos_all):.4f})")
    print("  --- Ranking & segmentation (needs queries; IoU needs --features_enc) ---")
    if rank_corr_all:
        print(f"  Spearman ρ  (mean ± std):             "
              f"{np.mean(rank_corr_all):.4f} ± {np.std(rank_corr_all):.4f}")
    else:
        print("  Spearman ρ: (no data)")
    if iou_all:
        print(f"  Pixel IoU  ori↔dec  (mean ± std):     "
              f"{np.mean(iou_all):.4f} ± {np.std(iou_all):.4f}")
    elif args.features_enc is None:
        print("  Pixel IoU: skipped (--features_enc not set)")
    else:
        print("  Pixel IoU: (no overlapping frames / missing enc files)")
    print(f"\n  Queries tested: {args.queries}")
    print(f"  Feature level:  {args.feature_level}  "
          f"(0=default,1=s,2=m,3=l)")
    print(f"  Top-percentile: {args.top_pct}%  for IoU threshold")
    print("=" * 60)

    print("\nHow to read:")
    print("  cos  ≈ 1.0 → AE reconstructs CLIP vectors nearly perfectly.")
    print("  MSE/RMSE/MAE → error in R^512 space (vectors are L2-normalized).")
    print("  ρ   ≈ 1.0 → AE preserves ranking of objects by query similarity.")
    print("  IoU ≈ 1.0 → 'which pixels are most relevant' is the same before/after AE.")
    print("  If IoU is low but cos is high → ranking is broken (scale / direction issue).")

    if args.out_metrics:
        outp = Path(args.out_metrics)
        outp.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"cosine_mean {np.mean(recon_cos_all):.8f}",
            f"cosine_std {np.std(recon_cos_all):.8f}",
            f"cosine_p5 {p5:.8f}",
            f"cosine_median {p50:.8f}",
            f"cosine_p95 {p95:.8f}",
            f"mse_mean {np.mean(mse_vec_all):.8f}",
            f"rmse_mean {np.mean(rmse_vec_all):.8f}",
            f"l2_error_mean {np.mean(l2_err_all):.8f}",
            f"mae_mean {np.mean(mae_vec_all):.8f}",
        ]
        if np.isfinite(pearson_r):
            lines.append(f"pearson_r_flat {pearson_r:.8f}")
        if rank_corr_all:
            lines.append(f"spearman_rho_mean {np.mean(rank_corr_all):.8f}")
        if iou_all:
            lines.append(f"pixel_iou_mean {np.mean(iou_all):.8f}")
        outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nMetrics written to {outp.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate language autoencoder quality.")
    p.add_argument("--features_orig", required=True,
                   help="language_features/ (original 512d)")
    p.add_argument("--features_enc",  default=None,
                   help="language_features_dimD/ for IoU (optional)")
    p.add_argument("--ae_ckpt",       required=True,
                   help="best_ckpt.pth from train_language_autoencoder")
    p.add_argument("--queries", nargs="+",
                   default=["a sofa", "a table", "a plant", "the floor", "a chair"],
                   help="Text queries for evaluation")
    p.add_argument("--feature_level", type=int, default=1,
                   help="SAM level (0=default,1=s,2=m,3=l)")
    p.add_argument("--top_pct",       type=float, default=10.0,
                   help="Top-%%  pixels for IoU threshold (default 10%%)")
    p.add_argument("--encoder_dims",  nargs="+", type=int, default=None,
                   help="Must match train_language_autoencoder (default: from Autoencoder)")
    p.add_argument("--decoder_dims",  nargs="+", type=int, default=None)
    p.add_argument("--clip_model",       default="ViT-B-16")
    p.add_argument("--clip_pretrained",  default="laion2b_s34b_b88k")
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for AE evaluation (default cuda:0).",
    )
    p.add_argument("--out_metrics", default=None,
                   help="Save numeric metrics to a text file (key value)")
    return p.parse_args()


if __name__ == "__main__":
    eval_ae(parse_args())
