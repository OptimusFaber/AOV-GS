"""
Гибридная тактика исследования: геометрия (непокрытые пиксели) + семантика SAM+CLIP.

В отличие от ActiveSGM (OneFormer + энтропия logits), здесь семантический сигнал —
насколько эмбеддинги масок кандидатного вида отличаются от уже собранных keyframe
эмбеддингов в ``language_features/``. Чем больше отличие, тем выше приоритет.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def keyframe_color_to_rgb_uint8(color: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Keyframe color (CHW float or HWC) → uint8 RGB."""
    if isinstance(color, torch.Tensor):
        arr = color.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.permute(1, 2, 0)
        if arr.dtype != torch.uint8:
            return (arr.numpy() * 255).clip(0, 255).astype(np.uint8)
        return arr.numpy()
    arr = np.asarray(color)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        return (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def _l2_normalize_rows(feats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if feats.size == 0:
        return feats.astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return (feats / np.maximum(norms, eps)).astype(np.float32)


def load_keyframe_mask_embeddings(
    lang_feat_dir: Union[str, Path],
    frame_id: int,
) -> np.ndarray:
    """Загрузить L2-нормированные CLIP-эмбеддинги масок keyframe (N, D)."""
    f_path = Path(lang_feat_dir) / f"{frame_id:06d}_f.npy"
    if not f_path.exists():
        return np.zeros((0, 512), dtype=np.float32)
    feats = np.load(str(f_path)).astype(np.float32)
    return _l2_normalize_rows(feats)


def mask_embedding_novelty(
    candidate_feats: np.ndarray,
    bank_feats: np.ndarray,
    aggregation: str = "mean",
) -> float:
    """
    Новизна кандидата относительно банка масок.

    Для каждой маски кандидата: dissim = 1 - max_k cos(e_cand, e_bank).
    Итог: mean или max по маскам кандидата.
    """
    if candidate_feats.shape[0] == 0:
        return 0.0
    if bank_feats.shape[0] == 0:
        return 1.0

    cand = _l2_normalize_rows(candidate_feats)
    bank = _l2_normalize_rows(bank_feats)
    sims = cand @ bank.T
    per_mask = 1.0 - sims.max(axis=1)
    if aggregation == "max":
        return float(per_mask.max())
    return float(per_mask.mean())


class KeyframeMaskEmbeddingBank:
    """Банк SAM+CLIP эмбеддингов масок с диска (language_features/)."""

    def __init__(self, lang_feat_dir: Union[str, Path]) -> None:
        self.lang_feat_dir = Path(lang_feat_dir)
        self._bank: Optional[np.ndarray] = None
        self._loaded_ids: set = set()

    def __len__(self) -> int:
        return 0 if self._bank is None else int(self._bank.shape[0])

    def refresh(
        self,
        keyframe_ids: List[int],
        keyframe_list: Optional[List[dict]] = None,
        encoder: Optional["PlanningSAMCLIPEncoder"] = None,
        encode_if_missing: bool = False,
    ) -> int:
        """Догрузить keyframe-эмбеддинги с диска или из RGB keyframe (fallback)."""
        kf_by_id = {int(kf["id"]): kf for kf in (keyframe_list or [])}
        new_blocks: List[np.ndarray] = []
        if self._bank is not None and self._bank.shape[0] > 0:
            new_blocks.append(self._bank)

        for kf_id in keyframe_ids:
            if kf_id in self._loaded_ids:
                continue
            feats = load_keyframe_mask_embeddings(self.lang_feat_dir, kf_id)
            source = "disk"
            if feats.shape[0] == 0 and encode_if_missing and encoder is not None:
                kf = kf_by_id.get(kf_id)
                if kf is not None and "color" in kf:
                    feats = encoder.encode_rgb(keyframe_color_to_rgb_uint8(kf["color"]))
                    source = "sam_clip_sync"
            self._loaded_ids.add(kf_id)
            if feats.shape[0] > 0:
                new_blocks.append(feats)
                logger.debug("Bank +%d masks from kf %d (%s)", feats.shape[0], kf_id, source)

        if new_blocks:
            self._bank = np.concatenate(new_blocks, axis=0)
        else:
            self._bank = np.zeros((0, 512), dtype=np.float32)
        return len(self)

    @property
    def feats(self) -> np.ndarray:
        if self._bank is None:
            return np.zeros((0, 512), dtype=np.float32)
        return self._bank


def _embed_masks_standalone(
    image_rgb: np.ndarray,
    masks: list,
    clip_model,
    clip_process,
    device: str,
    clip_batch_size: int,
    bbox_pad_px: int,
) -> np.ndarray:
    """Локальная обёртка вокруг логики SAMCLIPExtractor._embed_masks."""
    from src.semantic.sam_clip_extractor import (
        _resize_to_fit_and_letterbox_replicate,
        _tight_crop_with_padding,
    )

    if not masks:
        return np.zeros((0, 512), dtype=np.float32)

    crops = []
    for m in masks:
        crop = _tight_crop_with_padding(m, image_rgb, pad_px=bbox_pad_px)
        crops.append(_resize_to_fit_and_letterbox_replicate(crop, out_hw=224))

    all_embs: list = []
    for start in range(0, len(crops), clip_batch_size):
        batch = crops[start : start + clip_batch_size]
        tensor = torch.from_numpy(
            np.stack(batch, axis=0).astype(np.float32)
        ).permute(0, 3, 1, 2) / 255.0
        tensor = clip_process(tensor).half().to(device)
        emb = clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 512), dtype=np.float32)


class PlanningSAMCLIPEncoder:
    """
    Синхронный SAM+CLIP для оценки кандидатных видов в планировщике.

    Модели грузятся лениво; лимиты масок ниже, чем у фонового SAMCLIPExtractor.
    """

    def __init__(
        self,
        sam_ckpt_path: str,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda:0",
        max_masks_per_frame: int = 40,
        clip_batch_size: int = 16,
        bbox_pad_px: int = 0,
    ) -> None:
        self.sam_ckpt_path = sam_ckpt_path
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained
        self.device = self._resolve_device(device)
        self.max_masks_per_frame = max_masks_per_frame
        self.clip_batch_size = clip_batch_size
        self.bbox_pad_px = bbox_pad_px
        self._mask_generator = None
        self._clip_model = None
        self._clip_process = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if not device.startswith("cuda"):
            return device
        try:
            idx = int(device.split(":")[-1]) if ":" in device else 0
            if idx < torch.cuda.device_count():
                return device
        except (ValueError, RuntimeError):
            pass
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        if self._mask_generator is not None:
            return

        from pathlib import Path as _Path

        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as exc:
            raise ImportError(
                "segment_anything required for hybrid exploration. "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        ckpt_name = _Path(self.sam_ckpt_path).name
        if "vit_h" in ckpt_name:
            model_type = "vit_h"
        elif "vit_l" in ckpt_name:
            model_type = "vit_l"
        else:
            model_type = "vit_b"

        sam = sam_model_registry[model_type](checkpoint=self.sam_ckpt_path)
        sam.to(device=self.device)
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )

        import open_clip
        import torchvision

        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.clip_pretrained,
            precision="fp16",
        )
        model.eval()
        self._clip_model = model.to(self.device)
        self._clip_process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        logger.info(
            "PlanningSAMCLIPEncoder ready (SAM %s, CLIP %s on %s).",
            model_type,
            self.clip_model_name,
            self.device,
        )

    @torch.no_grad()
    def encode_rgb(self, image_rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 (H,W,3) → (N, 512) float32 L2-нормированные эмбеддинги масок."""
        from src.semantic.sam_clip_extractor import (
            _habitat_rgb_for_sam,
            _sam_masks_to_habitat,
            _to_uint8_numpy,
        )

        self._ensure_loaded()
        image_rgb = _to_uint8_numpy(image_rgb)
        image_rgb_sam = _habitat_rgb_for_sam(image_rgb)

        import cv2

        image_bgr_sam = cv2.cvtColor(image_rgb_sam, cv2.COLOR_RGB2BGR)
        masks_all_sam = self._mask_generator.generate(image_bgr_sam)
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        if len(masks_all_sam) > self.max_masks_per_frame:
            masks_all_sam = sorted(
                masks_all_sam, key=lambda m: m["area"], reverse=True
            )[: self.max_masks_per_frame]

        masks_all = _sam_masks_to_habitat(masks_all_sam)
        embs = _embed_masks_standalone(
            image_rgb=image_rgb,
            masks=masks_all,
            clip_model=self._clip_model,
            clip_process=self._clip_process,
            device=self.device,
            clip_batch_size=self.clip_batch_size,
            bbox_pad_px=self.bbox_pad_px,
        )
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return embs.astype(np.float32)


class HybridSemanticExplorationScorer:
    """
    Скорер семантической новизны для этапа исследования (stage 0).

    Использует банк keyframe-эмбеддингов с диска и SAM+CLIP на симулированном RGB.
    """

    def __init__(
        self,
        lang_feat_dir: Union[str, Path],
        sam_ckpt_path: str,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda:0",
        max_masks_per_candidate: int = 40,
        novelty_aggregation: str = "mean",
        min_bank_masks: int = 8,
    ) -> None:
        self.bank = KeyframeMaskEmbeddingBank(lang_feat_dir)
        self.encoder = PlanningSAMCLIPEncoder(
            sam_ckpt_path=sam_ckpt_path,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            device=device,
            max_masks_per_frame=max_masks_per_candidate,
        )
        self.novelty_aggregation = novelty_aggregation
        self.min_bank_masks = min_bank_masks
        self.encode_keyframes_if_missing = True
        self.log_scores = False
        self.last_stats: Dict[str, Any] = {}

    def refresh_bank(
        self,
        keyframe_ids: List[int],
        keyframe_list: Optional[List[dict]] = None,
    ) -> int:
        return self.bank.refresh(
            keyframe_ids,
            keyframe_list=keyframe_list,
            encoder=self.encoder,
            encode_if_missing=self.encode_keyframes_if_missing,
        )

    def score_candidate_rgb(self, image_rgb: np.ndarray) -> tuple:
        """
        Returns (novelty, did_run_sam, n_cand_masks).
        If bank too small: (0.0, False, 0) — caller should treat as inactive semantic.
        """
        if len(self.bank) < self.min_bank_masks:
            return 0.0, False, 0, 0.0
        t0 = time.perf_counter()
        cand_feats = self.encoder.encode_rgb(image_rgb)
        sam_sec = time.perf_counter() - t0
        novelty = mask_embedding_novelty(
            cand_feats,
            self.bank.feats,
            aggregation=self.novelty_aggregation,
        )
        return novelty, True, int(cand_feats.shape[0]), sam_sec

    def score_batch(
        self,
        images_rgb: List[np.ndarray],
        keyframe_ids: List[int],
        keyframe_list: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Вернуть тензор новизны [N] для softmax-взвешивания."""
        t_bank = time.perf_counter()
        bank_masks = self.refresh_bank(keyframe_ids, keyframe_list)
        bank_sec = time.perf_counter() - t_bank

        scores: List[float] = []
        n_sam_calls = 0
        total_sam_sec = 0.0
        total_cand_masks = 0

        if bank_masks < self.min_bank_masks:
            self.last_stats = {
                "bank_masks": bank_masks,
                "n_candidates": len(images_rgb),
                "semantic_active": False,
                "reason": "bank_too_small",
                "bank_sec": bank_sec,
            }
            return torch.zeros(len(images_rgb), dtype=torch.float32)

        for img in images_rgb:
            novelty, ran_sam, n_masks, sam_sec = self.score_candidate_rgb(img)
            scores.append(novelty)
            if ran_sam:
                n_sam_calls += 1
                total_sam_sec += sam_sec
                total_cand_masks += n_masks

        scores_t = torch.tensor(scores, dtype=torch.float32)
        self._fill_last_stats(
            bank_masks=bank_masks,
            n_candidates=len(images_rgb),
            n_sam_calls=n_sam_calls,
            total_cand_masks=total_cand_masks,
            bank_sec=bank_sec,
            sam_sec=total_sam_sec,
            scores=scores,
        )
        return scores_t

    def score_batch_from_topk(
        self,
        raw_novelties: List[float],
        top_idx: List[int],
        n_candidates: int,
        keyframe_ids: List[int],
        keyframe_list: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Обновить stats после того, как novelty для top-K уже посчитаны снаружи."""
        t_bank = time.perf_counter()
        bank_masks = self.refresh_bank(keyframe_ids, keyframe_list)
        bank_sec = time.perf_counter() - t_bank

        if bank_masks < self.min_bank_masks:
            self.last_stats = {
                "bank_masks": bank_masks,
                "n_candidates": n_candidates,
                "semantic_active": False,
                "reason": "bank_too_small",
                "bank_sec": bank_sec,
            }
            return torch.zeros(n_candidates, dtype=torch.float32)

        self._fill_last_stats(
            bank_masks=bank_masks,
            n_candidates=n_candidates,
            n_sam_calls=len(top_idx),
            total_cand_masks=0,
            bank_sec=bank_sec,
            sam_sec=0.0,
            scores=raw_novelties,
        )
        return torch.tensor(raw_novelties, dtype=torch.float32)

    def _fill_last_stats(
        self,
        bank_masks: int,
        n_candidates: int,
        n_sam_calls: int,
        total_cand_masks: int,
        bank_sec: float,
        sam_sec: float,
        scores: List[float],
    ) -> None:
        scores_t = torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros(0)
        self.last_stats = {
            "bank_masks": bank_masks,
            "n_candidates": n_candidates,
            "n_sam_calls": n_sam_calls,
            "total_cand_masks": total_cand_masks,
            "semantic_active": True,
            "bank_sec": bank_sec,
            "sam_sec": sam_sec,
            "novelty_min": float(scores_t.min().item()) if len(scores) else 0.0,
            "novelty_max": float(scores_t.max().item()) if len(scores) else 0.0,
            "novelty_mean": float(scores_t.mean().item()) if len(scores) else 0.0,
        }
