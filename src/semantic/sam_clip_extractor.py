"""
SAM + CLIP feature extractor — формат совместим с LangSplat.

Для каждого RGB-кадра:
  1. Запускает SAM для 4 уровней детализации (default, s, m, l)
  2. Для каждой маски делает кроп → resize 224×224 → CLIP embed (512d)
  3. Сохраняет два файла, идентичных формату LangSplat preprocess.py:

     {frame_id:06d}_s.npy  — int32 (4, H, W)   карта сегментации
                              значение = индекс маски в _f.npy
                              (с кумулятивными смещениями по уровням)
                              -1 = фон

     {frame_id:06d}_f.npy  — float16 (N_total, 512)
                              CLIP-эмбеддинги всех масок всех уровней подряд

Эти файлы используются напрямую как входные данные для:
  - train_language_autoencoder.py  (читает _f.npy)
  - train_language_field.py / langsplatam.py  (читает _s.npy + _f.npy → пиксельную карту)

Запуск в фоновом треде, кадры добавляются через submit().
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Порядок уровней — как в LangSplat
_LEVELS = ['default', 's', 'm', 'l']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8_numpy(color: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """[H,W,3] uint8 или float tensor → uint8 numpy BGR (для SAM/cv2)."""
    if isinstance(color, torch.Tensor):
        arr = color.detach().cpu()
        if arr.dtype != torch.uint8:
            arr = (arr * 255).clamp(0, 255).byte()
        arr = arr.numpy()
    else:
        arr = color
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr  # RGB uint8


def _habitat_rgb_for_sam(image_rgb: np.ndarray) -> np.ndarray:
    """RGB уже в порядке строк OpenCV (симулятор: ``pinhole_vertical_flip`` в ``habitat.py``)."""
    return np.ascontiguousarray(image_rgb)


def _sam_masks_to_habitat(masks: list) -> list:
    """Маски SAM в той же индексации строк, что и тензоры SLAM / датасет."""
    out: list = []
    for m in masks:
        mc = dict(m)
        seg = np.asarray(m["segmentation"], dtype=bool)
        mc["segmentation"] = np.ascontiguousarray(seg)
        mc["bbox"] = np.asarray(m["bbox"], dtype=np.float64)
        out.append(mc)
    return out


def get_seg_img(mask: dict, image: np.ndarray) -> np.ndarray:
    """
    Кроп изображения по bbox маски.

    Важно: оставляем фон внутри bbox (не зануляем вне-маску), т.к. так CLIP
    видит контекст объекта. Паддинг bbox применяется отдельно.
    """
    x, y, w, h = np.int32(mask["bbox"])
    return image[y:y + h, x:x + w, ...]


def _tight_crop_with_padding(mask: dict, image: np.ndarray, pad_px: int) -> np.ndarray:
    """
    BBox crop with padding on each side, BUT only if the expanded bbox stays
    strictly within image bounds. If expansion would go out of bounds, fall
    back to the original bbox crop (no padding).
    """
    H, W = image.shape[:2]
    x, y, w, h = np.int32(mask["bbox"])
    x0 = int(x) - int(pad_px)
    y0 = int(y) - int(pad_px)
    x1 = int(x + w) + int(pad_px)
    y1 = int(y + h) + int(pad_px)

    # If any side would go out of bounds, do NOT pad at all.
    if x0 < 0 or y0 < 0 or x1 > int(W) or y1 > int(H) or x1 <= x0 or y1 <= y0:
        return get_seg_img(mask, image)

    return image[y0:y1, x0:x1, ...]


def pad_img(img: np.ndarray) -> np.ndarray:
    """Дополнить изображение до квадрата (legacy helper)."""
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def _resize_to_fit_and_letterbox_replicate(img: np.ndarray, out_hw: int = 224) -> np.ndarray:
    """
    Aspect-ratio preserving resize to fit within (out_hw, out_hw) + letterbox padding
    using BORDER_REPLICATE (no cropping, but padded pixels are replicated).
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_hw, out_hw, 3), dtype=np.uint8)
    if h == out_hw and w == out_hw:
        return img

    # Scale so that the long side becomes out_hw (fit inside square).
    long = max(h, w)
    scale = float(out_hw) / float(long)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = max(1, min(out_hw, new_w))
    new_h = max(1, min(out_hw, new_h))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox padding to exact square using border replication.
    dh = out_hw - new_h
    dw = out_hw - new_w
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)


def _mask_centroid_xy(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def _seg_to_bbox_area(seg: np.ndarray) -> tuple[np.ndarray, int]:
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), 0
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w = max(1.0, x1 - x0 + 1.0)
    h = max(1.0, y1 - y0 + 1.0)
    area = int(seg.sum())
    return np.array([x0, y0, w, h], dtype=np.float64), area


def _merge_masks_corrclip_style(
    masks: list,
    embs: np.ndarray,
    sim_thresh: float,
    max_dist_px: float,
) -> tuple[list, np.ndarray]:
    """
    CorrCLIP-inspired mask merging:
    merge small masks into nearby semantically similar larger masks.
    """
    n = len(masks)
    if n <= 1 or embs.shape[0] != n:
        return masks, embs

    feats = embs.astype(np.float32, copy=False)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    areas = np.array([int(m["area"]) for m in masks], dtype=np.int64)
    cents = np.array([_mask_centroid_xy(m["segmentation"]) for m in masks], dtype=np.float32)
    alive = np.ones((n,), dtype=bool)

    # Merge from small to large to reduce over-segmentation fragments.
    for i in np.argsort(areas):
        if not alive[i]:
            continue
        ci = cents[i]
        best_j = -1
        best_sim = -1.0
        for j in range(n):
            if i == j or not alive[j]:
                continue
            if areas[j] < areas[i]:
                continue
            d = float(np.linalg.norm(ci - cents[j]))
            if d > max_dist_px:
                continue
            s = float(np.dot(feats[i], feats[j]))
            if s < sim_thresh:
                continue
            if s > best_sim:
                best_sim = s
                best_j = j
        if best_j < 0:
            continue

        # Merge i -> best_j
        seg_j = np.logical_or(masks[best_j]["segmentation"], masks[i]["segmentation"])
        bbox_j, area_j = _seg_to_bbox_area(seg_j)
        wi = float(max(areas[i], 1))
        wj = float(max(areas[best_j], 1))
        merged_feat = (wj * feats[best_j] + wi * feats[i]) / (wj + wi + 1e-8)
        merged_feat /= (np.linalg.norm(merged_feat) + 1e-8)

        masks[best_j]["segmentation"] = np.ascontiguousarray(seg_j)
        masks[best_j]["bbox"] = bbox_j
        masks[best_j]["area"] = int(area_j)
        feats[best_j] = merged_feat
        cents[best_j] = np.array(_mask_centroid_xy(seg_j), dtype=np.float32)
        areas[best_j] = int(area_j)
        alive[i] = False

    keep = np.where(alive)[0]
    # Keep deterministic ordering by area desc (same spirit as SAM sorting).
    keep = keep[np.argsort(-areas[keep])]
    merged_masks = [masks[k] for k in keep]
    merged_feats = feats[keep].astype(np.float16)
    return merged_masks, merged_feats


def _suppress_interclass_corrclip_style(
    embs: np.ndarray,
    masks: list,
    alpha: float,
    sim_thresh: float,
    sigma_px: float,
) -> np.ndarray:
    """
    CorrCLIP-inspired inter-class suppression on mask embeddings.
    """
    n = embs.shape[0]
    if n <= 1 or alpha <= 0.0:
        return embs
    feats = embs.astype(np.float32, copy=False)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    cents = np.array([_mask_centroid_xy(m["segmentation"]) for m in masks], dtype=np.float32)
    sims = np.clip(feats @ feats.T, -1.0, 1.0)
    np.fill_diagonal(sims, 0.0)

    dxy = cents[:, None, :] - cents[None, :, :]
    dist2 = (dxy ** 2).sum(axis=-1)
    spatial = np.exp(-dist2 / (2.0 * (max(float(sigma_px), 1.0) ** 2)))
    inter_w = np.maximum(sims - float(sim_thresh), 0.0) * spatial
    suppress = inter_w @ feats
    refined = feats - float(alpha) * suppress
    refined /= (np.linalg.norm(refined, axis=1, keepdims=True) + 1e-8)
    return refined.astype(np.float16)


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class SAMCLIPExtractor:
    """
    Фоновый поток для извлечения SAM+CLIP фич в формате LangSplat.

    Parameters
    ----------
    save_dir : str | Path
        Куда сохранять {frame_id:06d}_s.npy и {frame_id:06d}_f.npy.
    sam_ckpt_path : str
        Путь к чекпоинту SAM ViT-H.
    clip_model : str
        Имя модели open_clip, напр. "ViT-B-16".
    clip_pretrained : str
        Веса open_clip, напр. "laion2b_s34b_b88k".
    device : str
        CUDA-устройство для SAM и CLIP.
    queue_size : int
        Максимум кадров в очереди.
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        sam_ckpt_path: str,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda:0",
        queue_size: int = 64,
        submit_timeout_s: float = 1.0,
        bbox_pad_px: int = 20,
        debug_dir: Optional[Union[str, Path]] = None,
        clip_batch_size: int = 32,
        max_masks_per_frame: int = 150,
        corrclip_mask_merge: bool = True,
        corrclip_merge_sim_thresh: float = 0.86,
        corrclip_merge_dist_px: float = 80.0,
        corrclip_interclass_suppress_alpha: float = 0.15,
        corrclip_interclass_sim_thresh: float = 0.78,
        corrclip_interclass_sigma_px: float = 120.0,
        # Параметры ниже оставлены для обратной совместимости, не используются
        levels: tuple = ("default", "s", "m", "l"),
        save_fp16: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir: Optional[Path] = Path(debug_dir) if debug_dir is not None else None
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.sam_ckpt_path = sam_ckpt_path
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained
        self.device = self._resolve_device(device)
        self.bbox_pad_px = int(bbox_pad_px)
        self.clip_batch_size = int(clip_batch_size)
        self.max_masks_per_frame = int(max_masks_per_frame)
        self.submit_timeout_s = float(submit_timeout_s)
        self.corrclip_mask_merge = bool(corrclip_mask_merge)
        self.corrclip_merge_sim_thresh = float(corrclip_merge_sim_thresh)
        self.corrclip_merge_dist_px = float(corrclip_merge_dist_px)
        self.corrclip_interclass_suppress_alpha = float(corrclip_interclass_suppress_alpha)
        self.corrclip_interclass_sim_thresh = float(corrclip_interclass_sim_thresh)
        self.corrclip_interclass_sigma_px = float(corrclip_interclass_sigma_px)
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._mask_generator = None
        self._clip_model = None
        self._clip_process = None
        self._running = False
        self._accepting_submissions = False
        self._submitted_frames = 0
        self._dropped_frames = 0
        self._processed_frames = 0
        self._failed_frames = 0
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._accepting_submissions = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="SAMCLIPWorker"
        )
        self._thread.start()
        logger.info("SAMCLIPExtractor started on %s", self.device)

    def submit(self, frame_id: int, color: Union[np.ndarray, torch.Tensor]) -> None:
        """Добавить кадр в очередь обработки.

        GPU-тензор немедленно конвертируется в CPU numpy, чтобы очередь не
        удерживала ссылки на GPU-память.

        Используется bounded blocking put (submit_timeout_s), чтобы не терять
        кадры бесшумно при кратковременных всплесках нагрузки.
        """
        if not self._running or not self._accepting_submissions:
            return
        if isinstance(color, torch.Tensor):
            color = color.detach().cpu().numpy()
        try:
            self._queue.put((frame_id, color), block=True, timeout=self.submit_timeout_s)
            with self._stats_lock:
                self._submitted_frames += 1
        except queue.Full:
            with self._stats_lock:
                self._dropped_frames += 1
            logger.warning(
                "SAMCLIPExtractor: queue full for %.2fs, dropping frame %d. "
                "Increase sam_clip.queue_size or reduce SAM/CLIP load.",
                self.submit_timeout_s, frame_id
            )

    def flush(self) -> None:
        """Дождаться обработки всех кадров в очереди."""
        self._queue.join()

    def stop(self, wait: bool = True, drain: bool = True, join_timeout_s: float = 120.0) -> None:
        """Завершить фоновый поток.

        Args:
            wait: ждать ли завершения воркер-потока.
            drain: обработать ли все уже поставленные кадры перед остановкой.
                   False = быстрый stop: кадры в очереди будут отброшены.
            join_timeout_s: таймаут ожидания завершения потока.
        """
        self._accepting_submissions = False

        if not drain:
            while True:
                try:
                    item = self._queue.get_nowait()
                    self._queue.task_done()
                    if item is None:
                        break
                except queue.Empty:
                    break

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put(None, block=True)

        if wait and self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
        self._running = False
        logger.info("SAMCLIPExtractor stopped. stats=%s", self.stats())

    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "submitted_frames": int(self._submitted_frames),
                "processed_frames": int(self._processed_frames),
                "failed_frames": int(self._failed_frames),
                "dropped_frames": int(self._dropped_frames),
                "queue_size_current": int(self._queue.qsize()),
                "accepting_submissions": bool(self._accepting_submissions),
                "running": bool(self._running),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_device(self, device: str) -> str:
        if not device.startswith("cuda"):
            return device
        try:
            idx = int(device.split(":")[-1]) if ":" in device else 0
            if idx < torch.cuda.device_count():
                return device
        except (ValueError, RuntimeError):
            pass
        fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.warning("Device %r unavailable, falling back to %r.", device, fallback)
        return fallback

    def _load_models(self) -> None:
        """Загрузить SAM и CLIP (один раз, в воркер-треде)."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as exc:
            raise ImportError(
                "segment_anything is required. "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        # Определяем тип модели по имени файла чекпоинта
        ckpt_name = Path(self.sam_ckpt_path).name
        if "vit_h" in ckpt_name:
            model_type = "vit_h"
        elif "vit_l" in ckpt_name:
            model_type = "vit_l"
        else:
            model_type = "vit_b"
        logger.info("SAM model type: %s (from %s)", model_type, ckpt_name)
        sam = sam_model_registry[model_type](checkpoint=self.sam_ckpt_path)
        sam.to(device=self.device)
        # Единственный генератор — как в LangSplat preprocess.py
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
        logger.info("SAM (ViT-H) loaded.")

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("open_clip_torch is required. pip install open_clip_torch") from exc

        import torchvision
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.clip_pretrained,
            precision="fp16",
        )
        model.eval()
        model = model.to(self.device)
        self._clip_model = model
        # Препроцессинг как в LangSplat (resize 224 + normalize)
        self._clip_process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        logger.info("CLIP (%s / %s) loaded.", self.clip_model_name, self.clip_pretrained)

    def _masks_at_level(self, masks_all: list, level: str) -> list:
        """
        LangSplat разбивает маски на 4 уровня по размеру area.
        Воспроизводим аналогичное разбиение:
          default — все маски
          s — маски area < 32^2
          m — маски area в [32^2, 96^2)
          l — маски area >= 96^2
        """
        if level == 'default':
            return masks_all
        thresholds = {'s': (0, 32 * 32), 'm': (32 * 32, 96 * 96), 'l': (96 * 96, float('inf'))}
        lo, hi = thresholds[level]
        return [m for m in masks_all if lo <= m['area'] < hi]

    @torch.no_grad()
    def _embed_masks(self, image_rgb: np.ndarray, masks: list) -> np.ndarray:
        """
        Для каждой маски: кроп → pad → resize 224×224 → CLIP encode.
        Возвращает (N, 512) float16 numpy, L2-нормализован.

        Обработка ведётся батчами по clip_batch_size масок, чтобы не создавать
        один огромный тензор [N, 3, 224, 224] в GPU-памяти при большом N.
        """
        if not masks:
            return np.zeros((0, 512), dtype=np.float16)

        crops = []
        for m in masks:
            # BBox crop that contains the mask. Add +/-pad_px only if the expanded
            # bbox stays within image bounds; otherwise keep the original bbox.
            crop = _tight_crop_with_padding(m, image_rgb, pad_px=self.bbox_pad_px)
            # Keep aspect ratio with no cropping: resize-to-fit + replicate-letterbox to 224×224.
            crops.append(_resize_to_fit_and_letterbox_replicate(crop, out_hw=224))

        all_embs: list = []
        for start in range(0, len(crops), self.clip_batch_size):
            batch = crops[start : start + self.clip_batch_size]
            tensor = torch.from_numpy(
                np.stack(batch, axis=0).astype(np.float32)
            ).permute(0, 3, 1, 2) / 255.0
            tensor = self._clip_process(tensor).half().to(self.device)
            emb = self._clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())
            del tensor, emb

        result = torch.cat(all_embs, dim=0).float().numpy().astype(np.float16)
        return result

    def _process_frame(self, frame_id: int, color: Union[np.ndarray, torch.Tensor]) -> None:
        """Основная функция: SAM → CLIP → сохранение в формате LangSplat."""
        s_path = self.save_dir / f"{frame_id:06d}_s.npy"
        f_path = self.save_dir / f"{frame_id:06d}_f.npy"
        if s_path.exists() and f_path.exists():
            logger.debug("Frame %d already processed, skipping.", frame_id)
            return

        image_rgb = _to_uint8_numpy(color)  # (H, W, 3) uint8 RGB, row 0 = top
        H, W = image_rgb.shape[:2]

        image_rgb_sam = _habitat_rgb_for_sam(image_rgb)
        image_bgr_sam = cv2.cvtColor(image_rgb_sam, cv2.COLOR_RGB2BGR)
        masks_all_sam = self._mask_generator.generate(image_bgr_sam)

        # Освобождаем GPU-кэш после SAM, чтобы SLAM в главном треде получил
        # как можно больше свободной памяти до следующего шага.
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Ограничиваем общее число масок: оставляем наибольшие по площади,
        # чтобы контролировать пиковый размер CLIP-батча.
        if len(masks_all_sam) > self.max_masks_per_frame:
            masks_all_sam = sorted(masks_all_sam, key=lambda m: m['area'], reverse=True)[
                :self.max_masks_per_frame
            ]

        masks_all = _sam_masks_to_habitat(masks_all_sam)

        # Debug: overlay в той же ориентации, что и SAM (читаемый segmentframes/)
        if self.debug_dir is not None and masks_all_sam:
            rng = np.random.default_rng(seed=42)
            overlay = image_rgb_sam.copy()
            for m in sorted(masks_all_sam, key=lambda x: x['area'], reverse=True):
                color_mask = rng.integers(0, 256, 3, dtype=np.uint8)
                overlay[m['segmentation']] = (
                    overlay[m['segmentation']] * 0.4 + color_mask * 0.6
                ).astype(np.uint8)
            out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.debug_dir / f"frame_{frame_id:06d}.jpg"), out_bgr)
            logger.debug("Debug mask saved for frame %d (%d masks)", frame_id, len(masks_all_sam))

        seg_maps = []   # будет (4, H, W) int32
        all_embeds = [] # будет (N_total, 512) float16
        cumsum = 0

        for level in _LEVELS:
            masks_lvl = self._masks_at_level(masks_all, level)
            embs = self._embed_masks(image_rgb, masks_lvl)
            if len(masks_lvl) > 0 and embs.shape[0] > 0:
                if self.corrclip_mask_merge:
                    masks_lvl, embs = _merge_masks_corrclip_style(
                        masks_lvl,
                        embs,
                        sim_thresh=self.corrclip_merge_sim_thresh,
                        max_dist_px=self.corrclip_merge_dist_px,
                    )
                if self.corrclip_interclass_suppress_alpha > 0.0:
                    embs = _suppress_interclass_corrclip_style(
                        embs,
                        masks_lvl,
                        alpha=self.corrclip_interclass_suppress_alpha,
                        sim_thresh=self.corrclip_interclass_sim_thresh,
                        sigma_px=self.corrclip_interclass_sigma_px,
                    )

            # Build segmentation map AFTER CorrCLIP-style refinement/merge.
            seg_map = -np.ones((H, W), dtype=np.int32)
            for i, m in enumerate(masks_lvl):
                seg_map[m["segmentation"]] = i + cumsum
            seg_maps.append(seg_map)

            if embs.shape[0] > 0:
                all_embeds.append(embs)
            cumsum += len(masks_lvl)

        # Освобождаем кэш после всех CLIP-проходов для этого кадра
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        seg_tensor = np.stack(seg_maps, axis=0)  # (4, H, W) int32
        feat_tensor = np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0, 512), dtype=np.float16)

        np.save(str(s_path), seg_tensor)
        np.save(str(f_path), feat_tensor)

        logger.debug(
            "Frame %d saved: seg %s, feats %s → %s",
            frame_id, seg_tensor.shape, feat_tensor.shape, self.save_dir,
        )

    def _worker_loop(self) -> None:
        # Models are loaded lazily on the first frame to avoid occupying GPU
        # memory during the early SLAM initialization phase.
        models_loaded = False
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    break
                if not models_loaded:
                    self._load_models()
                    models_loaded = True
                frame_id, color = item
                try:
                    self._process_frame(frame_id, color)
                    with self._stats_lock:
                        self._processed_frames += 1
                except Exception as exc:
                    with self._stats_lock:
                        self._failed_frames += 1
                    logger.error("SAMCLIPExtractor: frame %d failed: %s", frame_id, exc, exc_info=True)
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Утилита загрузки — аналог get_language_feature() из LangSplat/cameras.py
# ---------------------------------------------------------------------------

def load_frame_features(
    save_dir: Union[str, Path],
    frame_id: int,
    feature_level: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Загрузить (_s.npy, _f.npy) и вернуть пиксельную карту фич для одного уровня.

    Аналог Camera.get_language_feature() из LangSplat/scene/cameras.py.

    Parameters
    ----------
    save_dir : путь к language_features/ или language_features_dim3/
    frame_id : числовой ID кадра
    feature_level : 0=default, 1=s, 2=m, 3=l

    Returns
    -------
    point_feature : float32 (D, H, W)  — пиксельная карта фич
    mask          : bool   (1, H, W)   — маска валидных пикселей
    """
    save_dir = Path(save_dir)
    seg_map = np.load(str(save_dir / f"{frame_id:06d}_s.npy"))    # (4, H, W) int32
    feature_map = np.load(str(save_dir / f"{frame_id:06d}_f.npy"))  # (N, D)
    feature_map = feature_map.astype(np.float32)

    H, W = seg_map.shape[1], seg_map.shape[2]
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # (H, W)

    seg = seg_map[feature_level, y, x]  # (H, W) int32
    valid = seg >= 0
    point_feature = np.zeros((H, W, feature_map.shape[1]), dtype=np.float32)
    point_feature[valid] = feature_map[seg[valid]]

    # (D, H, W) + (1, H, W)
    point_feature = point_feature.transpose(2, 0, 1)
    mask = valid[np.newaxis, :, :]
    return point_feature, mask
