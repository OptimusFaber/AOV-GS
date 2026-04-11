#!/usr/bin/env python3
"""
Оценка: SAM (автомаски) + CLIP → для каждого **текстового класса** выбирается маска с
максимальным cosine к эмбеддингу запроса → **mIoU** с GT semantic Replica.

Режимы:

1) **Один класс на весь бенч** — ``--class_name chair`` (как раньше).

2) **Список классов на каждый кадр из JSON** — ``--queries_json path.json``:
   для каждого изображения задаются имена классов Replica; SAM один раз на кадр,
   эмбеддинги масок один раз, тексты батчом → для каждого класса свой argmax по маскам.

Формат ``queries.json``::

    {
      "version": 1,
      "by_sample_id": {
        "0": ["chair", "table", "monitor"],
        "1": ["sofa"]
      }
    }

Необязательно: значение-объект вместо списка::

    "0": {
      "classes": ["chair", "table"],
      "queries": { "chair": "a wooden chair", "table": "office desk" }
    }

Если для ``sample_id`` нет записи в JSON — кадр пропускается (или задайте
``--queries_from_gt`` — классы берутся из уникальных id на GT, имена из ``info_semantic``).

GT: ``semantic_map_*.npy``. Запуск из New-Proj::

    python scripts/eval_clip_sam_miou.py \\
        --manifest data/benchmarks/replica_sem100/manifest.json \\
        --queries_json replica_sem_benchmark/sample_queries.json \\
        --sam_ckpt ckpts/sam_vit_b_01ec64.pth \\
        --clip_model ViT-B-16 \\
        --clip_pretrained laion2b_s34b_b88k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_SCRIPTS = Path(__file__).resolve().parent
_PROJ = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sam_query_match as sqm  # noqa: E402


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_PROJ / path).resolve()


def _resolve_against_manifest(manifest_path: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    cand = (manifest_path.parent / path).resolve()
    if cand.exists():
        return cand
    return (_PROJ / path).resolve()


def _load_name_to_id(info_semantic_path: Path) -> dict[str, int]:
    data = json.loads(info_semantic_path.read_text(encoding="utf-8"))
    m: dict[str, int] = {}
    for c in data["classes"]:
        name = c["name"].strip().lower()
        m[name] = int(c["id"])
        m[name.replace("-", " ")] = int(c["id"])
    return m


def _load_id_to_name(info_semantic_path: Path) -> dict[int, str]:
    data = json.loads(info_semantic_path.read_text(encoding="utf-8"))
    return {int(c["id"]): c["name"] for c in data["classes"]}


def _resolve_class_id(query: str, name_to_id: dict[str, int]) -> int:
    q = query.strip().lower()
    q = q.removeprefix("a ").removeprefix("the ").removeprefix("an ").strip()
    q_hyp = q.replace(" ", "-")
    if q in name_to_id:
        return name_to_id[q]
    if q_hyp in name_to_id:
        return name_to_id[q_hyp]
    for k, v in name_to_id.items():
        if "-" not in k and k == q:
            return v
    raise ValueError(f"Класс не найден в info_semantic: {query!r}. Примеры: chair, table, sofa")


def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def _parse_queries_entry(entry: Any) -> tuple[list[str], dict[str, str]]:
    """Возвращает (class_names, optional per-class query override)."""
    if isinstance(entry, list):
        return [str(x).strip() for x in entry], {}
    if isinstance(entry, dict):
        classes = [str(x).strip() for x in entry.get("classes", [])]
        qmap = {k.strip().lower(): str(v) for k, v in entry.get("queries", {}).items()}
        return classes, qmap
    raise ValueError(f"Неверная запись queries JSON: {entry!r}")


def _load_queries_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "by_sample_id" in data:
        return data["by_sample_id"]
    if "samples" in data:
        out: dict[str, Any] = {}
        for row in data["samples"]:
            sid = str(row["id"])
            if "classes" in row:
                out[sid] = row["classes"]
            else:
                out[sid] = row
        return out
    raise ValueError("JSON: ожидаются ключи by_sample_id или samples")


def _classes_from_semantic(
    sem: np.ndarray, id_to_name: dict[int, str]
) -> list[str]:
    """Уникальные классы на карте (id > 0), имена из info_semantic."""
    u = np.unique(sem.astype(np.int64))
    names: list[str] = []
    for uid in u:
        if uid <= 0:
            continue
        if int(uid) in id_to_name:
            names.append(id_to_name[int(uid)])
    return names


@torch.inference_mode()
def _embed_texts_batch(
    texts: list[str],
    clip_model,
    tok,
    device: torch.device,
    ae,
) -> torch.Tensor:
    """(Q, D) L2-normalized (или латент при AE)."""
    tokens = tok(texts).to(device)
    q = F.normalize(clip_model.encode_text(tokens), dim=-1)
    if ae is not None:
        q = ae.encode(q.float())
    return q


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sam_ckpt", default="ckpts/sam_vit_b_01ec64.pth")
    ap.add_argument("--clip_model", default="ViT-B-16")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--encoder", default=None, help="AE .pth (Laion CLIP → latent)")
    ap.add_argument(
        "--class_name",
        default=None,
        help="Один класс на все кадры (режим без --queries_json)",
    )
    ap.add_argument(
        "--queries_json",
        type=Path,
        default=None,
        help="JSON: классы на каждый sample_id (см. докстринг)",
    )
    ap.add_argument(
        "--queries_from_gt",
        action="store_true",
        help="Игнорировать списки в JSON и брать классы с GT (уникальные id)",
    )
    ap.add_argument(
        "--text_template",
        default="a {class_name}",
        help="Шаблон текста CLIP, если нет переопределения в JSON (по умолчанию: 'a {class_name}')",
    )
    ap.add_argument("--text_query", default=None, help="Только для режима --class_name")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--include_empty_gt", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    if args.class_name is None and args.queries_json is None and not args.queries_from_gt:
        raise SystemExit("Задайте --class_name или --queries_json или --queries_from_gt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    man_path = _resolve(args.manifest)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    info_path = _resolve_against_manifest(
        man_path, manifest.get("info_semantic", "data/replica_v1/office_0/habitat/info_semantic.json")
    )
    name_to_id = _load_name_to_id(info_path)
    id_to_name = _load_id_to_name(info_path)

    queries_by_id: dict[str, Any] = {}
    if args.queries_json is not None:
        qpath = _resolve(args.queries_json)
        queries_by_id = _load_queries_json(qpath)

    ae = None
    if args.encoder:
        ae = sqm.load_ae(_resolve(args.encoder), device)

    clip_model, tok, preprocess = sqm.load_clip(args.clip_model, args.clip_pretrained, device)
    generator, _ = sqm.load_sam_generator(_resolve(args.sam_ckpt), device)

    ious: list[float] = []
    skipped = 0
    n_pairs = 0

    samples = manifest["samples"]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    single_class = args.class_name is not None and args.queries_json is None and not args.queries_from_gt
    if single_class:
        class_id = _resolve_class_id(args.class_name, name_to_id)
        text = args.text_query if args.text_query else f"a {args.class_name.strip()}"

    for sample in samples:
        sid = str(sample["id"])
        rgb_path = _resolve_against_manifest(man_path, sample["rgb"])
        sem_path = _resolve_against_manifest(man_path, sample["semantic"])
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            print(f"[skip] no rgb {rgb_path}")
            skipped += 1
            continue
        sem = np.load(str(sem_path))
        if sem.shape[:2] != bgr.shape[:2]:
            print(f"[skip] shape mismatch {rgb_path}")
            skipped += 1
            continue

        if single_class:
            class_names = [args.class_name]
            query_overrides: dict[str, str] = {}
            texts_for_clip = [text]
            class_ids = [class_id]
        else:
            if args.queries_from_gt:
                class_names = _classes_from_semantic(sem, id_to_name)
                query_overrides = {}
            else:
                if sid not in queries_by_id:
                    skipped += 1
                    continue
                class_names, query_overrides = _parse_queries_entry(queries_by_id[sid])
            if not class_names:
                skipped += 1
                continue
            texts_for_clip = []
            class_ids = []
            for cn in class_names:
                cn_l = cn.strip().lower()
                qtxt = query_overrides.get(cn_l)
                if qtxt is None:
                    qtxt = args.text_template.format(class_name=cn)
                texts_for_clip.append(qtxt)
                class_ids.append(_resolve_class_id(cn, name_to_id))

        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks_all = generator.generate(image_rgb)
        masks_all.sort(key=lambda r: -r["area"])
        if not masks_all:
            skipped += 1
            continue

        img_emb = sqm.embed_masks_clip(image_rgb, masks_all, clip_model, preprocess, device)
        if ae is not None:
            img_emb = ae.encode(img_emb.float())

        q = _embed_texts_batch(texts_for_clip, clip_model, tok, device, ae)
        sim = torch.matmul(img_emb.float(), q.float().T)
        best_indices = sim.argmax(dim=0).cpu().numpy()

        for ci, cid in enumerate(class_ids):
            gt = sem.astype(np.int64) == cid
            if not gt.any() and not args.include_empty_gt:
                continue
            best_i = int(best_indices[ci])
            pred = masks_all[best_i]["segmentation"].astype(bool)
            if not gt.any():
                iou = 0.0 if pred.any() else float("nan")
            else:
                iou = _binary_iou(pred, gt)
            n_pairs += 1
            if not np.isnan(iou):
                ious.append(iou)

    miou = float(np.nanmean(ious)) if ious else float("nan")
    mode = "single_class" if single_class else "multi_class_json_or_gt"
    print(f"mode={mode}  pairs_evaluated={len(ious)}  total_pairs_seen={n_pairs}  skipped_frames={skipped}")
    print(f"mIoU (mean binary IoU over class-image pairs): {miou:.4f}")


if __name__ == "__main__":
    main()
