"""
OpenCLIP (model_name, pretrained) pairs for ``eval_clip_sam_systematic.py`` + rough VRAM (GB).

Notes
-----
* **ViT-H-14 / laion2b_s32b_b82k** — в ``open_clip`` тег ``laion2b_s32b_b82k`` есть у **ViT-L-14**,
  у **ViT-H-14** LAION-вес — ``laion2b_s32b_b79k`` (добавлен он).
* **MetaCLIP H/14** — ``ViT-H-14`` + ``metaclip_fullcc`` (или ``metaclip_altogether``).
* **MobileCLIP2** — в ``open_clip_torch`` 2.32 нет имён ``MobileCLIP2-*``; стоит
  ``pip install -U open_clip_torch`` и при появлении пар — добавить в список.
  Пока: **MobileCLIP-B** / ``datacompdr`` как ближайший лёгкий вариант из того же семейства Apple.
* Оценки VRAM — ориентир для ``--vram_limit_gb`` (fp16 inference, типичный батч кропов).
"""
from __future__ import annotations

# (model_name, pretrained)
CLIP_CONFIGS_DEFAULT: list[tuple[str, str]] = [
    # Original sweep (9)
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-32", "datacomp_xl_s13b_b90k"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("ViT-B-16-SigLIP", "webli"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-L-14", "dfn2b"),
    ("ViT-L-14", "datacomp_xl_s13b_b90k"),
    ("MobileCLIP-S1", "datacompdr"),
    ("MobileCLIP-S2", "datacompdr"),
    # ViT-H LAION (H-14 uses b79k in open_clip; L-14 has b82k — see module docstring)
    ("ViT-H-14", "laion2b_s32b_b79k"),
    # MetaCLIP H/14
    ("ViT-H-14", "metaclip_fullcc"),
    # SigLIP2 B / L
    ("ViT-B-16-SigLIP2", "webli"),
    ("ViT-L-16-SigLIP2-384", "webli"),
    # EVA02-CLIP
    ("EVA02-B-16", "merged2b_s8b_b131k"),
    ("EVA02-L-14", "merged2b_s4b_b131k"),
    # DFN ViT-H (large Data Filtering / “DFN-5B” scale in open_clip)
    ("ViT-H-14", "dfn5b"),
    # Mobile family (MobileCLIP2-* requires newer open_clip — see docstring)
    ("MobileCLIP-B", "datacompdr"),
]

# Peak inference VRAM guess (GB) by **model_name** (same architecture → same bucket).
CLIP_VRAM_ESTIMATES_GB: dict[str, float] = {
    "ViT-B-32": 0.5,
    "ViT-B-16": 0.5,
    "ViT-B-16-SigLIP": 0.6,
    "ViT-B-16-SigLIP2": 0.9,
    "ViT-L-14": 1.2,
    "ViT-L-16-SigLIP2-384": 2.0,
    "ViT-H-14": 2.8,
    "EVA02-B-16": 0.65,
    "EVA02-L-14": 1.35,
    "MobileCLIP-S1": 0.3,
    "MobileCLIP-S2": 0.4,
    "MobileCLIP-B": 0.35,
    "MobileCLIP2-S0": 0.25,
    "MobileCLIP2-S2": 0.35,
    "MobileCLIP2-B": 0.40,
}

def _unique_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def clip_configs_for_eval() -> list[tuple[str, str]]:
    """
    Sweep list: static catalog + ``MobileCLIP2-*`` if present in installed ``open_clip``.
    """
    out = list(CLIP_CONFIGS_DEFAULT)
    try:
        import open_clip

        listed = set(open_clip.list_pretrained())
        for pair in (
            ("MobileCLIP2-S0", "dfndr2b"),
            ("MobileCLIP2-S2", "dfndr2b"),
            ("MobileCLIP2-B", "dfndr2b"),
        ):
            if pair in listed and pair not in out:
                out.append(pair)
    except Exception:
        pass
    return out


# Unique pairs for download / warm-cache (includes optional MobileCLIP2)
CLIP_DOWNLOAD_PAIRS: list[tuple[str, str]] = _unique_pairs(clip_configs_for_eval())
