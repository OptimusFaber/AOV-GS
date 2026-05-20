"""
Shared CLI helpers: mask collector (SAM vs CorrCLIP) and LangSplat mode.

Used by training / inference shell wrappers and Python entry points.
"""

from __future__ import annotations


def parse_mask_collector(value: str) -> int:
    """
    Return ``corrclip`` flag for ``activesgm.py`` (0 = plain SAM, 1 = CorrCLIP postproc).

    Accepted: ``sam``, ``plain``, ``default`` → 0; ``corrclip``, ``corr`` → 1.
    """
    v = value.strip().lower().replace("-", "").replace("_", "")
    if v in ("sam", "plain", "default", "0", "off", "no"):
        return 0
    if v in ("corrclip", "corr", "1", "on", "yes"):
        return 1
    raise ValueError(
        f"mask_collector must be 'sam' (plain SAM) or 'corrclip', got {value!r}"
    )


def mask_collector_label(corrclip_flag: int) -> str:
    return "corrclip" if int(corrclip_flag) else "sam"


def parse_lang_mode(value: str | None) -> bool | None:
    """
    Return ``use_langsplat_v2``: True = LangSplatV2, False = LangSplat (legacy AE),
    None = auto-detect from checkpoint at runtime.
    """
    if value is None:
        return None
    v = value.strip().lower().replace("-", "").replace("_", "")
    if v in ("auto", ""):
        return None
    if v in ("langsplatv2", "v2", "lsv2"):
        return True
    if v in ("langsplat", "legacy", "v1", "ls"):
        return False
    raise ValueError(
        f"lang_mode must be 'langsplatv2', 'langsplat', or 'auto', got {value!r}"
    )


def resolve_use_langsplat_v2(
    lang_mode: str | None,
    *,
    legacy_flag: bool = False,
) -> bool:
    """Combine ``--lang_mode`` and deprecated ``--legacy`` for training scripts."""
    parsed = parse_lang_mode(lang_mode)
    if parsed is not None:
        return parsed
    if legacy_flag:
        return False
    return True  # default: LangSplatV2
