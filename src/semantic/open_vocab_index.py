"""
Open-vocabulary keyframe index powered by CLIP.

Maintains a dictionary  {keyframe_id → CLIPRecord}  where each record
stores the camera pose (c2w, 4×4) and the corresponding CLIP image
embedding.  At query time the index performs a fast cosine search and
returns the top-K keyframe poses ranked by similarity to a text prompt.

Design goals
------------
* **Non-blocking** – ``update()`` is called from the main planning
  loop only every ``update_every`` steps (configurable).  Heavy CLIP
  inference happens in-place but is shielded by a dirty-flag so that
  re-encoding the same keyframes is avoided.
* **Thread-safe** – a lightweight ``threading.Lock`` guards the shared
  state, making it safe to call ``update()`` from a background thread
  if the user chooses to do so.
* **Device-flexible** – CLIP runs on the device configured in
  ``CLIPEncoder``, while all pose tensors live on CPU to avoid wasting
  GPU memory.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.semantic.clip_encoder import CLIPEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CLIPRecord:
    """Single keyframe entry in the index."""
    kf_id: int
    c2w: torch.Tensor          # [4, 4], camera-to-world, CPU
    embedding: torch.Tensor    # [D], L2-normalised, CLIP device


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class OpenVocabIndex:
    """CLIP-based open-vocabulary index over accumulated keyframes.

    Parameters
    ----------
    clip_encoder:
        A pre-built :class:`CLIPEncoder` instance.
    update_every:
        Number of planning steps between index refresh calls.
        Setting to ``1`` refreshes on every planning step.
    top_k:
        Default number of results returned by :meth:`query`.
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder,
        update_every: int = 10,
        top_k: int = 5,
    ) -> None:
        self.encoder = clip_encoder
        self.update_every = update_every
        self.top_k = top_k

        # {kf_id: CLIPRecord}
        self._records: Dict[int, CLIPRecord] = {}
        self._lock = threading.Lock()

        # Track which ids have already been encoded to avoid re-work
        self._encoded_ids: set = set()

        # Step counter for deciding when to refresh
        self._step: int = 0

    # ------------------------------------------------------------------
    # Updating the index
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the index to *path* (a ``.pt`` file).

        Saves all kf_ids, c2w poses and CLIP embeddings so the index can
        be reloaded later for the interactive validation script.
        """
        with self._lock:
            records = list(self._records.values())
        if not records:
            logger.warning("OpenVocabIndex.save(): index is empty, nothing saved.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "kf_ids":     [r.kf_id    for r in records],
            "c2w_poses":  torch.stack([r.c2w       for r in records]),   # [N,4,4]
            "embeddings": torch.stack([r.embedding for r in records]),   # [N,D]
            "model_name": self.encoder.model_name,
            "pretrained":  self.encoder.pretrained,
        }
        torch.save(payload, path)
        logger.info(
            "OpenVocabIndex: saved %d keyframe embeddings → %s",
            len(records), path,
        )
        print(f"[CLIP] Saved index: {len(records)} keyframes → {path}")

    @classmethod
    def load(cls, path: str, clip_encoder: "CLIPEncoder", **kwargs) -> "OpenVocabIndex":
        """Restore a previously saved index from *path*.

        Parameters
        ----------
        path:
            Path to the ``.pt`` file written by :meth:`save`.
        clip_encoder:
            An initialised :class:`CLIPEncoder` (used for future text queries).
        """
        payload = torch.load(path, map_location="cpu")
        index = cls(clip_encoder, **kwargs)
        for kf_id, c2w, emb in zip(
            payload["kf_ids"],
            payload["c2w_poses"],
            payload["embeddings"],
        ):
            index._records[kf_id] = CLIPRecord(
                kf_id=int(kf_id),
                c2w=c2w,
                embedding=emb,
            )
            index._encoded_ids.add(int(kf_id))
        logger.info(
            "OpenVocabIndex: loaded %d keyframes from %s", len(index._records), path
        )
        print(f"[CLIP] Loaded index: {len(index._records)} keyframes from {path}")
        return index

    def maybe_update(self, keyframe_list: List[dict]) -> None:
        """Call this every planning step.

        The method only performs CLIP inference when the internal counter
        reaches ``update_every``.  Newly added keyframes that were not yet
        encoded are processed regardless of the counter.

        Parameters
        ----------
        keyframe_list:
            The SLAM ``keyframe_list`` (list of dicts with keys
            ``'id'``, ``'est_w2c'``, ``'color'``).
        """
        self._step += 1
        new_kfs = [kf for kf in keyframe_list if kf['id'] not in self._encoded_ids]
        if not new_kfs:
            return
        if self._step % self.update_every != 0 and self._step != 1:
            return
        self.update(new_kfs)

    def update(self, keyframe_list: List[dict]) -> None:
        """Encode a list of keyframes and add them to the index.

        Already-encoded keyframes (by id) are silently skipped.

        Parameters
        ----------
        keyframe_list:
            Same format as ``SlatamOurs.keyframe_list``.
        """
        new_kfs = [kf for kf in keyframe_list if kf['id'] not in self._encoded_ids]
        if not new_kfs:
            return

        print(f"[CLIP] Encoding {len(new_kfs)} new keyframe(s)…")
        logger.debug("OpenVocabIndex: encoding %d new keyframe(s)…", len(new_kfs))

        images = [kf['color'] for kf in new_kfs]
        embeddings = self.encoder.encode_images_batch(images)  # [N, D]

        with self._lock:
            for kf, emb in zip(new_kfs, embeddings):
                kf_id = kf['id']
                c2w = torch.inverse(kf['est_w2c'].cpu().float())  # w2c → c2w
                self._records[kf_id] = CLIPRecord(
                    kf_id=kf_id,
                    c2w=c2w,
                    embedding=emb.cpu(),
                )
                self._encoded_ids.add(kf_id)

        total = len(self._records)
        print(f"[CLIP] Index updated: {total} keyframes total.")
        logger.debug("OpenVocabIndex: total keyframes indexed = %d", total)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        text_prompt: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[float, int, torch.Tensor]]:
        """Find the keyframes most relevant to *text_prompt*.

        Parameters
        ----------
        text_prompt:
            Free-form text, e.g. ``"a white office chair"``.
        top_k:
            Number of results to return.  Defaults to ``self.top_k``.

        Returns
        -------
        List of ``(score, kf_id, c2w_pose)`` tuples sorted by
        descending similarity score.  ``c2w_pose`` is a ``[4, 4]``
        CPU tensor in the SplaTAM coordinate system.
        """
        k = top_k if top_k is not None else self.top_k

        with self._lock:
            if not self._records:
                logger.warning("OpenVocabIndex.query(): index is empty.")
                return []
            records = list(self._records.values())

        text_emb = self.encoder.encode_text(text_prompt).cpu()

        emb_matrix = torch.stack([r.embedding for r in records])  # [N, D]
        scores = (emb_matrix @ text_emb).tolist()                 # [N]

        ranked = sorted(zip(scores, records), key=lambda x: -x[0])
        results: List[Tuple[float, int, torch.Tensor]] = [
            (score, rec.kf_id, rec.c2w)
            for score, rec in ranked[:k]
        ]

        logger.debug(
            "OpenVocabIndex.query('%s'): top score = %.4f (kf_id=%d)",
            text_prompt,
            results[0][0] if results else float("nan"),
            results[0][1] if results else -1,
        )
        return results

    def get_goal_pose(self, text_prompt: str) -> Optional[torch.Tensor]:
        """Return the single best-matching camera-to-world pose.

        Parameters
        ----------
        text_prompt:
            Free-form navigation goal.

        Returns
        -------
        ``[4, 4]`` CPU c2w tensor, or ``None`` if the index is empty.
        """
        results = self.query(text_prompt, top_k=1)
        if not results:
            return None
        score, kf_id, c2w = results[0]
        logger.info(
            "Goal pose found: kf_id=%d, CLIP score=%.4f", kf_id, score
        )
        return c2w

    def get_top_k_poses(
        self,
        text_prompt: str,
        top_k: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Return top-K c2w poses ranked by CLIP similarity.

        Convenience wrapper around :meth:`query`.
        """
        return [c2w for _, _, c2w in self.query(text_prompt, top_k=top_k)]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"OpenVocabIndex("
            f"indexed={len(self._records)}, "
            f"update_every={self.update_every}, "
            f"top_k={self.top_k})"
        )

    def get_all_scores(self, text_prompt: str) -> Dict[int, float]:
        """Return a ``{kf_id: score}`` dict for all indexed keyframes.

        Useful for visualising the CLIP relevance map over the scene.
        """
        results = self.query(text_prompt, top_k=len(self._records))
        return {kf_id: score for score, kf_id, _ in results}
