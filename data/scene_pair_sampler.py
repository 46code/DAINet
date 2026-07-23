"""Batch sampler ensuring every scene in a batch contributes >= 2 directions.

Necessary for the cross-direction R-invariance loss to have valid same-scene
pairs in every batch.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class ScenePairBatchSampler(Sampler[list[int]]):
    """Yield batches built from pairs-per-scene.

    A batch of size `batch_size` contains `batch_size // 2` distinct scenes,
    each contributing 2 randomly chosen direction samples. If `batch_size` is
    odd, one extra random sample (any scene) is appended.
    """

    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 to form pairs")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.by_scene: dict[str, list[int]] = defaultdict(list)
        for i, (scene, _did) in enumerate(samples):
            self.by_scene[scene].append(i)
        self.scenes: list[str] = list(self.by_scene.keys())
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        pairs_per_batch = self.batch_size // 2
        leftover = self.batch_size % 2

        # Build a per-scene direction queue so EVERY (scene, dir) sample is
        # visited exactly once per epoch. Previously each scene contributed
        # only 2 directions per epoch — ~7% of the dataset was used.
        scene_queues: dict[str, list[int]] = {}
        for scene, idxs in self.by_scene.items():
            q = list(idxs)
            if self.shuffle:
                rng.shuffle(q)
            scene_queues[scene] = q

        scenes = list(self.scenes)
        active = [s for s in scenes if len(scene_queues[s]) >= 2]
        if self.shuffle:
            rng.shuffle(active)

        while len(active) >= pairs_per_batch:
            batch: list[int] = []
            picked = active[:pairs_per_batch]
            new_active: list[str] = []
            for scene in picked:
                q = scene_queues[scene]
                batch.extend(q[:2])
                del q[:2]
                if len(q) >= 2:
                    new_active.append(scene)
            # Carry remaining (unpicked) scenes forward
            remaining = active[pairs_per_batch:]
            if leftover:
                # Append one extra from any scene that still has samples
                for s in remaining + new_active:
                    if scene_queues[s]:
                        batch.append(scene_queues[s].pop(0))
                        break
            active = remaining + new_active
            if self.shuffle:
                # Rotate so a re-queued scene doesn't always go right back
                # into the next batch — keeps locality low.
                rng.shuffle(active)
            yield batch

    def __len__(self) -> int:
        pairs_per_batch = self.batch_size // 2
        # Each scene contributes floor(len(idxs)/2) pairs across the epoch.
        # Total batches ≈ total_pairs // pairs_per_batch.
        total_pairs = sum(len(v) // 2 for v in self.by_scene.values())
        return max(total_pairs // max(pairs_per_batch, 1), 0)


class SingleDirectionPerSceneSampler(Sampler[list[int]]):
    """Yield batches that contain at most ONE direction per scene per epoch.

    The single-direction ablation (`abl_single_direction`,
    `dataset.directions_per_scene: 1`) tests the value of the 25-direction
    supervision. With one direction per scene there are never two same-scene
    samples in a batch, so the cross-direction losses (``xdir_relight``,
    ``dir_consistency_R``) are inactive by construction — no explicit zeroing
    needed (they already return 0 with no same-scene pairs).

    Each epoch picks a *fresh* random direction per scene (seed + epoch), so the
    model still sees direction variety across epochs while never pairing two
    directions of the same scene within a step. ``ScenePairBatchSampler`` cannot
    be used here: it requires ≥ 2 directions per scene and would yield zero
    batches. The ``_epoch`` counter mirrors that class so the resume path in
    ``Trainer.fit`` advances it identically.
    """

    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.by_scene: dict[str, list[int]] = defaultdict(list)
        for i, (scene, _did) in enumerate(samples):
            self.by_scene[scene].append(i)
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        # One random direction-sample per scene this epoch.
        chosen = [rng.choice(idxs) for idxs in self.by_scene.values()]
        if self.shuffle:
            rng.shuffle(chosen)
        for start in range(0, len(chosen), self.batch_size):
            batch = chosen[start : start + self.batch_size]
            if batch:
                yield batch

    def __len__(self) -> int:
        n_scenes = len(self.by_scene)
        return (n_scenes + self.batch_size - 1) // max(self.batch_size, 1)
