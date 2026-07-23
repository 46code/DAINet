"""Deterministic train/val/test scene splits and leave-K-directions-out helper."""

from __future__ import annotations

import hashlib
import random
from typing import Iterable


def deterministic_split(
    scenes: Iterable[str],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """SHA256-bucket every scene name into one of three splits.

    Always returns the same partition for a given set of scene names.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")
    train_scenes: list[str] = []
    val_scenes: list[str] = []
    test_scenes: list[str] = []
    for s in sorted(scenes):
        h = hashlib.sha256(s.encode()).hexdigest()
        b = int(h[:8], 16) / 2**32
        if b < train:
            train_scenes.append(s)
        elif b < train + val:
            val_scenes.append(s)
        else:
            test_scenes.append(s)
    return train_scenes, val_scenes, test_scenes


def train_val_split(
    scenes: Iterable[str], val_ratio: float = 0.1
) -> tuple[list[str], list[str]]:
    """Deterministic train/val partition of a single scene list.

    Used when train and test live in separate folders on disk and we only want
    to split the training folder. When the scene list is large the SHA256
    bucketing produces a clean stratification; for very small lists (~ tens of
    scenes) we top up the underpopulated bucket from the other side so that
    both buckets contain at least one scene whenever the input has two or
    more.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    sorted_scenes = sorted(scenes)
    train_scenes: list[str] = []
    val_scenes: list[str] = []
    for s in sorted_scenes:
        h = hashlib.sha256(s.encode()).hexdigest()
        b = int(h[:8], 16) / 2**32
        if b < val_ratio:
            val_scenes.append(s)
        else:
            train_scenes.append(s)
    # Robustness for tiny datasets (smoke tests, fast configs on toy data):
    # if either side is empty but the input has >= 2 scenes, move one across
    # deterministically (by sorted order) so callers don't end up with an
    # empty dataloader.
    if len(sorted_scenes) >= 2:
        if not train_scenes:
            train_scenes.append(val_scenes.pop(0))
        elif not val_scenes:
            val_scenes.append(train_scenes.pop())
    return train_scenes, val_scenes


def subset_scenes(scenes: Iterable[str], ratio: float, seed: int = 1337) -> list[str]:
    """Deterministically subset a scene list by ratio.

    Uses SHA256 bucketing so the same scenes are selected across runs for any
    given (scene names, ratio) pair. Ratio < 1.0 yields a smaller deterministic
    subset for ablation thinning (set `dataset.subset_ratio` < 1.0 in a config).
    """
    if ratio >= 1.0:
        return sorted(scenes)
    if ratio <= 0.0:
        return []
    out: list[str] = []
    for s in sorted(scenes):
        # Salt the hash so subset selection is independent of train/val/test bucketing.
        h = hashlib.sha256(f"{seed}::{s}".encode()).hexdigest()
        b = int(h[:8], 16) / 2**32
        if b < ratio:
            out.append(s)
    return out


def leave_k_directions_out(directions: list[int], k: int, seed: int = 42) -> tuple[list[int], list[int]]:
    """Hold out k directions for OOD generalization eval.

    Returns:
        (train_dirs, held_out_dirs) — disjoint, sorted lists.
    """
    rng = random.Random(seed)
    perm = list(directions)
    rng.shuffle(perm)
    held_out = sorted(perm[:k])
    train_dirs = sorted(perm[k:])
    return train_dirs, held_out
