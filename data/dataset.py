"""DAINetDataset — pairs a per-direction sRGB JPG input with a flat-lit JPG target.

Modes:

    train  — full conditioning + augmentation. Returns input, target, seg,
             meta-derived (phi, theta, brightness_norm), chrome-probe SH.
    val    — full conditioning, no augmentation. Same payload as train.
    test   — benchmark-fair. Returns ONLY input_rgb, target, scene,
             direction_id. No seg, no meta, no probes — never opened on
             disk. This matches the supervision other baselines receive
             when benchmarking against this model.

Data scope (locked): only `data/raw/mit_mi/jpg/{train,test}/<scene>/` and
`data/raw/mit_mi/jpg_gt/{train,test}/<scene>/`. No EXR, no albedo, no shading.

Per-sample dict (train / val):

    input_rgb:        [3, H, W] float32 in [0, 1]
    input_seg:        [1, H, W] int32 SAM2 ids (globally consistent K_fusion
                                 space when centroids are built; otherwise
                                 raw equivalence-class ids)
    region_seg:       [1, H, W] int32 chromaticity super-pixel ids
    material_seg:     [1, H, W] int32 MIT-MI material ids remapped through
                                 data/raw/mit_mi/material_taxonomy.json. Unknown / not-
                                 mapped pixels carry IGNORE_INDEX=255. Used
                                 by the material classification head + per-
                                 material reflectance variance loss
                                 (training-only auxiliary supervision).
    target:           [3, H, W] float32 in [0, 1]
    phi:              float32 scalar
    theta:            float32 scalar
    brightness_norm:  float32 scalar
    direction_id:     int64   scalar
    scene:            str
    sh_target:        [3, (l_max+1)**2] float32 chrome-SH coefficients
    has_sh:           bool scalar
    has_seg:          bool scalar
    has_material:     bool scalar — True when materials_mip2.png provided
                                    real (non-ignore) supervision.
    has_meta:         bool scalar
    normals:          [3, H, W] float32 in [-1, 1] — precomputed surface
                                                    normals (DSINE), fed as
                                                    extra encoder input.
                                                    Zeros when absent.
    has_normals:      bool scalar — True when normal_mip2.npy exists.

Per-sample dict (test):

    input_rgb:        [3, H, W] float32 in [0, 1]
    target:           [3, H, W] float32 in [0, 1]
    probe_mask:       [1, H, W] uint8  (1=scene, 0=probe region)
    direction_id:     int64   scalar
    scene:            str

Note on probe_mask: meta.json is opened only to derive the probe bounding
boxes / boundary polygons for chrome+gray spheres. The mask is consumed by
metrics and losses (not by the model), so the model still receives RGB-only
input — the benchmark-fair guarantee for `model.forward` is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import AugmentPipeline
from .material_io import IGNORE_INDEX as MATERIAL_IGNORE_INDEX
from .material_io import load_material_mask, num_classes as material_num_classes
from .meta_io import default_dir_entry, load_meta
from .probe_io import load_chrome_probe
from .probe_mask import build_probe_mask
from .probe_sh import chrome_to_sh
from .segmentation import load_chroma_superpixels, load_sam_mask, segment_image


def discover_scenes(jpg_root: str | Path, split: str = "train") -> list[str]:
    """List scene directories under `<jpg_root>/<split>/`."""
    root = Path(jpg_root) / split
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


class DAINetDataset(Dataset):
    def __init__(
        self,
        jpg_root: str | Path,
        jpg_gt_root: str | Path,
        scenes: Sequence[str],
        size: tuple[int, int] = (512, 640),
        augment: bool = False,
        sh_l_max: int = 4,
        direction_ids: Sequence[int] | None = None,
        mode: str = "train",
        augment_cfg: dict | None = None,
        sam_root: str | Path | None = None,
        superpixel_root: str | Path | None = None,
        split: str = "train",
        material_taxonomy_path: str | Path | None = None,
        use_material: bool = True,
        normals_root: str | Path | None = None,
    ):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be train/val/test, got {mode!r}")
        self.jpg_root = Path(jpg_root)
        self.jpg_gt_root = Path(jpg_gt_root)
        self.scenes = list(scenes)
        self.size = size
        self.augment = augment and mode != "test"
        self.sh_l_max = sh_l_max
        self.n_sh = (sh_l_max + 1) ** 2
        self.mode = mode
        self.split = split
        # SAM2 cache lives at <sam_root>/<split>/<scene>/sam_mip2.npy
        # Chromaticity super-pixels at <superpixel_root>/<split>/<scene>/chroma_clusters_mip2.npy
        # (loader transparently falls back to legacy .png caches if present)
        self.sam_root = Path(sam_root) if sam_root else Path("data/raw/mit_mi/sam_masks")
        self.superpixel_root = (
            Path(superpixel_root) if superpixel_root else Path("data/raw/mit_mi/superpixels")
        )
        # Normals cache (soft-required, training + val). Built offline by
        # scripts/precompute_normals.py (DSINE wrapper). Missing files emit
        # zeros + has_normals=False so the encoder's pre-fuse 1x1 conv
        # behaves as RGB-only on those samples.
        self.normals_root = (
            Path(normals_root)
            if normals_root
            else Path("data/raw/mit_mi/normals")
        )
        # Material auxiliary supervision (training/val only). The mask lives
        # next to the JPGs at <jpg_root>/<scene>/materials_mip2.png. Test mode
        # never opens it (benchmark-fair). If the taxonomy file is missing we
        # silently disable material loading.
        self.material_taxonomy_path = (
            Path(material_taxonomy_path) if material_taxonomy_path else None
        )
        self.use_material = bool(use_material) and self.mode != "test"
        self.k_material = 0
        if self.use_material and self.material_taxonomy_path is not None:
            try:
                self.k_material = material_num_classes(self.material_taxonomy_path)
            except FileNotFoundError:
                self.use_material = False
        # Augmentation pipeline applies to INPUT only (target stays canonical).
        # Built once per dataset; per-sample randomness comes from a seeded rng.
        self._aug_pipeline = AugmentPipeline(augment_cfg) if self.augment else None

        allowed_dirs = set(range(25)) if direction_ids is None else set(direction_ids)

        # Enumerate (scene, direction_id) pairs by what's on disk
        self.samples: list[tuple[str, int]] = []
        for scene in self.scenes:
            sd = self.jpg_root / scene
            if not sd.exists():
                continue
            for did in sorted(allowed_dirs):
                if (sd / f"dir_{did}_mip2.jpg").exists():
                    self.samples.append((scene, did))

        # Per-scene asset reuse is handled by bounded module-level lru_caches
        # in data/segmentation.py and data/material_io.py. The dataset itself
        # holds no scene-level dict cache — those were unbounded and, combined
        # with persistent_workers=True × num_workers, OOM-killed the host.
        # (scene, dir_id) -> (sh_coeffs, has_sh). SH coefficients are tiny
        # (3*(l_max+1)^2 floats), so this dict stays small.
        self._sh_cache: dict[tuple[str, int], tuple[np.ndarray, bool]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _scene_dir(self, scene: str) -> Path:
        return self.jpg_root / scene

    def _load_target(self, scene: str) -> np.ndarray:
        candidates = [
            self.jpg_gt_root / scene / "target_clean.jpg",
            self.jpg_root / scene / "target_clean.jpg",
        ]
        for p in candidates:
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"target_clean.jpg missing for scene={scene}")

    def _is_single_view(self) -> bool:
        """True when sam_root points at the single-view cache (sam_masks_sv),
        which must be (re)built with --n_views 1 --min_views 1. Drives the
        precompute hint emitted when a cache file is missing."""
        return self.sam_root.name.endswith("_sv")

    def _load_sam_seg(self, scene: str) -> np.ndarray | None:
        """SAM2 segmentation map (semantic), used by encoder FiLM only."""
        p = self.sam_root / self.split / scene / "sam_mip2.npy"
        return load_sam_mask(p)

    def _load_chroma_superpixels(self, scene: str) -> np.ndarray | None:
        """Chromaticity K-means super-pixels, used by region-aware losses."""
        p = self.superpixel_root / self.split / scene / "chroma_clusters_mip2.npy"
        return load_chroma_superpixels(p)

    def _load_normals(self, scene: str) -> np.ndarray | None:
        """Pre-computed surface normals (DSINE).

        Returns float32 H×W×3 in [-1, 1] if the cache exists, else None
        (the dataset emits has_normals = False and zeros are passed in).
        The normals are world-space (or DSINE's canonical camera frame —
        kept exactly as the precompute script wrote them; the encoder
        treats them as a learned feature, not a physical normal field,
        so the frame convention is internal).
        """
        p = self.normals_root / self.split / scene / "normal_mip2.npy"
        if not p.exists():
            return None
        try:
            arr = np.load(p)
        except Exception:
            return None
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    def _load_material(self, scene: str) -> np.ndarray | None:
        """MIT-MI material mask, remapped to contiguous training ids.

        Training-only auxiliary signal. Missing files return None — the
        sample then carries a 255-filled (ignore) stub and ``has_material``
        is False.
        """
        if not self.use_material or self.material_taxonomy_path is None:
            return None
        p = self.jpg_root / scene / "materials_mip2.png"
        return load_material_mask(p, self.material_taxonomy_path)

    def _load_sh(self, scene: str, direction_id: int) -> tuple[np.ndarray, bool]:
        key = (scene, direction_id)
        if key in self._sh_cache:
            return self._sh_cache[key]
        chrome = load_chrome_probe(self._scene_dir(scene), direction_id)
        if chrome is None:
            result = (np.zeros((3, self.n_sh), dtype=np.float32), False)
        else:
            result = (chrome_to_sh(chrome, l_max=self.sh_l_max), True)
        self._sh_cache[key] = result
        return result

    def _resize(self, arr: np.ndarray, interp: int) -> np.ndarray:
        h, w = self.size
        return cv2.resize(arr, (w, h), interpolation=interp)

    def __getitem__(self, idx: int) -> dict:
        scene, did = self.samples[idx]
        sd = self._scene_dir(scene)

        # Load input + target (uint8 RGB)
        inp_path = sd / f"dir_{did}_mip2.jpg"
        bgr = cv2.imread(str(inp_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(inp_path)
        inp_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mip2_h, mip2_w = inp_u8.shape[:2]
        tgt_u8 = self._load_target(scene)

        # Resize
        inp_u8 = self._resize(inp_u8, cv2.INTER_LINEAR)
        tgt_u8 = self._resize(tgt_u8, cv2.INTER_LINEAR)

        # Probe-region mask (val/test only). 1 = scene content, 0 = chrome/gray
        # sphere. Built from meta.json bounding boxes / polygons; consumed by
        # metrics and losses, never by the model.
        probe_mask_t: torch.Tensor | None = None
        if self.mode in ("val", "test"):
            meta_path = sd / "meta.json"
            pm = build_probe_mask(meta_path, (mip2_h, mip2_w), self.size)
            probe_mask_t = torch.from_numpy(pm.astype(np.uint8))[None, ...]

        # Augmentation applies to INPUT only — the target is the canonical
        # flat-lit reference. The model learns to invert these perturbations.
        # RNG is seeded from os entropy on every call so the same sample gets
        # a fresh perturbation every epoch (previous `default_rng(idx)` froze
        # augmentation after epoch 1).
        if self.augment and self._aug_pipeline is not None:
            rng = np.random.default_rng()
            inp_u8 = self._aug_pipeline(inp_u8, rng)

        # To CHW float [0,1]
        inp = np.ascontiguousarray(inp_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)
        tgt = np.ascontiguousarray(tgt_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)

        if self.mode == "test":
            # Single-RGB-input contract: the model still receives ONLY the sRGB
            # image, but we attach the priors it computes from that RGB at full
            # capacity — SAM2 ids (precomputed for the test split through the
            # FROZEN train centroids) and DSINE normals — so test scoring matches
            # deployment. (φ,θ,b) are NOT fed to the model (the direction head
            # predicts them); meta is loaded for DIAGNOSTICS only (φ/θ heatmap,
            # failure buckets) — scripts/test.py never passes it to the model.
            sam_seg = self._load_sam_seg(scene)
            if sam_seg is None:
                # Single-view caches (sam_masks_sv, used by abl_singleview_sam)
                # MUST be rebuilt with --n_views 1 --min_views 1, else stage raw
                # defaults to 25-view fusion (corrupting the ablation) and skips
                # every scene (default --min_views 20 > the 1 view on disk).
                raw_flags = " --n_views 1 --min_views 1" if self._is_single_view() else ""
                raise FileNotFoundError(
                    f"SAM2 cache missing for test scene={scene}. Precompute the "
                    f"{self.split} split first:\n  python scripts/precompute_sam.py "
                    f"--stage raw --root {self.jpg_root.parent} --splits {self.split} "
                    f"--out {self.sam_root}{raw_flags}\n"
                    f"  python scripts/precompute_sam.py --stage final --root "
                    f"{self.jpg_root.parent} --splits {self.split} --out {self.sam_root} "
                    f"--centroids data/raw/mit_mi/sam_fusion_centroids.npy"
                )
            sam_seg = self._resize(sam_seg, cv2.INTER_NEAREST)
            input_seg = np.ascontiguousarray(sam_seg.astype(np.int32))[None, ...]

            normals_arr = self._load_normals(scene)
            if normals_arr is None:
                normals = np.zeros((3, *self.size), dtype=np.float32)
                has_normals = False
            else:
                if normals_arr.shape[:2] != tuple(self.size):
                    normals_arr = self._resize(normals_arr, cv2.INTER_LINEAR)
                normals = np.ascontiguousarray(
                    normals_arr.astype(np.float32).transpose(2, 0, 1)
                )
                has_normals = True

            meta = load_meta(sd)
            dir_meta = meta["directions"].get(did, default_dir_entry(did))
            payload = {
                "input_rgb": torch.from_numpy(inp),
                "input_seg": torch.from_numpy(input_seg),  # SAM2 — for FiLM
                "target": torch.from_numpy(tgt),
                "normals": torch.from_numpy(normals),
                "has_normals": torch.tensor(has_normals, dtype=torch.bool),
                "direction_id": torch.tensor(did, dtype=torch.long),
                "scene": scene,
                # ---- diagnostics only (NOT model inputs at test) ----
                "phi": torch.tensor(dir_meta["phi"], dtype=torch.float32),
                "theta": torch.tensor(dir_meta["theta"], dtype=torch.float32),
                "brightness_norm": torch.tensor(
                    dir_meta["brightness_normalization"], dtype=torch.float32
                ),
                "has_meta": torch.tensor(meta["present"], dtype=torch.bool),
            }
            if probe_mask_t is not None:
                payload["probe_mask"] = probe_mask_t
            return payload

        # train / val: load dual conditioning
        # 1) SAM2 mask → encoder FiLM (semantic / geometric).
        # 2) Chromaticity super-pixels → region-aware losses (material).
        sam_seg = self._load_sam_seg(scene)
        if sam_seg is None:
            raw_flags = " --n_views 1 --min_views 1" if self._is_single_view() else ""
            raise FileNotFoundError(
                f"SAM2 cache missing for scene={scene}. "
                f"Run: python scripts/precompute_sam.py --stage raw "
                f"--root {self.jpg_root.parent} "
                f"--splits {self.split} --out {self.sam_root}{raw_flags}"
            )
        sam_seg = self._resize(sam_seg, cv2.INTER_NEAREST)
        input_seg = np.ascontiguousarray(sam_seg.astype(np.int32))[None, ...]

        region_seg_arr = self._load_chroma_superpixels(scene)
        if region_seg_arr is None:
            raise FileNotFoundError(
                f"Chromaticity super-pixel cache missing for scene={scene}. "
                f"Run: python scripts/precompute_superpixels.py --gt_root "
                f"{self.jpg_gt_root.parent} --splits {self.split} "
                f"--out {self.superpixel_root}"
            )
        region_seg_arr = self._resize(region_seg_arr, cv2.INTER_NEAREST)
        region_seg = np.ascontiguousarray(region_seg_arr.astype(np.int32))[None, ...]
        has_seg = True

        # Material auxiliary supervision (training-only). Soft-fail: scenes
        # without materials_mip2.png emit a 255-filled ignore stub and the
        # loss is gated by has_material below.
        material_arr = self._load_material(scene)
        if material_arr is None:
            material_seg = np.full((1, *self.size), MATERIAL_IGNORE_INDEX, dtype=np.int32)
            has_material = False
        else:
            material_arr = self._resize(material_arr.astype(np.int32), cv2.INTER_NEAREST)
            material_seg = np.ascontiguousarray(material_arr.astype(np.int32))[None, ...]
            # If every pixel ended up in the ignore bucket the supervision is
            # degenerate; treat that as missing.
            has_material = bool((material_seg != MATERIAL_IGNORE_INDEX).any())

        meta = load_meta(sd)
        has_meta = meta["present"]
        dir_meta = meta["directions"].get(did, default_dir_entry(did))

        sh, has_sh = self._load_sh(scene, did)

        # Surface normals (DSINE, precomputed). Per-scene cache — DSINE
        # outputs are direction-invariant geometry, so one pass per
        # scene is enough. Missing cache → zeros + has_normals=False;
        # the encoder's pre-fuse 1x1 conv treats that path as RGB-only.
        normals_arr = self._load_normals(scene)
        if normals_arr is None:
            normals = np.zeros((3, *self.size), dtype=np.float32)
            has_normals = False
        else:
            if normals_arr.shape[:2] != tuple(self.size):
                normals_arr = self._resize(normals_arr, cv2.INTER_LINEAR)
            normals = np.ascontiguousarray(
                normals_arr.astype(np.float32).transpose(2, 0, 1)
            )
            has_normals = True

        payload = {
            "input_rgb": torch.from_numpy(inp),
            "input_seg": torch.from_numpy(input_seg),  # SAM2 — for FiLM
            "region_seg": torch.from_numpy(region_seg),  # super-pixels — for region losses
            "material_seg": torch.from_numpy(material_seg),  # MIT materials — aux supervision
            "target": torch.from_numpy(tgt),
            "phi": torch.tensor(dir_meta["phi"], dtype=torch.float32),
            "theta": torch.tensor(dir_meta["theta"], dtype=torch.float32),
            "brightness_norm": torch.tensor(
                dir_meta["brightness_normalization"], dtype=torch.float32
            ),
            "direction_id": torch.tensor(did, dtype=torch.long),
            "scene": scene,
            "sh_target": torch.from_numpy(sh).float(),
            "has_sh": torch.tensor(has_sh, dtype=torch.bool),
            "has_seg": torch.tensor(has_seg, dtype=torch.bool),
            "has_material": torch.tensor(has_material, dtype=torch.bool),
            "has_meta": torch.tensor(has_meta, dtype=torch.bool),
            "normals": torch.from_numpy(normals),
            "has_normals": torch.tensor(has_normals, dtype=torch.bool),
        }
        if probe_mask_t is not None:
            payload["probe_mask"] = probe_mask_t
        return payload
