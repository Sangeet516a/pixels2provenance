"""Tabular pre-processing, the ImageNet input of the frozen CNNs, and the galaxy / label tables."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .utils import signed_log1p


class TabularPrep:
    """Median imputation -> signed log -> RobustScaler. Columns that are entirely
    NaN in the training data (e.g. colours when only one band exists) are dropped."""

    def __init__(self, columns):
        self.columns = list(columns)
        self.kept = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()

    def fit(self, df: pd.DataFrame):
        cols = [c for c in self.columns if c in df.columns]
        X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        keep = ~np.all(np.isnan(X), axis=0)
        self.kept = [c for c, k in zip(cols, keep) if k]
        X = signed_log1p(self.imputer.fit_transform(df[self.kept].apply(pd.to_numeric, errors="coerce")))
        self.scaler.fit(X)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.kept).apply(pd.to_numeric, errors="coerce")
        return self.scaler.transform(signed_log1p(self.imputer.transform(X))).astype(np.float32)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "TabularPrep":
        return joblib.load(path)


def load_cutouts(ctx):
    """PSF-matched cutouts written by Stage 1d (memory-mapped, row = cutout_index)."""
    return np.load(ctx.require("1d", "cutouts.npy"), mmap_mode="r")


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def cnn_stamp(cube, seg, r_petro, area_px, cfg, atlas_cfg=None):
    """One galaxy -> the image the CNNs see (Stage 4a), still (3, s, s) blue -> red, sky-subtracted:
      1. cosmic rays: the Stage 9 single-band / sharpness cleaner (supervised.cnn_clean_cosmic_rays);
      2. neighbours: pixels of OTHER sources (Stage 1c segmentation code 2, grown by 2 px) are replaced by the sky
         level, so the features describe the target only (supervised.cnn_mask_neighbours);
      3. crop: a square of cnn_crop_petro x the Petrosian radius (at least cnn_crop_min px, at most the 128-px
         cutout) centred on the galaxy, so the galaxy FILLS the image (median galaxy ~10 x 10 px of 128 x 128:
         without the crop the pooled CNN features describe mostly sky)."""
    from scipy import ndimage
    a = np.nan_to_num(np.asarray(cube, np.float32))
    n = a.shape[-1]
    if cfg.get("cnn_clean_cosmic_rays", True):
        from .stages.stage9_galaxy_atlas import clean_cosmic_rays
        a = clean_cosmic_rays(a, list(range(a.shape[0])), atlas_cfg or {})
    border = np.concatenate([a[:, :8].reshape(a.shape[0], -1), a[:, -8:].reshape(a.shape[0], -1)], axis=1)
    sky = np.median(border, axis=1)
    if cfg.get("cnn_mask_neighbours", True) and seg is not None:
        sg = np.asarray(seg)
        nb = ndimage.binary_dilation(sg == 2, iterations=2) & (sg != 1)
        if nb.any():
            a = np.where(nb[None], sky[:, None, None], a)
    a = a - sky[:, None, None]
    if cfg.get("cnn_crop", True):
        r = float(r_petro) if r_petro is not None and np.isfinite(r_petro) and r_petro > 0 else \
            (float(np.sqrt(area_px / np.pi)) if area_px is not None and np.isfinite(area_px) and area_px > 0 else 16.0)
        side = int(np.clip(round(float(cfg.get("cnn_crop_petro", 4.0)) * r), int(cfg.get("cnn_crop_min", 32)), n))
        side += side % 2
        h, c = side // 2, n // 2
        a = a[:, c - h:c + h, c - h:c + h]
    return a


def imagenet_input(cubes, size=224, q_lo=1.0, q_hi=99.5, soft=0.1):
    """Stamps (list or array of (3, s, s), bands blue -> red; sizes may differ) -> ImageNet-ready tensor (N, 3, size, size).
      1. one asinh stretch per stamp, the SAME for all three bands (percentiles of all its pixels), so the colour
         of the galaxy survives (a separate stretch per band would make every galaxy grey);
      2. channels R, G, B = reddest, middle, bluest band (F775W, F606W, F435W); a band without data stays 0;
      3. ImageNet mean / std per channel (the statistics the frozen BatchNorm layers expect);
      4. bilinear resize to size x size (224 = the image size the networks were trained on)."""
    import torch
    import torch.nn.functional as F
    out = []
    for cube in cubes:
        x = np.nan_to_num(np.asarray(cube, np.float32))[::-1]                  # blue->red  ->  R, G, B
        lo, hi = np.percentile(x, [q_lo, q_hi])
        v = np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)
        v = np.arcsinh(v / soft) / np.arcsinh(1.0 / soft)
        t = torch.from_numpy(np.ascontiguousarray((v - IMAGENET_MEAN) / IMAGENET_STD, dtype=np.float32))[None]
        if t.shape[-1] != size:
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        out.append(t)
    return torch.cat(out)


def physical_matrix(df, columns):
    """The physical (Pass-2) features of `columns` (18 in the final model) as they are: trees need no scaling, and NaN (not measurable) is
    handled natively by XGBoost."""
    return df.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)


def galaxy_table(ctx):
    """Every galaxy-track object with Pass-2 features (overwriting Pass 1) and labels."""
    from .context import load_objects
    from .utils import as_bool, galaxy_track
    df = load_objects(ctx, ["sources", "pass1", "segregation", "labels", "pass2"])
    df = df[galaxy_track(df)].copy()
    df["is_pseudo_label"] = as_bool(df.is_pseudo_label.fillna(True))
    df["edge_flag"] = as_bool(df.edge_flag)
    return df


def _trainable_buffer(ctx, df):
    """GZH-labelled buffer galaxies far enough from every test and external galaxy to share no pixels or neighbours
    with them (supervised.train_on_buffer; buffer_min_separation_arcsec = 2 cutouts by default)."""
    from scipy.spatial import cKDTree
    from .utils import unit_vectors
    sup = ctx.cfg["supervised"]
    buf = df[~df.is_pseudo_label & (df.split_tag == "buffer") & ~df.edge_flag]
    if ctx.cfg["frame_qc"].get("exclude_flagged"):
        buf = buf[buf.frame_quality_flag == 1]
    held = df[df.split_tag.isin(["test", "external"])]
    if buf.empty or held.empty:
        return buf.assign(gz_label_id=buf.gz_label_id.astype(int))
    chord = cKDTree(unit_vectors(held.ra, held.dec)).query(unit_vectors(buf.ra, buf.dec))[0]
    far = np.degrees(2 * np.arcsin(chord / 2)) * 3600 >= float(sup.get("buffer_min_separation_arcsec", 12.8))
    buf = buf[far]
    return buf.assign(gz_label_id=buf.gz_label_id.astype(int))


def supervised_splits(ctx):
    """GZH-labelled galaxies per split. Truncated (edge_flag) sources are excluded,
    and QC-flagged ones too when frame_qc.exclude_flagged is true. supervised.train_on_buffer adds far-away
    buffer galaxies to train (they keep split_tag 'buffer')."""
    df = galaxy_table(ctx)
    lab = df[~df.is_pseudo_label & df.split_tag.isin(["train", "dev", "test"]) & ~df.edge_flag]
    if ctx.cfg["frame_qc"].get("exclude_flagged"):
        lab = lab[lab.frame_quality_flag == 1]
    lab = lab.assign(gz_label_id=lab.gz_label_id.astype(int))
    from .modes import is_apply
    if is_apply(ctx):
        # the frozen model never saw this field: EVERY labelled galaxy (any split, buffer too) is test data
        from .context import StageSkipped
        allg = df[~df.is_pseudo_label & ~df.edge_flag]
        if ctx.cfg["frame_qc"].get("exclude_flagged"):
            allg = allg[allg.frame_quality_flag == 1]
        if allg.empty:
            raise StageSkipped("apply mode: no GZH-labelled galaxies in this field, so no in-field evaluation "
                               "(every galaxy is still classified in Stage 8)")
        allg = allg.assign(gz_label_id=allg.gz_label_id.astype(int)).reset_index(drop=True)
        return {"train": allg.iloc[0:0], "dev": allg.iloc[0:0], "test": allg}
    splits = {s: lab[lab.split_tag == s].reset_index(drop=True) for s in ("train", "dev", "test")}
    if ctx.cfg["supervised"].get("train_on_buffer"):
        extra = _trainable_buffer(ctx, df)
        splits["train"] = pd.concat([splits["train"], extra], ignore_index=True)
    empty = [s for s, v in splits.items() if len(v) == 0]
    if empty:
        m = df[~df.is_pseudo_label]
        per = m.split_tag.fillna("collar").value_counts().to_dict()
        per_ok = m[~m.edge_flag].split_tag.fillna("collar").value_counts().to_dict()
        raise RuntimeError(
            f"No labelled galaxies in split(s) {empty}.\n"
            f"  GZH-matched galaxies per split: {per}\n"
            f"  ... after removing edge-truncated ones: {per_ok}\n"
            "  With one pointing, Stage 0 cuts it into stripes: look at outputs/stage0_partition/split_regions.png, "
            "lower partition.collar_arcmin, or add a second pointing.")
    return splits
