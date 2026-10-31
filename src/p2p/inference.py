"""The supervised model of the pipeline (feature-level fusion) and its frozen, calibrated inference path.

    galaxy cutout (128 x 128, 3 bands) -> cosmic rays cleaned, neighbours masked, cropped to ~4 Petrosian radii
        -> ImageNet input (R, G, B = F775W, F606W, F435W; asinh; 224 x 224); features averaged over 4 rotations
        -> ResNet-18 (ImageNet, frozen)        -> 512 features      \
        -> EfficientNet-B0 (ImageNet, frozen)  -> 1,280 features     >  ENSEMBLE (fusion): one 1,810-feature vector,
    + 18 physical features (Pass 2)            -> 18 features       /   block weights w1, w2, w3
        -> PCA of each CNN block (fitted inside each CV fold; 32 + 32 components) + 18 physical
        -> XGBoost (5 morphology classes)  +  XGBoost (merger yes / no)
    FINAL morphology model (Stage 4b): equal-weight average of supervised.ensemble_members (fine-tuned ResNet-18 +
    fine-tuned EfficientNet-B0 [p2p/finetune.py] + XGBoost on the 18 physical features + the fusion XGBoost above)
        -> calibration (Stage 4c)

Used by Stages 4a (feature extraction), 4b (training), 4c, 4d, 6a, 6c, 7b and 8.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from .context import Ctx, load_json
from .features import imagenet_input, load_cutouts, physical_matrix
from .nets import N_CLASSES, logit, softmax_np

CNN_BLOCKS = (("resnet18", 512), ("efficientnet_b0", 1280))
BLOCKS = ("resnet18", "efficientnet_b0", "physical")


# ================================================================ Stage 4a: frozen CNN features
def build_extractors(pretrained=True):
    """The two frozen ImageNet networks (no training: BatchNorm uses its ImageNet statistics, Dropout is off)."""
    from .nets import EfficientNetB0, ResNet18, frozen
    return {"resnet18": frozen(ResNet18(in_ch=3, pretrained=pretrained)),
            "efficientnet_b0": frozen(EfficientNetB0(pretrained=pretrained))}


def extract_cnn_features(ctx: Ctx, rows, nets=None, log=None):
    """Galaxies (DataFrame with cutout_index, r_petro, area_px) -> (n, 1792) float32: [ResNet-18 512 | EfficientNet-B0
    1280]. Each galaxy is cleaned, masked and cropped (features.cnn_stamp), stretched and resized to input_size, and
    the features are AVERAGED over supervised.cnn_tta_views rotations (0, 90, 180, 270 deg): the class of a galaxy
    does not depend on its orientation on the sky, so averaging removes orientation noise from the features."""
    import torch
    from .features import cnn_stamp
    from .modes import cnn_input_settings
    cfg = cnn_input_settings(ctx)             # apply mode: exactly the bundle's training settings
    size, bs = int(cfg.get("input_size", 224)), int(cfg.get("feature_batch_size", 64))
    views = max(1, min(4, int(cfg.get("cnn_tta_views", 4))))
    nets = nets or build_extractors(bool(cfg.get("pretrained", True)))
    dev = ctx.device()
    for m in nets.values():
        m.to(dev)
    cut = load_cutouts(ctx)
    seg = np.load(ctx.require("1c", "segstamps.npy"), mmap_mode="r")
    idx = np.asarray(rows["cutout_index"], int)
    rp = pd.to_numeric(rows.get("r_petro", pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(float)
    ar = pd.to_numeric(rows.get("area_px", pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(float)
    atlas = ctx.cfg.get("atlas", {})
    out = np.zeros((len(idx), sum(n for _, n in CNN_BLOCKS)), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            sl = np.arange(s, min(s + bs, len(idx)))
            stamps = [cnn_stamp(cut[idx[i]], seg[idx[i]], rp[i], ar[i], cfg, atlas) for i in sl]
            x = imagenet_input(stamps, size).to(dev)
            f = 0
            for k in range(views):
                xk = torch.rot90(x, k, dims=(2, 3)) if k else x
                f = f + torch.cat([nets[b](xk) for b, _ in CNN_BLOCKS], 1)
            out[sl] = (f / views).cpu().numpy()
            if log and (s // bs) % 25 == 0:
                log(f"  [features] {sl[-1] + 1:,} / {len(idx):,} galaxies ({time.time() - t0:.0f} s)")
    return out


def feature_names(blocks, physical):
    """Names of the model inputs, block by block (blocks: {name: (start, stop)} of the matrix the model sees)."""
    names = []
    for b, (a, z) in blocks.items():
        names += list(physical) if b == "physical" else [f"{b}_{'pc' if (z - a) < dict(CNN_BLOCKS)[b] else ''}{i}"
                                                         for i in range(z - a)]
    return names


def raw_blocks(n_physical):
    """Column ranges of the 1,810-feature fusion matrix (before PCA)."""
    return {"resnet18": (0, 512), "efficientnet_b0": (512, 1792), "physical": (1792, 1792 + n_physical)}


def block_slices(n_physical):
    return {b: slice(a, z) for b, (a, z) in raw_blocks(n_physical).items()}


def block_feature_weights(weights, blocks):
    """ENSEMBLE weights {block: w} -> one sampling weight per feature column: w_block / features in the block.
    XGBoost draws the columns of each tree WITHOUT replacement in proportion to these weights, so a block's share
    of the drawn columns follows w only while the block has more columns than its share: with the 1,792 raw CNN
    columns every physical feature was drawn for every tree whatever w3 (the first 8-pointing run); with the CNN
    blocks reduced by PCA (BlockPCA) the weights act as intended. (Multiplying the features by w would do nothing:
    a tree split is a threshold, unchanged by scaling a feature.)"""
    total = max(z for _, z in blocks.values())
    fw = np.zeros(total)
    for b, (a, z) in blocks.items():
        fw[a:z] = max(float(weights.get(b, 0.0)), 1e-6) / (z - a)
    return fw


class BlockPCA:
    """PCA of each CNN block (z-scored first), fitted on the TRAINING rows only - inside every CV fold and on all
    pooled galaxies for the final model. 512 + 1,280 correlated, mostly noisy activations become k + k components;
    physical features pass unchanged. numpy only (no sklearn), deterministic."""

    def __init__(self, k, blocks):
        self.k, self.blocks, self.p = int(k), dict(blocks), {}

    def fit(self, X):
        for b, (a, z) in self.blocks.items():
            if b in dict(CNN_BLOCKS) and 0 < self.k < (z - a):
                Z = np.asarray(X[:, a:z], np.float64)
                mu, sd = Z.mean(0), Z.std(0)
                sd[sd < 1e-8] = 1.0
                _, S, Vt = np.linalg.svd((Z - mu) / sd, full_matrices=False)
                k = min(self.k, Vt.shape[0])
                self.p[b] = (mu, sd, Vt[:k].T.copy(), float((S[:k] ** 2).sum() / (S ** 2).sum()))
        return self

    def transform(self, X):
        parts = []
        for b, (a, z) in self.blocks.items():
            Z = np.asarray(X[:, a:z], np.float64)
            if b in self.p:
                mu, sd, W, _ = self.p[b]
                Z = np.where(np.isfinite(Z), Z, mu)
                Z = ((Z - mu) / sd) @ W
            parts.append(Z)
        return np.hstack(parts).astype(np.float32)

    def out_blocks(self):
        res, s = {}, 0
        for b, (a, z) in self.blocks.items():
            n = self.p[b][2].shape[1] if b in self.p else z - a
            res[b] = (s, s + n)
            s += n
        return res

    def explained(self):
        return {b: round(v[3], 4) for b, v in self.p.items()}


def fusion_matrix(ctx: Ctx, df, physical):
    """The 1,810-feature vector of every row of df: CNN features from Stage 4a + the physical columns."""
    ids = np.load(ctx.require("4a", "object_id.npy"))
    F = np.load(ctx.require("4a", "cnn_features.npy"), mmap_mode="r")
    pos = {int(o): i for i, o in enumerate(ids)}
    rows = np.array([pos.get(int(o), -1) for o in df["object_id"]])
    cnn = np.zeros((len(df), F.shape[1]), np.float32)
    have = rows >= 0
    cnn[have] = F[rows[have]]
    if (~have).any():                                  # a galaxy Stage 4a did not see: extract it now
        cnn[~have] = extract_cnn_features(ctx, df[~have])
    return np.hstack([cnn, physical_matrix(df, physical)])


# ================================================================ XGBoost members (Stage 4b)
class XGBMember:
    """One XGBoost classifier (5 morphology classes, or merger yes / no) with balanced sample weights.
    Runs in a separate process (p2p.xgb_worker): next to PyTorch on macOS XGBoost can crash Python.
    Classes missing from the training data are mapped back to all output columns."""

    def __init__(self, n_classes, seed=0, params=None, feature_weights=None):
        self.n_classes, self.seed, self.params = int(n_classes), seed, dict(params or {})
        self.feature_weights = None if feature_weights is None else np.asarray(feature_weights, float)
        self.classes_, self.raw, self.n_estimators = None, None, None

    @staticmethod
    def _prepare(y, conf=None):
        """Balanced class weights x (optional) label confidence, renormalised so every class keeps the same total
        weight: confident labels count more WITHIN a class, the class balance is unchanged."""
        classes = np.unique(y)
        remap = {c: i for i, c in enumerate(classes)}
        yy = np.array([remap[c] for c in y])
        counts = np.bincount(yy)
        w = (len(yy) / (len(counts) * np.maximum(counts, 1)))[yy]          # balanced sample weights
        if conf is not None:
            c = np.clip(np.nan_to_num(np.asarray(conf, float), nan=1.0), 0.05, None)
            for k in range(len(counts)):
                m = yy == k
                c[m] = c[m] / c[m].mean()
            w = w * c
        return classes, remap, yy, w

    def cv_curve(self, X, y, Xva, yva, grid, conf=None):
        """Held-out probabilities (len(grid), n_va, n_classes) after each tree count in grid + history."""
        from . import xgb_worker
        classes, remap, yy, w = self._prepare(y, conf)
        yv = np.array([remap.get(c, -1) for c in yva])
        p, history = xgb_worker.run_isolated(xgb_worker.fit_curve, np.asarray(X, np.float32), yy, w,
                                             np.asarray(Xva, np.float32), yv, len(classes), list(grid), self.seed,
                                             self.params, self.feature_weights)
        out = np.zeros((len(grid), len(Xva), self.n_classes))
        out[:, :, classes] = p
        return out / out.sum(2, keepdims=True), history

    def fit(self, X, y, n_estimators, conf=None):
        from . import xgb_worker
        self.classes_, _, yy, w = self._prepare(y, conf)
        self.n_estimators = int(n_estimators)
        self.raw = xgb_worker.run_isolated(xgb_worker.fit_fixed, np.asarray(X, np.float32), yy, w, self.n_estimators,
                                           len(self.classes_), self.seed, self.params, self.feature_weights)
        return self

    def predict_proba(self, X):
        from . import xgb_worker
        out = np.zeros((len(X), self.n_classes))
        out[:, self.classes_] = xgb_worker.run_isolated(xgb_worker.predict, self.raw, np.asarray(X, np.float32))
        return out / out.sum(1, keepdims=True)

    def contribs(self, X):
        """TreeSHAP values (n, outputs, features)."""
        from . import xgb_worker
        return xgb_worker.run_isolated(xgb_worker.contribs, self.raw, np.asarray(X, np.float32))

    def summary(self, names):
        from . import xgb_worker
        return xgb_worker.run_isolated(xgb_worker.summary, self.raw, list(names), self.params)


def xgb_cv(X, y, folds, grid, n_classes, seed, params, weights=None, blocks=None, pca_k=0, conf=None):
    """5-fold CV of one XGBoost configuration: OOF probabilities after every tree count in grid
    (len(grid), n, n_classes) and the per-fold histories.
      blocks  : {block: (start, stop)} of X; weights : ensemble weights {block: w} -> XGBoost feature_weights
      pca_k   : >0 -> BlockPCA of the CNN blocks fitted on the TRAINING fold only (no information from the
                held-out fold enters the features), applied to both
      conf    : label confidence per row (GZH vote fraction) -> sample weights"""
    oof = np.zeros((len(grid), len(y), n_classes))
    hist = []
    for tr, va in folds:
        Xtr, Xva, ob = X[tr], X[va], blocks
        if pca_k and blocks:
            red = BlockPCA(pca_k, blocks).fit(Xtr)
            Xtr, Xva, ob = red.transform(Xtr), red.transform(Xva), red.out_blocks()
        fw = block_feature_weights(weights, ob) if (weights and ob) else None
        m = XGBMember(n_classes, seed, params, fw)
        oof[:, va], h = m.cv_curve(Xtr, y[tr], Xva, y[va], grid, None if conf is None else conf[tr])
        hist.append(h)
    return oof, hist


# ================================================================ frozen inference path
def load_fusion(ctx: Ctx):
    """The trained Stage 4b models + Stage 4c calibration. The fine-tuned CNNs are loaded only when a galaxy is
    missing from finetuned_probs.npz (e.g. apply mode on a new field)."""
    def opt(name):
        p = ctx.path("4b", name)
        return joblib.load(p) if p.exists() else None
    meta = load_json(ctx.require("4b", "fusion_meta.json"))
    m = {"meta": meta, "morph": joblib.load(ctx.require("4b", "xgb_morphology.joblib")),
         "merger": opt("xgb_merger.joblib"), "pca": opt("cnn_pca.joblib"), "physical": opt("xgb_physical.joblib"),
         "members": list(meta.get("ensemble_members") or ["xgb_fusion"])}
    p = ctx.path("4c", "temperature.json")
    m["calibration"] = load_json(p) if p.exists() else None
    return m


def member_probabilities(ctx: Ctx, df, models, member, X=None):
    """Class probabilities (n, 5) of one ensemble member for the galaxies in df."""
    if member == "xgb_fusion":
        return models["morph"].predict_proba(X)
    if member == "xgb_physical":
        if models.get("physical") is None:
            raise RuntimeError("ensemble member xgb_physical: outputs/stage4b_xgboost_fusion/xgb_physical.joblib missing")
        return models["physical"].predict_proba(physical_matrix(df, models["meta"]["physical_columns"]))
    from . import finetune
    kind = {v: k for k, v in finetune.MEMBER.items()}[member]
    P = np.full((len(df), N_CLASSES), np.nan)
    pf = ctx.path("4b", "finetuned_probs.npz")
    if pf.exists():
        d = np.load(pf)
        if member in d.files:
            pos = {int(o): i for i, o in enumerate(d["object_id"])}
            rows = np.array([pos.get(int(o), -1) for o in df["object_id"]])
            P[rows >= 0] = d[member][rows[rows >= 0]]
    miss = ~np.isfinite(P).all(1)
    if miss.any():                                    # galaxies not predicted in Stage 4b: run the network now
        key = f"_net_{kind}"
        if key not in models:
            models[key] = finetune.load(ctx.require("4b", f"finetuned_{kind}.pt"), ctx.device())
        net, cfg = models[key]
        P[miss] = finetune.predict_rows(ctx, net, cfg, df[miss], ctx.device())
    return P


def combine_members(parts, weights=None):
    """Average of the members' class probabilities: equal weights, or {member: weight} (supervised.ensemble_member_weights)."""
    if not weights:
        return np.mean(list(parts.values()), axis=0)
    missing = sorted(set(parts) - set(weights))
    if missing:
        raise ValueError(f"supervised.ensemble_member_weights has no weight for {missing}")
    total = sum(float(weights[m]) for m in parts)
    return sum(float(weights[m]) / total * p for m, p in parts.items())


def predict_galaxies(ctx: Ctx, df, models=None, calibrated=True):
    """Class probabilities (n, 5) and merger probability (n,) of the galaxies in df (rows of galaxy_table):
    the average of the Stage 4b ensemble members (equal weights unless configured), then the Stage 4c calibration.
    X = the matrix the fusion XGBoost saw (after the CNN PCA), columns as meta["feature_names"] / meta["blocks"]."""
    models = models or load_fusion(ctx)
    X = fusion_matrix(ctx, df, models["meta"]["physical_columns"])
    if models.get("pca") is not None:                  # the CNN blocks -> their principal components (Stage 4b)
        X = models["pca"].transform(X)
    parts = {m: member_probabilities(ctx, df, models, m, X) for m in models["members"]}
    P = combine_members(parts, models["meta"].get("ensemble_member_weights"))
    pm = models["merger"].predict_proba(X)[:, 1] if models["merger"] is not None else \
        np.full(len(df), models["meta"].get("merger_base_rate", 0.0))
    cal = models["calibration"]
    if calibrated and cal:
        P, pm = apply_calibration(P, pm, cal)
    return {"morph": P, "merger": pm, "X": X, "members": parts}


def apply_calibration(P, pm, cal):
    """Stage 4c: temperature on the class probabilities, Platt scaling on the merger probability."""
    P = softmax_np(np.log(np.clip(P, 1e-12, 1)) / cal["T_morph"])
    pm = 1 / (1 + np.exp(-(cal.get("merger_platt_a", 1.0) * logit(pm) + cal.get("merger_platt_b", 0.0))))
    return P, pm


def oof_predictions(ctx: Ctx):
    """Calibrated out-of-fold predictions of the pooled train + dev galaxies (each from a model that never saw that
    galaxy) as {object_id, morph, merger}; None if Stage 4b left none (apply mode)."""
    f, c = ctx.path("4b", "oof_fusion.npz"), ctx.path("4c", "temperature.json")
    if not f.exists() or not c.exists():
        return None
    d = np.load(f)
    P, pm = apply_calibration(d["morph"], d["merger"], load_json(c))
    return {"object_id": d["object_id"], "morph": P, "merger": pm}


__all__ = ["BLOCKS", "BlockPCA", "CNN_BLOCKS", "N_CLASSES", "XGBMember", "apply_calibration", "block_feature_weights",
           "block_slices", "build_extractors", "combine_members", "extract_cnn_features", "feature_names", "fusion_matrix", "load_fusion",
           "member_probabilities", "oof_predictions", "predict_galaxies", "raw_blocks", "xgb_cv"]
