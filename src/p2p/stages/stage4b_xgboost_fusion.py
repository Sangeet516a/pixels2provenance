"""Stage 4b - the supervised ensemble: feature-level fusion XGBoost, physical XGBoost and fine-tuned CNNs.

Final model (supervised.ensemble_members: resnet18_ft + efficientnet_b0_ft + xgb_physical + xgb_fusion): the EQUAL-WEIGHT
average of the members' class probabilities (no weight is fitted, so nothing is tuned on the held-out folds),
calibrated in Stage 4c. Members, all cross-validated on the same 5 folds:
  xgb_fusion        : XGBoost on frozen CNN features (PCA) + 18 physical features (below)
  xgb_physical      : XGBoost on the 18 physical features alone
  resnet18_ft       : ResNet-18, ImageNet weights, last two stages fine-tuned on the galaxies (p2p/finetune.py)
  efficientnet_b0_ft: EfficientNet-B0, likewise
Why (30 Oct, same galaxies and folds): frozen fusion 0.466 Macro-F1; average of the fine-tuned CNNs and the physical
XGBoost 0.507 - each fine-tuned CNN alone (0.45 / 0.46) matched the whole frozen fusion, and the image and physics
members make different mistakes, which an average exploits.

Feature-level fusion XGBoost (member xgb_fusion; also the source of the merger model and the SHAP of Stage 6c):

Ensemble: every labelled galaxy becomes ONE vector of 1,810 features =
              ResNet-18 (512) + EfficientNet-B0 (1,280)   [Stage 4a, frozen ImageNet CNNs]
            + 18 physical features (Gini, M20, concentration, asymmetry, colours, ... ; Pass 2, Stage 5b).
          PCA: each CNN block is reduced to supervised.cnn_pca_components (32) principal components, fitted on the
          training fold only inside every CV fold (and on all pooled galaxies for the final model, cnn_pca.joblib):
          ~3,200 labelled galaxies cannot support 1,792 noisy, correlated CNN columns, and with the reduced blocks
          the ensemble weights really decide which block each tree draws from.
          Ensemble weights w1 (ResNet-18), w2 (EfficientNet-B0), w3 (physical): the share of the features each
          tree may draw from each block (XGBoost feature_weights; scaling the features themselves would not
          change a tree). The candidate sets in supervised.fusion_block_weights are compared by 5-fold CV and the
          set with the lowest out-of-fold log-loss is used (weight_search.png).
XGBoost : "compile" in supervised.xgboost: softmax over the 5 GZH classes (multi:softprob), shallow trees
          (max_depth 4), learning rate 0.05, 80 % of the galaxies and 30 % of the features per tree,
          L2 penalty reg_lambda on the leaf values, balanced class weights x GZH label confidence (vote fraction,
          renormalised within each class: clear labels count more, the class balance is unchanged).
          supervised.xgb_grid (max_depth x colsample_bytree) is searched together with the ensemble weights.
Fit     : 5-fold cross-validation on the pooled TRAIN + DEV galaxies (grouped by 30" sky cell); the number of
          trees is chosen by the lowest out-of-fold log-loss; the final model is trained on all pooled galaxies.
          Keras-style history per boosting round (loss / val_loss / accuracy / val_accuracy).
Merger  : a second XGBoost (merger yes / no, logistic) on the same 1,810 features and ensemble weights.
Output  : xgb_morphology.joblib (fusion), xgb_physical.joblib, finetuned_<network>.pt, finetuned_probs.npz,
          xgb_merger.joblib, fusion_meta.json, model_summary.txt, weight_search.png / .csv,
          training_history_xgboost.png / .json, training_history_<network>.png / .json, ensemble_members.csv,
          cv_curve.png, cv_folds.csv, cv_metrics.json, oof_fusion.npz (final ensemble), oof_members.npz,
          dev_metrics.json
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ..context import Ctx, save_json
from ..features import supervised_splits
from ..inference import (BLOCKS, BlockPCA, XGBMember, block_feature_weights, combine_members, feature_names,
                         fusion_matrix, raw_blocks, xgb_cv)
from ..nets import MORPH_CLASSES, morph_metrics, nll
from ..supervised_cv import (choose_epoch, curve_metrics, detailed_scores, detailed_text, make_folds, mean_history,
                             plot_curve, plot_history, pooled, summarise)


def param_grid(cfg, params):
    """supervised.xgb_grid {param: [values]} -> list of XGBoost parameter dicts (the base 'compile' params + one combo)."""
    import itertools
    g = cfg.get("xgb_grid") or {}
    keys = list(g)
    return [dict(params, **dict(zip(keys, v))) for v in itertools.product(*[g[k] for k in keys])] or [dict(params)]


def tree_grid(cfg):
    return [10] + list(range(25, int(cfg.get("xgb_max_trees", 400)) + 1, 25))


def run(ctx: Ctx):
    from ..modes import install_bundle, is_apply
    if is_apply(ctx):                       # no training: copy the frozen model bundle
        return install_bundle(ctx, "4b")
    cfg = ctx.cfg["supervised"]
    out = ctx.stage_dir("4b")
    params = dict(cfg.get("xgboost") or {})
    metric = str(cfg.get("selection_metric", "log_loss"))
    grid = tree_grid(cfg)

    # ---- data: GZH-labelled galaxies of train + dev, pooled for 5-fold cross-validation ----
    splits = supervised_splits(ctx)
    pool = pooled(splits)
    folds, fold_id, groups = make_folds(pool, int(cfg.get("cv_folds", 5)), ctx.seed,
                                        float(cfg.get("cv_group_arcsec", 30.0)), float(cfg.get("cv_link_arcsec", 6.4)))
    pool.assign(fold=fold_id, sky_cell=groups)[["object_id", "orig_split", "fold", "sky_cell", "gz_label"]] \
        .to_csv(out / "cv_folds.csv", index=False)
    physical = [c for c in ctx.cfg["representation"]["tabular_features"] if c in pool.columns]
    # supervised.extra_physical_features: for the classifiers only, so the clustering of Stages 2a-2b does not change
    extra = list(cfg.get("extra_physical_features") or [])
    physical += [c for c in extra if c in pool.columns and c not in physical]
    if set(extra) - set(physical):
        ctx.log.info(f"[Stage 4b] WARNING: extra_physical_features not measured: {sorted(set(extra) - set(physical))}")
    X = fusion_matrix(ctx, pool, physical)
    blocks = raw_blocks(len(physical))
    pca_k = int(cfg.get("cnn_pca_components", 0) or 0)
    conf = pool.gz_vote_frac.to_numpy(float) if cfg.get("label_confidence_weights", False) else None
    y = pool.gz_label_id.astype(int).to_numpy()
    ym = pool.gz_merger_flag.fillna(0).astype(int).to_numpy()
    counts = pd.crosstab(pool.orig_split, pool.gz_label)
    ctx.log.info(f"[Stage 4b] ENSEMBLE input: {X.shape[0]} galaxies x {X.shape[1]} features "
                 f"(ResNet-18 512 + EfficientNet-B0 1,280 + physical {len(physical)})\n{counts.to_string()}\n"
                 f"[Stage 4b] CNN blocks -> {pca_k or 'all'} principal components each (fitted inside every CV fold) | "
                 f"sample weights: balanced classes{' x GZH label confidence' if conf is not None else ''}\n"
                 f"[Stage 4b] compile: XGBClassifier(objective multi:softprob, {params}, n_estimators <= {max(grid)}) "
                 f"| grid {cfg.get('xgb_grid') or '-'} | metrics: log-loss, accuracy, Macro-F1 | {len(folds)}-fold CV "
                 f"(folds of {[len(v) for _, v in folds]}, grouped by {cfg.get('cv_group_arcsec', 30)}\" sky cells)")

    # ---- 1. ENSEMBLE weights x XGBoost grid: each candidate cross-validated, lowest OOF log-loss wins ----
    rows, runs = [], {}
    combos = param_grid(cfg, params)
    for w in cfg.get("fusion_block_weights") or [[1 / 3, 1 / 3, 1 / 3]]:
        w = np.asarray(w, float) / np.sum(w)
        wd = dict(zip(BLOCKS, w))
        for pi, p in enumerate(combos):
            oof, hist = xgb_cv(X, y, folds, grid, len(MORPH_CLASSES), ctx.seed, p, wd, blocks, pca_k, conf)
            cur = curve_metrics(y, oof, folds)
            k = choose_epoch(cur, 1, metric)[0] - 1
            key = tuple(float(v) for v in np.round(w, 3)) + (pi,)
            runs[key] = {"w": w, "params": p, "oof": oof, "hist": hist, "curve": cur, "k": k}
            rows.append({"w_resnet18": key[0], "w_efficientnet_b0": key[1], "w_physical": key[2], "combo": pi,
                         "max_depth": p.get("max_depth"), "colsample_bytree": p.get("colsample_bytree"),
                         "trees": grid[k], "oof_log_loss": cur[k]["nll"], "oof_macro_f1": cur[k]["macro_f1"],
                         "oof_accuracy": cur[k]["accuracy"]})
            ctx.log.info(f"  [ensemble] w = {list(key[:3])}, max_depth {p.get('max_depth')}, colsample_bytree "
                         f"{p.get('colsample_bytree')}: best {grid[k]} trees, OOF log-loss {cur[k]['nll']:.4f}, "
                         f"Macro-F1 {cur[k]['macro_f1']:.4f}, accuracy {cur[k]['accuracy']:.4f}")
    search = pd.DataFrame(rows)
    search.to_csv(out / "weight_search.csv", index=False)
    best_row = search.loc[search.oof_log_loss.idxmin() if metric != "macro_f1" else search.oof_macro_f1.idxmax()]
    best = runs[(best_row.w_resnet18, best_row.w_efficientnet_b0, best_row.w_physical, int(best_row.combo))]
    w, params, k, n_trees = best["w"], best["params"], best["k"], grid[best["k"]]
    wd = dict(zip(BLOCKS, w))
    ctx.log.info(f"[Stage 4b] ENSEMBLE weights chosen: w1 (ResNet-18) = {w[0]:.3f}, w2 (EfficientNet-B0) = "
                 f"{w[1]:.3f}, w3 (physical) = {w[2]:.3f}; max_depth {params.get('max_depth')}, colsample_bytree "
                 f"{params.get('colsample_bytree')} -> {n_trees} trees per class")
    _plot_search(search, best_row, out)

    # ---- 2. history of the chosen configuration (Keras-style, per boosting round) ----
    history = mean_history({key: [h[key] for h in best["hist"]] for key in best["hist"][0]})
    plot_history(history, "XGBoost (fusion)", out / "training_history_xgboost.png",
                 xlabel="Boosting rounds (trees per class)", chosen=n_trees)
    save_json(dict(history, chosen_trees=n_trees), out / "training_history_xgboost.json")
    r = sorted({i for i in (9, 49, 99, 199, 299, max(grid) - 1, n_trees - 1) if i < len(history["loss"])})
    ctx.log.info("[Stage 4b] XGBoost history (mean over the CV folds; val = held-out fold):\n" + "\n".join(
        f"  round {i + 1:3d}  loss {history['loss'][i]:.4f}  accuracy {history['accuracy'][i]:.4f}  "
        f"val_loss {history['val_loss'][i]:.4f}  val_accuracy {history['val_accuracy'][i]:.4f}" for i in r))
    plot_curve({"XGBoost fusion": best["curve"]}, {"XGBoost fusion": k + 1}, out / "cv_curve.png",
               f"Stage 4b XGBoost fusion: out-of-fold Macro-F1 per tree count index (grid {grid[0]}, {grid[1]}, ...)")
    members_oof = {"xgb_fusion": best["oof"][k]}

    # ---- 2b. physical-feature XGBoost and fine-tuned CNNs (the other ensemble members) ----
    Xp = X[:, blocks["physical"][0]:blocks["physical"][1]]
    oof_p, _ = xgb_cv(Xp, y, folds, grid, len(MORPH_CLASSES), ctx.seed, params, conf=conf)
    cur_p = curve_metrics(y, oof_p, folds)
    kp = choose_epoch(cur_p, 1, metric)[0] - 1
    members_oof["xgb_physical"] = oof_p[kp]
    model_phys = XGBMember(len(MORPH_CLASSES), ctx.seed, params).fit(Xp, y, grid[kp], conf)
    joblib.dump(model_phys, out / "xgb_physical.joblib")
    ctx.log.info(f"[Stage 4b] physical XGBoost: {grid[kp]} trees, OOF Macro-F1 {cur_p[kp]['macro_f1']:.4f}, "
                 f"log-loss {cur_p[kp]['nll']:.4f}")
    want = list(cfg.get("ensemble_members") or ["xgb_fusion"])
    ft = _finetune_members(ctx, out, pool, y, folds, want, members_oof)
    members = [m for m in want if m in members_oof]
    if not members:
        members = ["xgb_fusion"]
    if len(members) < len(want):
        ctx.log.info(f"[Stage 4b] WARNING: ensemble members {sorted(set(want) - set(members))} not available "
                     f"(supervised.finetune.enabled?) - the ensemble uses {members}")
    weights = cfg.get("ensemble_member_weights") or None
    P = combine_members({m: members_oof[m] for m in members}, weights)
    comp = _member_table(y, members_oof, members, folds)
    if weights:                  # the configured weights next to the equal-weight rows
        s = summarise(y, P, folds)
        comp.loc[comp.final_model, "final_model"] = False
        row = {"members": " + ".join(f"{float(weights[m]):g} {m}" for m in members), "final_model": True,
               "macro_f1": s["macro_f1"], "ci_low": s["macro_f1_95ci"][0], "ci_high": s["macro_f1_95ci"][1],
               "accuracy": s["accuracy"], "log_loss": s["nll"], "fold_sd": s["fold_sd"]}
        comp = pd.concat([comp, pd.DataFrame([row])], ignore_index=True).sort_values("macro_f1", ascending=False)
    comp.to_csv(out / "ensemble_members.csv", index=False)
    ctx.log.info("[Stage 4b] members and combinations (out-of-fold, same folds; equal-weight averages):\n"
                 + comp.to_string(index=False))
    res = summarise(y, P, folds)
    det = detailed_scores(y, P, pool.gz_vote_frac.to_numpy(float))
    res.update(det)
    dev = (pool.orig_split == "dev").to_numpy()
    if dev.any():
        res["dev_only"] = morph_metrics(y[dev], P[dev])
    from_buffer = (pool.split_tag == "buffer").to_numpy()
    if from_buffer.any():        # the score to compare with a run without them: the same galaxies
        res["without_buffer"] = morph_metrics(y[~from_buffer], P[~from_buffer])
        ctx.log.info(f"[Stage 4b] {int(from_buffer.sum())} buffer galaxies in training (supervised.train_on_buffer); "
                     f"out-of-fold score without them: Macro-F1 {res['without_buffer']['macro_f1']}, accuracy "
                     f"{res['without_buffer']['accuracy']} (n = {res['without_buffer']['n']})")
    ctx.log.info(f"[Stage 4b] FINAL MODEL = {'weighted' if weights else 'equal-weight'} average of {members}"
                 + (f", weights {weights}" if weights else ""))
    ctx.log.info("[Stage 4b] out-of-fold scores in detail:\n" + detailed_text(det)
                 + (f"\n  dev galaxies only: Macro-F1 {res['dev_only']['macro_f1']}, accuracy {res['dev_only']['accuracy']} "
                    f"(n = {res['dev_only']['n']})" if dev.any() else ""))

    # ---- 3. merger: second XGBoost (yes / no) on the same features and ensemble weights ----
    merger_trees, pm = None, np.full(len(y), float(ym.mean()))
    if 0 < ym.sum() < len(ym):
        oof_m, _ = xgb_cv(X, ym, folds, grid, 2, ctx.seed, params, wd, blocks, pca_k)
        lls = [nll(ym, o) for o in oof_m]
        km = int(np.argmin(lls))
        merger_trees, pm = grid[km], oof_m[km][:, 1]
        res.update({k_: v for k_, v in morph_metrics(y, P, ym, pm).items() if k_.startswith("merger")})
        ctx.log.info(f"[Stage 4b] merger XGBoost: {int(ym.sum())} mergers of {len(ym)}; {merger_trees} trees "
                     f"(OOF log-loss {lls[km]:.4f}); OOF merger PR-AUC {res.get('merger_pr_auc')} "
                     f"(prevalence {res.get('merger_prevalence')})")

    # ---- 4. final models: all pooled galaxies (PCA refitted on all of them, saved for inference) ----
    reducer, Xf, ob = None, X, blocks
    if pca_k:
        reducer = BlockPCA(pca_k, blocks).fit(X)
        Xf, ob = reducer.transform(X), reducer.out_blocks()
        joblib.dump(reducer, out / "cnn_pca.joblib")
        ctx.log.info(f"[Stage 4b] CNN PCA (all pooled galaxies): {ob} | explained variance {reducer.explained()}")
    else:
        (out / "cnn_pca.joblib").unlink(missing_ok=True)
    fw = block_feature_weights(wd, ob)
    names = feature_names(ob, physical)
    model_xgb = XGBMember(len(MORPH_CLASSES), ctx.seed, params, fw).fit(Xf, y, n_trees, conf)
    joblib.dump(model_xgb, out / "xgb_morphology.joblib")
    if merger_trees:
        model_merger = XGBMember(2, ctx.seed, params, fw).fit(Xf, ym, merger_trees)
        joblib.dump(model_merger, out / "xgb_merger.joblib")
    else:
        (out / "xgb_merger.joblib").unlink(missing_ok=True)

    # ---- 5. summary (trees, parameters, which block the model uses) ----
    s = model_xgb.summary(names)
    shares = _block_shares(s["gain_importance"], physical)
    txt = _summary_text(s, shares, w, physical, ob)
    (out / "model_summary.txt").write_text(txt + "\n")
    ctx.log.info("[Stage 4b] model summary:\n" + txt)

    meta = {"physical_columns": physical, "n_features": int(Xf.shape[1]), "n_features_raw": int(X.shape[1]),
            "blocks": {b: list(v) for b, v in ob.items()}, "raw_blocks": {b: list(v) for b, v in blocks.items()},
            "feature_names": names, "cnn_pca_components": pca_k,
            "cnn_pca_explained_variance": reducer.explained() if reducer else None,
            "label_confidence_weights": conf is not None,
            "cnn_input": {k_: cfg.get(k_) for k_ in ("cnn_crop", "cnn_crop_petro", "cnn_crop_min", "cnn_mask_neighbours",
                                                     "cnn_clean_cosmic_rays", "cnn_tta_views")},
            "ensemble_weights": {b: round(float(v), 4) for b, v in zip(BLOCKS, w)}, "trees": n_trees,
            "merger_trees": merger_trees, "merger_base_rate": round(float(ym.mean()), 4),
            "input_size": int(cfg.get("input_size", 224)), "xgboost": params, "classes": list(MORPH_CLASSES),
            "n_pool": int(len(pool)), "selection": f"5-fold CV, minimum out-of-fold {metric}",
            "gain_share_by_block": shares, "ensemble_members": members, "ensemble_member_weights": weights,
            "xgb_physical_trees": int(grid[kp]),
            "finetune": ft, "member_scores": comp.to_dict(orient="records")}
    save_json(meta, out / "fusion_meta.json")
    np.savez(out / "oof_fusion.npz", object_id=pool.object_id.to_numpy(), y=y, ym=ym, morph=P, merger=pm, fold=fold_id)
    np.savez(out / "oof_members.npz", object_id=pool.object_id.to_numpy(), y=y, **members_oof)
    save_json({"summary": res, "curve": best["curve"], "weight_search": rows, "chosen_params": params},
              out / "cv_metrics.json")
    save_json(dict(res, note="out-of-fold (cross-validated) metrics on the pooled train + dev galaxies"),
              out / "dev_metrics.json")
    ctx.log.info(f"[Stage 4b] out-of-fold result: accuracy {res['accuracy']} | Macro-F1 {res['macro_f1']} "
                 f"(95% CI {res['macro_f1_95ci']}) | balanced accuracy {res['balanced_acc']} | log-loss {res['nll']} "
                 f"| per fold {res['fold_macro_f1']} | importance by block {shares}")
    return {"oof_macro_f1": res["macro_f1"], "oof_accuracy": res["accuracy"], "trees": n_trees,
            "ensemble_members": members, "ensemble_weights": meta["ensemble_weights"]}


def _finetune_members(ctx, out, pool, y, folds, want, members_oof):
    """Fine-tuned ResNet-18 / EfficientNet-B0: 5-fold CV (out-of-fold probabilities + Keras-style history), then the
    final network on all pooled galaxies, saved, and its probabilities for EVERY galaxy (finetuned_probs.npz)."""
    from .. import finetune
    from ..features import galaxy_table
    cfg = finetune.ft_config(ctx)
    kinds = [k for k in cfg["networks"] if finetune.MEMBER[k] in want] if cfg.get("enabled", True) else []
    for k in finetune.KINDS:
        if k not in kinds:
            (out / f"finetuned_{k}.pt").unlink(missing_ok=True)
    if not kinds:
        (out / "finetuned_probs.npz").unlink(missing_ok=True)
        return {"enabled": False}
    device = ctx.device()
    ctx.log.info(f"[Stage 4b] fine-tuning {kinds} on {len(y)} galaxies ({device}): input {cfg['input_size']} px, "
                 f"{cfg['epochs']} epochs, AdamW lr {cfg['lr']} (one-cycle), weight decay {cfg['weight_decay']}, "
                 f"batch {cfg['batch_size']}, label smoothing {cfg['label_smoothing']}, dropout {cfg['dropout']}, "
                 f"trained stages: {cfg['trainable']}, {cfg['tta_views']} rotations at prediction")
    Ximg = finetune.stamps(ctx, pool, cfg["input_size"])
    soft = finetune.vote_targets(ctx, pool) if cfg.get("soft_labels") else None
    if soft is not None:
        ctx.log.info(f"[Stage 4b] CNN targets: GZH vote fractions (mean per class {soft.mean(0).round(3).tolist()})")
    gal = galaxy_table(ctx)
    probs, info = {"object_id": gal.object_id.to_numpy()}, {"enabled": True, "config": cfg, "networks": {}}
    for kind in kinds:
        oof, hists = finetune.cross_validate(kind, Ximg, y, folds, cfg, device, ctx.seed, log=ctx.log.info, soft=soft)
        members_oof[finetune.MEMBER[kind]] = oof[-1]
        history = {key: np.mean([h[key] for h in hists], axis=0).tolist() for key in hists[0] if hists[0][key]}
        if all(history.get(k_) for k_ in ("loss", "val_loss", "accuracy", "val_accuracy")):
            plot_history(history, f"{kind} (fine-tuned)", out / f"training_history_{kind}.png", xlabel="Epochs",
                         chosen=int(cfg["epochs"]))
        save_json(history, out / f"training_history_{kind}.json")
        m = morph_metrics(y, oof[-1])
        ctx.log.info(f"[Stage 4b] {kind} fine-tuned: OOF Macro-F1 {m['macro_f1']}, accuracy {m['accuracy']}, "
                     f"log-loss {nll(y, oof[-1]):.4f}")
        model, _, _ = finetune.train(kind, Ximg, y, cfg, device, ctx.seed, log=ctx.log.info, tag=" final", soft=soft)
        finetune.save(model, out / f"finetuned_{kind}.pt", cfg)
        probs[finetune.MEMBER[kind]] = finetune.predict_rows(ctx, model, cfg, gal, device, log=ctx.log.info)
        info["networks"][kind] = {"oof_macro_f1": m["macro_f1"], "oof_accuracy": m["accuracy"],
                                  "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                                  "total_params": int(sum(p.numel() for p in model.parameters()))}
        del model
    np.savez(out / "finetuned_probs.npz", **probs)
    return info


def _member_table(y, oof, members, folds):
    """Every member alone and the main equal-weight combinations, out-of-fold on the same folds."""
    import itertools
    names = list(oof)
    combos = [(n,) for n in names] + [c for r in (2, 3, 4) for c in itertools.combinations(names, r)]
    rows = []
    for c in combos:
        P = np.mean([oof[n] for n in c], axis=0)
        s = summarise(y, P, folds)
        rows.append({"members": " + ".join(c), "final_model": set(c) == set(members), "macro_f1": s["macro_f1"],
                     "ci_low": s["macro_f1_95ci"][0], "ci_high": s["macro_f1_95ci"][1], "accuracy": s["accuracy"],
                     "log_loss": s["nll"], "fold_sd": s["fold_sd"]})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def _block_shares(gain, physical):
    tot = sum(gain.values()) or 1.0
    share = {b: 0.0 for b in BLOCKS}
    for name, v in gain.items():
        b = "resnet18" if name.startswith("resnet18_") else "efficientnet_b0" if name.startswith("efficientnet_b0_") \
            else "physical"
        share[b] += v / tot
    return {b: round(v, 4) for b, v in share.items()}


def _summary_text(s, shares, w, physical, ob):
    line = "=" * 66
    rows = [line, 'Model: "Stage 4b - XGBoost (feature fusion)"', line,
            "input            " + " + ".join(f"{b} {z - a}" for b, (a, z) in ob.items())
            + f" = {max(z for _, z in ob.values())} features (raw 1,792 CNN columns -> PCA per block)",
            f"ensemble weights w1 ResNet-18 {w[0]:.3f} | w2 EfficientNet-B0 {w[1]:.3f} | w3 physical {w[2]:.3f}",
            f"objective        {s['objective']} ({s['outputs']} classes)",
            f"boosting rounds  {s['boosting_rounds']} (x {s['outputs']} classes = {s['n_trees']} trees)",
            f"leaves           {s['leaves_total']:,} in total, {s['leaves_per_tree_mean']} per tree",
            "parameters       " + ", ".join(f"{k}={v}" for k, v in s["params"].items()), "-" * 66,
            "importance (total gain) by block: " + ", ".join(f"{b} {v:.1%}" for b, v in shares.items()),
            "top 15 features:"]
    tot = sum(s["gain_importance"].values()) or 1.0
    rows += [f"  {k:<22s} {v / tot:6.2%}" for k, v in list(s["gain_importance"].items())[:15]]
    rows.append(line)
    return "\n".join(rows)


def _plot_search(search, best_row, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = [f"{r.w_resnet18:.2f} / {r.w_efficientnet_b0:.2f} / {r.w_physical:.2f}  d{r.max_depth} c{r.colsample_bytree}"
           for r in search.itertuples()]
    col = ["#eb6834" if (r.w_resnet18, r.w_efficientnet_b0, r.w_physical, r.combo) ==
           (best_row.w_resnet18, best_row.w_efficientnet_b0, best_row.w_physical, best_row.combo) else "#2a78d6"
           for r in search.itertuples()]
    fig, axs = plt.subplots(1, 2, figsize=(13, max(4.5, 0.28 * len(lab) + 1.2)))
    axs[0].barh(lab, search.oof_log_loss, color=col)
    axs[0].set_xlim(search.oof_log_loss.min() * 0.97, search.oof_log_loss.max() * 1.01)
    axs[0].set_xlabel("out-of-fold log-loss (lower = better)")
    axs[1].barh(lab, search.oof_macro_f1, color=col)
    axs[1].set_xlabel("out-of-fold Macro-F1")
    for ax in axs:
        ax.set_ylabel("w1 ResNet-18 / w2 EfficientNet-B0 / w3 physical, max_depth, colsample_bytree")
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Stage 4b: ENSEMBLE weight search (5-fold CV; orange = chosen)")
    fig.tight_layout()
    fig.savefig(out / "weight_search.png", dpi=110, facecolor="white")
    plt.close(fig)
