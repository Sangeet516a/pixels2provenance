"""Stage 7b - Cross-catalogue and external evaluation.

Purpose : how well does the frozen model transfer to an independent expert label scheme, and to new sky?
Method  : (1) CANDELS: every Phase-7 galaxy (all scored splits, Stage 7a) with a CANDELS label and NO GZH label is
              predicted by the frozen, calibrated model (Stage 4b ensemble + 4c calibration); these galaxies never
              had a training label, so none of them was learned from, whatever split its sky region is in. The
              prediction is mapped to the shared taxonomy and scored with Macro-F1 (+ 95 % bootstrap interval),
              overall and per split. Galaxies in the GZH overlap (used for the ceiling) and expert point sources
              are excluded.
          (2) External GZH score - only when evaluation.seal_external is false: the GZH-labelled galaxies of the
              external pointings (GOODS-N-7 and the external stripes of GOODS-S, GOODS-N-2, GOODS-N-4) are scored like the Stage 6a test
              (Macro-F1 + 95 % CI, accuracy, balanced accuracy, ECE, per class, per pointing): same labels, new sky.
Output  : crosscat_predictions.csv, cross_catalogue_metrics.json; external_gzh_predictions.csv,
          external_gzh_metrics.json, confusion_matrix_external.png (when unsealed)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from ..context import Ctx, StageSkipped, save_json
from ..features import galaxy_table
from ..inference import predict_galaxies
from ..nets import MORPH_CLASSES, N_CLASSES, morph_metrics
from ..supervised_cv import shape_scores, shape_text
from ..utils import bootstrap_ci, expected_calibration_error
from .stage7a_taxonomy import COMMON, GZH_TO_COMMON


def run(ctx: Ctx):
    from ..sealing import external_sealed
    out = ctx.stage_dir("7b")
    summary = {}
    if external_sealed(ctx):
        ctx.log.info("[Stage 7b] external galaxies SEALED (evaluation.seal_external: true) - their GZH labels are not "
                     "scored; the CANDELS test below uses train, dev, test and buffer galaxies")
    else:
        gzh = _external_gzh(ctx, out)
        if gzh:
            summary["external_gzh_macro_f1"] = gzh["macro_f1"]
    p = ctx.path("7a", "candels_objects.csv")
    c = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["in_gzh_overlap", "k15_common", "object_id"])
    c["in_gzh_overlap"] = c.in_gzh_overlap.astype(str).str.lower().eq("true")
    test = c[~c.in_gzh_overlap & c.k15_common.notna()]
    if test.empty:
        ctx.log.info("[Stage 7b] no CANDELS-labelled galaxies outside the GZH overlap - cross-catalogue part skipped")
        if not summary:
            raise StageSkipped("no CANDELS-labelled galaxies outside the GZH overlap (and no unsealed external GZH "
                               "galaxies)")
        return summary
    g = galaxy_table(ctx).set_index("object_id").loc[test.object_id].reset_index()
    res = predict_galaxies(ctx, g)
    pred = np.array(MORPH_CLASSES)[res["morph"].argmax(1)]
    test = test.assign(pred_class=pred, pred_common=pd.Series(pred).map(GZH_TO_COMMON).to_numpy(),
                       p_merger=res["merger"])
    test.to_csv(out / "crosscat_predictions.csv", index=False)
    labs = {c_: i for i, c_ in enumerate(COMMON)}
    yt = test.k15_common.map(labs).to_numpy()
    yp = test.pred_common.map(labs).to_numpy()
    f1 = lambda a, b: f1_score(a, b, labels=sorted(set(np.asarray(a).tolist()) | set(np.asarray(b).tolist())),
                               average="macro", zero_division=0)
    m = {"xcat_macro_f1": round(float(f1(yt, yp)), 4),
         "xcat_macro_f1_ci95": bootstrap_ci(lambda a, P: f1(a, P), yt, yp, seed=ctx.seed),
         "n_test": int(len(test)), "n_excluded_overlap": int(c.in_gzh_overlap.sum()),
         "n_excluded_point_source": int(c.k15_common.isna().sum()),
         "per_class": classification_report(test.k15_common, test.pred_common, labels=list(COMMON),
                                            output_dict=True, zero_division=0)}
    if "split_tag" in test:
        m["by_split"] = {}
        for k, g in test.groupby("split_tag"):
            a, b = g.k15_common.map(labs).to_numpy(), g.pred_common.map(labs).to_numpy()
            m["by_split"][str(k)] = {"n": int(len(g)), "xcat_macro_f1": round(float(f1(a, b)), 4)}
    save_json(m, out / "cross_catalogue_metrics.json")
    ctx.log.info(f"[Stage 7b] cross-catalogue Macro-F1 = {m['xcat_macro_f1']} on n = {m['n_test']} | per split "
                 f"{m.get('by_split')}\n" +
                 pd.crosstab(test.k15_common, test.pred_common, rownames=["CANDELS"], colnames=["model"]).to_string())
    summary["xcat_macro_f1"] = m["xcat_macro_f1"]
    return summary


def _external_gzh(ctx, out):
    """GZH-labelled galaxies of the external pointings, scored like the Stage 6a test (None if there are none)."""
    df = galaxy_table(ctx)
    ext = df[~df.is_pseudo_label & (df.split_tag == "external") & ~df.edge_flag]
    if ctx.cfg["frame_qc"].get("exclude_flagged"):
        ext = ext[ext.frame_quality_flag == 1]
    ext = ext.reset_index(drop=True)
    if ext.empty:
        ctx.log.info("[Stage 7b] no GZH-labelled galaxies in the external pointings - GZH part skipped")
        return None
    res = predict_galaxies(ctx, ext)
    y, P = ext.gz_label_id.astype(int).to_numpy(), res["morph"]
    ym = ext.gz_merger_flag.fillna(0).astype(int).to_numpy()
    pred = P.argmax(1)
    m = morph_metrics(y, P, ym, res["merger"])
    f1 = lambda yy, PP: f1_score(yy, PP.argmax(1), average="macro",
                                 labels=sorted(set(yy.tolist()) | set(PP.argmax(1).tolist())), zero_division=0)
    m["macro_f1_ci95"] = bootstrap_ci(f1, y, P, seed=ctx.seed)
    m["ece"] = round(expected_calibration_error(P, y), 4)
    m["n"] = int(len(y))
    m["per_class"] = classification_report(y, pred, labels=list(range(N_CLASSES)), target_names=list(MORPH_CLASSES),
                                           output_dict=True, zero_division=0)
    m["shape"] = shape_scores(y, P)
    m["per_pointing"] = {}
    for pt, idx in ext.groupby("pointing").indices.items():
        mm = morph_metrics(y[idx], P[idx])
        m["per_pointing"][str(pt)] = {"n": int(len(idx)), "accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"],
                                      "shape": shape_scores(y[idx], P[idx])}
    save_json(m, out / "external_gzh_metrics.json")
    pr = pd.DataFrame({"object_id": ext.object_id.to_numpy(), "pointing": ext.pointing.to_numpy(), "y_true": y,
                       "y_pred": pred, "p_merger": res["merger"]})
    for k, c_ in enumerate(MORPH_CLASSES):
        pr[f"p_{c_}"] = P[:, k]
    pr.to_csv(out / "external_gzh_predictions.csv", index=False)
    from .stage6a_test_eval import _confusion
    _confusion(y, pred, out, name="confusion_matrix_external.png",
               title="Stage 7b: external pointings, GZH labels (row-normalised, counts)")
    ctx.log.info(f"[Stage 7b] external GZH galaxies n={len(y)}: accuracy {m['accuracy']} | Macro-F1 {m['macro_f1']} "
                 f"(95% CI {m['macro_f1_ci95'][0]:.3f}-{m['macro_f1_ci95'][1]:.3f}) | balanced acc "
                 f"{m['balanced_acc']} | ECE {m['ece']}\n[Stage 7b] {shape_text(m['shape'])}\n[Stage 7b] per pointing: {m['per_pointing']}")
    return m
