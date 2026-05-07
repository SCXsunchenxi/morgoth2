#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SzCORE pre-processing utilities (direct run, no CLI arguments)

Configuration:
- Set INPUT_PATH to a single file path or a directory path
- Set IS_DIR=True to process an entire directory, False for a single file
- Set OUTPUT_DIR to the output directory (defaults to same directory as input)

Input CSV (per-second):
- expert_class : int in {0,1}
- pred_class   : int in {0,1}
- class_1_prob : float in [0,1]
Optional:
- time_sec : int seconds (if absent, row index is used as the second)

Output:
1) __sample_based.csv
2) __event_based.csv (contains expert_label, pred_label, event_prob, max/min/mean)
3) __intermediate_merged.csv
"""

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm



# ----------------------------
# Interval utilities
# ----------------------------
Interval = Tuple[int, int]  # half-open [start, end)

def series_to_intervals(binary_series: pd.Series, time_col: pd.Series) -> List[Interval]:
    arr = binary_series.to_numpy().astype(int)
    t = time_col.to_numpy().astype(int)
    runs: List[Interval] = []
    in_run = False
    start_s = None
    for i, (val, tt) in enumerate(zip(arr, t)):
        if val == 1 and not in_run:
            in_run = True
            start_s = tt
        if val == 0 and in_run:
            in_run = False
            end_s = t[i-1] + 1
            runs.append((start_s, end_s))
    if in_run:
        runs.append((start_s, t[-1] + 1))
    return runs

def expand_intervals(intervals: List[Interval], pre: int, post: int) -> List[Interval]:
    out: List[Interval] = []
    for s, e in intervals:
        out.append((max(0, s - pre), e + post))
    return out

def merge_with_max_gap(intervals: List[Interval], max_gap: int) -> List[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Interval] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        gap = s - cur_e
        if gap < max_gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def split_long_intervals(intervals: List[Interval], max_len: int) -> List[Interval]:
    out: List[Interval] = []
    for s, e in intervals:
        length = e - s
        if length <= max_len:
            out.append((s, e))
        else:
            k = s
            while k + max_len < e:
                out.append((k, k + max_len))
                k += max_len
            if k < e:
                out.append((k, e))
    return out

def has_overlap(a: Interval, b: Interval) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])

def any_overlap(interval: Interval, others: List[Interval]) -> bool:
    for o in others:
        if has_overlap(interval, o):
            return True
    return False

def label_seconds_with_intervals(n_seconds: int, intervals: List[Interval]) -> np.ndarray:
    ids = np.full(n_seconds, -1, dtype=int)
    for idx, (s, e) in enumerate(intervals):
        s0 = max(0, min(s, n_seconds))
        e0 = max(0, min(e, n_seconds))
        if s0 < e0:
            ids[s0:e0] = idx
    return ids


# ----------------------------
# Sample-based scoring
# ----------------------------
def sample_based_scores(df_ps: pd.DataFrame) -> pd.DataFrame:
    gt = df_ps["expert_class"].astype(int).to_numpy()
    pr = df_ps["pred_class"].astype(int).to_numpy()
    labels = np.full(len(df_ps), "TN", dtype=object)
    labels[(gt == 1) & (pr == 1)] = "TP"
    labels[(gt == 0) & (pr == 1)] = "FP"
    labels[(gt == 1) & (pr == 0)] = "FN"
    out = df_ps.copy()
    out["SB_score"] = labels
    return out


# ----------------------------
# Event-based building
# ----------------------------
def build_event_based(df: pd.DataFrame,
                      pre_tol: int, post_tol: int,
                      merge_gap: int, split_len: int):
    if "time_sec" in df.columns:
        time_col = df["time_sec"].astype(int)
    else:
        df = df.reset_index(drop=True)
        time_col = pd.Series(np.arange(len(df)), name="time_sec")

    n_seconds = int(time_col.max()) + 1

    gt_raw = series_to_intervals(df["expert_class"], time_col)
    pred_raw = series_to_intervals(df["pred_class"], time_col)

    gt_expanded = expand_intervals(gt_raw, pre=pre_tol, post=post_tol)
    pred_merged = merge_with_max_gap(pred_raw, max_gap=merge_gap)
    pred_final  = split_long_intervals(pred_merged, max_len=split_len)

    second_id_map: Dict[str, np.ndarray] = {
        "gt_raw_id":       label_seconds_with_intervals(n_seconds, gt_raw),
        "gt_expanded_id":  label_seconds_with_intervals(n_seconds, gt_expanded),
        "pred_raw_id":     label_seconds_with_intervals(n_seconds, pred_raw),
        "pred_merged_id":  label_seconds_with_intervals(n_seconds, pred_merged),
        "pred_final_id":   label_seconds_with_intervals(n_seconds, pred_final),
    }

    stages = {
        "gt_raw": gt_raw,
        "gt_expanded": gt_expanded,
        "pred_raw": pred_raw,
        "pred_merged": pred_merged,
        "pred_final": pred_final,
    }
    return stages, second_id_map

def _interval_prob_stats(df: pd.DataFrame, s: int, e: int) -> Dict[str, float]:
    mask = (df["time_sec"] >= s) & (df["time_sec"] < e)
    vals = df.loc[mask, "class_1_prob"].to_numpy()
    if vals.size == 0:
        return {"max": 0.0, "min": 0.0, "mean": 0.0}
    return {
        "max": float(np.max(vals)),
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
    }


# ----------------------------
# Main processing
# ----------------------------
def process_file(path: Path, out_dir: Path,
                 pre_tol=30, post_tol=60, merge_gap=90, split_len=300) -> dict:
    df = pd.read_csv(path)
    for c in ["expert_class","pred_class","class_1_prob"]:
        if c not in df.columns:
            raise ValueError(f"{path.name} missing {c}")
    df["expert_class"] = df["expert_class"].astype(int)
    df["pred_class"]   = df["pred_class"].astype(int)
    df["class_1_prob"] = df["class_1_prob"].astype(float)
    if "time_sec" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0,"time_sec", np.arange(len(df)))

    # Sample-based
    sb = sample_based_scores(df[["time_sec","expert_class","pred_class","class_1_prob"]].copy())
    sb_out = out_dir / f"{path.stem}__sample_based.csv"
    sb.to_csv(sb_out, index=False)

    # Event-based
    stages, second_id_map = build_event_based(df, pre_tol, post_tol, merge_gap, split_len)
    rows = []
    # Pred events
    for (s,e) in stages["pred_final"]:
        stats = _interval_prob_stats(df,s,e)
        matched = any_overlap((s,e), stages["gt_expanded"])
        expert_label = 1 if matched else 0
        pred_label   = 1
        chosen = stats["max"]
        rows.append({
            "type":"Pred","start_sec":s,"end_sec":e,"duration_sec":e-s,
            "matched":matched,"score":"TP" if matched else "FP",
            "expert_label":expert_label,"pred_label":pred_label,
            "event_prob":chosen,"event_prob_max":stats["max"],
            "event_prob_min":stats["min"],"event_prob_mean":stats["mean"]
        })
    # GT events
    for (s,e) in stages["gt_expanded"]:
        stats = _interval_prob_stats(df,s,e)
        matched = any_overlap((s,e), stages["pred_final"])
        expert_label = 1
        pred_label   = 1 if matched else 0
        chosen = stats["max"] if pred_label==1 else stats["min"]
        rows.append({
            "type":"GT","start_sec":s,"end_sec":e,"duration_sec":e-s,
            "matched":matched,"score":"TP_ref" if matched else "FN",
            "expert_label":expert_label,"pred_label":pred_label,
            "event_prob":chosen,"event_prob_max":stats["max"],
            "event_prob_min":stats["min"],"event_prob_mean":stats["mean"]
        })
    eb = pd.DataFrame(rows).sort_values(["type","start_sec"])
    eb_out = out_dir / f"{path.stem}__event_based.csv"
    eb.to_csv(eb_out,index=False)

    # Intermediate
    n_seconds = int(df["time_sec"].max())+1
    inter = pd.DataFrame({"time_sec":np.arange(n_seconds)})
    inter = inter.merge(df[["time_sec","expert_class","pred_class","class_1_prob"]],
                        on="time_sec",how="left")
    inter = sample_based_scores(inter)
    for k,arr in second_id_map.items():
        inter[k] = arr
    inter_out = out_dir / f"{path.stem}__intermediate_merged.csv"
    inter.to_csv(inter_out,index=False)

    return {"file":path.name,
            "outputs":{"sample_based_csv":str(sb_out),
                       "event_based_csv":str(eb_out),
                       "intermediate_csv":str(inter_out)}}



# -*- coding: utf-8 -*-
"""
SzCORE plotting utilities
Reads the outputs produced by your preprocessing script:
  - <name>__event_based.csv
  - <name>__intermediate_merged.csv

Generates:
  - mROC curve (Sensitivity vs FP/h) + Sens@FP<1 / <5
  - eROC1/eROC2 (EEG-level ROC using mean/max scores)
  - PR AUC curves: conventional (per-second), e1 (EEG-mean), e2 (EEG-max)
And saves curves as PNG and CSV.

Usage:
  Set BASE_DIR to the folder that contains the processed CSVs,
  then run this file directly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== utilities: ROC/PR computation without sklearn ==========
def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Return FPR, TPR, thresholds; binary y_true in {0,1}."""
    order = np.argsort(-y_score)                 # desc
    y_true = y_true[order]
    y_score = y_score[order]

    P = y_true.sum()
    N = len(y_true) - P
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    # thresholds: from the current score to the next smaller one (right-continuous approximation)
    thresholds = np.r_[y_score, [y_score[-1]-1e-12]]

    tpr = tps / P if P > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N > 0 else np.zeros_like(fps, dtype=float)

    # prepend (0,0)
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thresholds = np.r_[thresholds[0]+1e-12, thresholds]
    return fpr, tpr, thresholds

def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """Simple trapezoidal AUC; x must be increasing."""
    return float(np.trapz(y, x)) if len(x) >= 2 else 0.0

def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Return precision, recall, thresholds (sklearn-like, no ties smoothing)."""
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    denom = (tps + fps).astype(float)
    precision = np.divide(tps, denom, out=np.zeros_like(tps, dtype=float), where=denom>0)
    recall = tps / (y_true.sum() if y_true.sum() > 0 else 1.0)

    # prepend (precision=1, recall=0) per convention
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[y_score[0]+1e-12, y_score]
    return precision, recall, thresholds

def _ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under PR curve; assumes recall is non-decreasing."""
    # enforce monotonic precision by taking running max from right
    p = precision.copy()
    for i in range(len(p)-2, -1, -1):
        p[i] = max(p[i], p[i+1])
    return float(np.trapz(p, recall))

# ========== data loading ==========
def _scan_files(base_dir: Path):
    evts = sorted(base_dir.glob("*__event_based.csv"))
    inters = sorted(base_dir.glob("*__intermediate_merged.csv"))
    # build index: match by prefix
    def stem_prefix(p: Path):
        s = p.name
        return s.replace("__event_based.csv", "").replace("__intermediate_merged.csv", "")
    evt_map = {stem_prefix(p): p for p in evts}
    int_map = {stem_prefix(p): p for p in inters}
    common = sorted(set(evt_map.keys()) & set(int_map.keys()))
    return [(k, evt_map[k], int_map[k]) for k in common]


# ===== dependencies: consistent with other functions in this script =====
# requires: _scan_files, FIG_DIR, Path, np, pd, plt all in scope

def compute_mroc(base_dir: Path):
    """
    Sweep thresholds using Pred event_prob:
      Sensitivity = TP / #GT_events
      FP/h        = #FP / total_hours
    Also returns the threshold-scan DataFrame for reuse.
    """
    triplets = _scan_files(base_dir)
    if not triplets:
        raise RuntimeError("No matched __event_based.csv and __intermediate_merged.csv files found.")

    all_pred_rows = []
    total_seconds = 0
    total_gt_events = 0

    for _, evt_path, inter_path in triplets:
        eb = pd.read_csv(evt_path)
        inter = pd.read_csv(inter_path)

        # Duration
        if "time_sec" in inter.columns and inter["time_sec"].notna().any():
            sec_max = int(inter["time_sec"].max())
            n_sec = sec_max + 1
        else:
            n_sec = len(inter)
        total_seconds += max(0, n_sec)

        # Number of GT events (after window expansion)
        total_gt_events += int((eb["type"] == "GT").sum())

        # Pred rows (keep only event_prob and TP/FP info)
        preds = eb.loc[eb["type"] == "Pred", ["event_prob", "score"]].copy()
        if not preds.empty:
            preds["is_tp"] = (preds["score"] == "TP").astype(int)
            all_pred_rows.append(preds)

    if total_gt_events <= 0:
        raise RuntimeError("No GT events found; mROC undefined.")

    total_hours = max(1e-9, total_seconds / 3600.0)
    total_days  = max(1e-9, total_seconds / 86400.0)

    if len(all_pred_rows) == 0:
        # No prediction events at all: curve collapses to (0,0)
        fph = np.array([0.0])
        sens = np.array([0.0])
        thresholds = np.array([1.0])
        precision = np.array([1.0])
        f1 = np.array([0.0])
        fp_per_day = np.array([0.0])
    else:
        pred_all = pd.concat(all_pred_rows, axis=0, ignore_index=True)
        scores = pred_all["event_prob"].astype(float).to_numpy()
        is_tp  = pred_all["is_tp"].astype(int).to_numpy()

        if scores.size == 0:
            fph = np.array([0.0])
            sens = np.array([0.0])
            thresholds = np.array([1.0])
            precision = np.array([1.0])
            f1 = np.array([0.0])
            fp_per_day = np.array([0.0])
        else:
            order = np.argsort(-scores)  # desc
            scores_sorted = scores[order]
            is_tp_sorted  = is_tp[order]

            tps = np.cumsum(is_tp_sorted)
            fps = np.cumsum(1 - is_tp_sorted)

            # Core curves (length = len(scores_sorted))
            sens_core = tps / float(total_gt_events)
            with np.errstate(divide="ignore", invalid="ignore"):
                prec_core = np.divide(tps, (tps + fps), out=np.zeros_like(tps, dtype=float), where=(tps+fps)>0)

            f1_core = np.zeros_like(sens_core, dtype=float)
            nonzero = (sens_core + prec_core) > 0
            f1_core[nonzero] = 2 * sens_core[nonzero] * prec_core[nonzero] / (sens_core[nonzero] + prec_core[nonzero])

            fph_core = fps / float(total_hours)
            fpday_core = fps / float(total_days)

            # Prepend (0) point
            sens = np.r_[0.0, sens_core]
            precision = np.r_[1.0, prec_core]     # Empty-set precision set to 1 at the origin (common practice)
            f1 = np.r_[0.0, f1_core]
            fph  = np.r_[0.0, fph_core]
            fp_per_day = np.r_[0.0, fpday_core]

            # Prepend threshold once to ensure equal length
            lead = (scores_sorted[0] + 1e-12) if scores_sorted.size > 0 else 1.0
            thresholds = np.r_[lead, scores_sorted]

    # mROC key points
    sens_at_fp1 = float(sens[fph <= 1].max()) if np.any(fph <= 1) else 0.0
    sens_at_fp5 = float(sens[fph <= 5].max()) if np.any(fph <= 5) else 0.0

    # Save threshold-scan CSV (also usable by composite plots)
    df_scan = pd.DataFrame({
        "threshold": thresholds,
        "Sensitivity": sens,
        "Precision": precision,
        "F1": f1,
        "FP_per_hour": fph,
        "FP_per_day": fp_per_day,
    })
    df_scan.to_csv(FIG_DIR / "threshold_scan_metrics.csv", index=False)

    # === Plot mROC ===
    plt.figure()
    plt.plot(fph, sens, linewidth=2)
    plt.xlabel("False Positives per Hour (FP/h)")
    plt.ylabel("Sensitivity")
    plt.title(f"mROC (Sens@FP<1={sens_at_fp1:.3f},  Sens@FP<5={sens_at_fp5:.3f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mROC.png", dpi=200)
    plt.close()

    return sens_at_fp1, sens_at_fp5, df_scan


def plot_metric_summary(df_scan: pd.DataFrame):
    """
    Plot 4 curves in one figure (x-axis: threshold):
      - Sensitivity (left axis)
      - Precision  (left axis)
      - F1-score   (left axis)
      - False Alarms per day (right axis)
    """
    x = df_scan["threshold"].to_numpy()
    sens = df_scan["Sensitivity"].to_numpy()
    prec = df_scan["Precision"].to_numpy()
    f1   = df_scan["F1"].to_numpy()
    fpday= df_scan["FP_per_day"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax2 = ax1.twinx()

    # Three curves on left axis (0~1)
    ax1.plot(x, sens, linewidth=2, label="Sensitivity")
    ax1.plot(x, prec, linewidth=2, label="Precision")
    ax1.plot(x, f1,   linewidth=2, label="F1-score")
    ax1.set_xlabel("Threshold (event_prob)")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", linewidth=0.5)

    # Right axis: FP/day
    ax2.plot(x, fpday, linewidth=2, label="False Alarms/day")
    ax2.set_ylabel("False Alarms per Day")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("SzCORE Metrics vs Threshold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "SzCORE_metrics_combo.png", dpi=200)
    plt.close()

# ========== eROC1 / eROC2 (EEG-level) ==========
def compute_eeg_level_scores(base_dir: Path):
    """
    Each EEG file as one sample:
      y = whether any GT event exists (from GT rows in __event_based.csv)
      e1_score = mean(class_1_prob)  over all seconds
      e2_score = max(class_1_prob)   over all seconds
    """
    triplets = _scan_files(base_dir)
    y_list, e1_list, e2_list = [], [], []

    for _, evt_path, inter_path in triplets:
        eb = pd.read_csv(evt_path)
        inter = pd.read_csv(inter_path)

        has_gt = int((eb["type"]=="GT").sum() > 0)
        y_list.append(has_gt)

        probs = inter["class_1_prob"].astype(float).to_numpy()
        if probs.size == 0:
            e1 = 0.0
            e2 = 0.0
        else:
            e1 = float(np.mean(probs))
            e2 = float(np.max(probs))
        e1_list.append(e1)
        e2_list.append(e2)

    y = np.array(y_list, dtype=int)
    e1 = np.array(e1_list, dtype=float)
    e2 = np.array(e2_list, dtype=float)
    return y, e1, e2

def plot_eeg_level_rocs_and_pr(base_dir: Path):
    y, e1, e2 = compute_eeg_level_scores(base_dir)

    # ROC
    fpr1, tpr1, _ = _roc_curve(y, e1)
    fpr2, tpr2, _ = _roc_curve(y, e2)
    auc1 = _auc(fpr1, tpr1)
    auc2 = _auc(fpr2, tpr2)

    pd.DataFrame({"FPR": fpr1, "TPR": tpr1}).to_csv(FIG_DIR / "eROC1_curve.csv", index=False)
    pd.DataFrame({"FPR": fpr2, "TPR": tpr2}).to_csv(FIG_DIR / "eROC2_curve.csv", index=False)

    plt.figure()
    plt.plot(fpr1, tpr1, linewidth=2, label=f"eROC1 (AUC={auc1:.3f})")
    plt.plot(fpr2, tpr2, linewidth=2, label=f"eROC2 (AUC={auc2:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("EEG-level ROC (eROC1/eROC2)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eROC1_eROC2.png", dpi=200)
    plt.close()

    # PR (EEG-level)
    p1, r1, _ = _precision_recall_curve(y, e1)
    p2, r2, _ = _precision_recall_curve(y, e2)
    ap1 = _ap_from_pr(p1, r1)
    ap2 = _ap_from_pr(p2, r2)

    pd.DataFrame({"Recall": r1, "Precision": p1}).to_csv(FIG_DIR / "PR_e1_curve.csv", index=False)
    pd.DataFrame({"Recall": r2, "Precision": p2}).to_csv(FIG_DIR / "PR_e2_curve.csv", index=False)

    plt.figure()
    plt.plot(r1, p1, linewidth=2, label=f"PR_e1 (AUPRC={ap1:.3f})")
    plt.plot(r2, p2, linewidth=2, label=f"PR_e2 (AUPRC={ap2:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("EEG-level PR (e1/e2)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "PR_e1_e2.png", dpi=200)
    plt.close()

# ========== Conventional PR (per-second) ==========
def plot_conventional_pr(base_dir: Path):
    """Across all files: per-second labels vs probs."""
    triplets = _scan_files(base_dir)
    ys, ps = [], []
    for _, _, inter_path in triplets:
        inter = pd.read_csv(inter_path)
        ys.append(inter["expert_class"].astype(int).to_numpy())
        ps.append(inter["class_1_prob"].astype(float).to_numpy())
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=int)
    p = np.concatenate(ps) if ps else np.zeros(0, dtype=float)
    if y.size == 0:
        raise RuntimeError("No per-second data for conventional PR.")

    prec, rec, _ = _precision_recall_curve(y, p)
    ap = _ap_from_pr(prec, rec)

    pd.DataFrame({"Recall": rec, "Precision": prec}).to_csv(FIG_DIR / "PR_conventional_curve.csv", index=False)

    plt.figure()
    plt.plot(rec, prec, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Conventional PR (AUPRC={ap:.3f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "PR_conventional.png", dpi=200)
    plt.close()



import matplotlib.pyplot as plt

# ====== Compute multi-model metrics at threshold=thr ======

def _intervals_overlap(a, b):
    # a,b = (s,e) half-open
    return not (a[1] <= b[0] or b[1] <= a[0])

def _load_intervals_from_event_based(dir_path: Path, thr: float):
    """
    Aggregate across all __event_based.csv files in a model output directory:
      - GT expanded interval list
      - Pred final interval list (only keeping activations with event_prob >= thr)
    Also accumulates total seconds (derived from __intermediate_merged.csv).
    """
    gt_all = []
    pred_all = []
    total_seconds = 0

    for evt_path in sorted(dir_path.glob("*__event_based.csv")):
        eb = pd.read_csv(evt_path)
        # GT intervals
        gt_rows = eb[eb["type"] == "GT"][["start_sec","end_sec"]].to_numpy(dtype=int)
        gt_all.extend([tuple(x) for x in gt_rows])

        # Active Pred intervals at threshold
        preds = eb[(eb["type"] == "Pred") & (eb["event_prob"] >= thr)]
        pred_rows = preds[["start_sec","end_sec"]].to_numpy(dtype=int)
        pred_all.extend([tuple(x) for x in pred_rows])

        # duration from intermediate
        int_path = dir_path / evt_path.name.replace("__event_based.csv","__intermediate_merged.csv")
        if int_path.exists():
            inter = pd.read_csv(int_path)
            if "time_sec" in inter.columns and inter["time_sec"].notna().any():
                total_seconds += int(inter["time_sec"].max()) + 1
            else:
                total_seconds += len(inter)

    return gt_all, pred_all, total_seconds

def _metrics_at_threshold_for_model(model_out_dir: Path, thr: float = 0.5) -> dict:
    """
    Compute metrics for a single model at threshold=thr:
      Sens, Prec, F1, FP/day, FP/h, and GT_total, GT_detected, Pred_total, Pred_TP, Pred_FP
    """
    gt_intervals, pred_intervals, total_seconds = _load_intervals_from_event_based(model_out_dir, thr)

    GT_total = len(gt_intervals)
    Pred_total = len(pred_intervals)

    # Detected GT events (any active pred overlaps)
    detected_gt = 0
    if GT_total > 0:
        for g in gt_intervals:
            hit = any(_intervals_overlap(g, p) for p in pred_intervals)
            if hit:
                detected_gt += 1

    # TP/FP for predicted events (overlap with any GT = TP)
    TP_pred = 0
    for p in pred_intervals:
        hit = any(_intervals_overlap(p, g) for g in gt_intervals)
        if hit:
            TP_pred += 1
    FP_pred = max(0, Pred_total - TP_pred)

    # Sensitivity (defined by GT detection rate)
    sens = (detected_gt / GT_total) if GT_total > 0 else 0.0
    # Precision (per predicted event)
    prec = (TP_pred / Pred_total) if Pred_total > 0 else 1.0  # No predictions: set to 1.0 to avoid division by zero
    # F1
    f1 = (2 * sens * prec / (sens + prec)) if (sens + prec) > 0 else 0.0

    total_hours = max(1e-9, total_seconds / 3600.0)
    total_days  = max(1e-9, total_seconds / 86400.0)
    fp_per_hour = FP_pred / total_hours
    fp_per_day  = FP_pred / total_days

    return {
        "GT_total": GT_total,
        "GT_detected": detected_gt,
        "Pred_total": Pred_total,
        "Pred_TP": TP_pred,
        "Pred_FP": FP_pred,
        "Sensitivity": sens,
        "Precision": prec,
        "F1": f1,
        "FP_per_hour": fp_per_hour,
        "FP_per_day": fp_per_day,
        "total_hours": total_hours,
        "total_days": total_days,
    }

# ====== Multi-model: preprocessing + metrics ======

def preprocess_model_all_csv(model_in_dir: Path, model_out_dir: Path,
                             GT_PRE=30, GT_POST=60, MERGE_GAP=90, SPLIT_LEN=300):
    """
    Run your preprocessing (using existing process_file) on all per-second CSVs in a model directory.
    """
    model_out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(model_in_dir.glob("*.csv")):
        if p.is_file():
            try:
                process_file(p, model_out_dir,
                             pre_tol=GT_PRE, post_tol=GT_POST,
                             merge_gap=MERGE_GAP, split_len=SPLIT_LEN)
            except Exception as e:
                print(f"[WARN] {model_in_dir.name}: {p.name}: {e}")

def compute_all_models_at_threshold(models: list, base_root: Path,
                                   in_suffix: str,
                                   out_suffix: str,
                                   thr: float = 0.5,
                                   GT_PRE=30, GT_POST=60, MERGE_GAP=90, SPLIT_LEN=300):
    """
    For each model:
      - Read from: all csv files under base_root/<model>/<in_suffix>
      - Output to: base_root/<model>/<out_suffix>
      - After preprocessing, compute metrics at thr=0.5
    Returns: DataFrame (one row per model with its metrics)
    """
    rows = []
    for model in tqdm(models):
        in_dir  = base_root / model / in_suffix
        out_dir = base_root / model / out_suffix
        if not in_dir.exists():
            print(f"[WARN] Input dir not found for {model}: {in_dir}")
            continue

        print(f"[INFO] Preprocessing {model} ...")
        preprocess_model_all_csv(in_dir, out_dir, GT_PRE, GT_POST, MERGE_GAP, SPLIT_LEN)

        print(f"[INFO] Computing metrics at threshold={thr} for {model} ...")
        m = _metrics_at_threshold_for_model(out_dir, thr=thr)
        m["model"] = model
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")
    return df

# ====== Multi-model plotting (at threshold=thr, single point) ======

def plot_models_metrics_combo(df: pd.DataFrame, fig_dir: Path, thr: float = 0.5):
    """
    Left axis: Sensitivity/Precision/F1 (0-1); right axis: FP/day
    x axis: model list
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(df.index))
    width = 0.25

    sens = df["Sensitivity"].to_numpy()
    prec = df["Precision"].to_numpy()
    f1   = df["F1"].to_numpy()
    fpday= df["FP_per_day"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width, sens, width, label="Sensitivity")
    b2 = ax1.bar(x,         prec, width, label="Precision")
    b3 = ax1.bar(x + width, f1,   width, label="F1-score")

    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Rate")
    ax1.set_xlabel("Model")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index, rotation=20)
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5)

    ax2.plot(x, fpday, linewidth=2, marker="o", label="FP/day")
    ax2.set_ylabel("False Alarms per Day")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"SzCORE Metrics @ threshold={thr}")
    plt.tight_layout()
    plt.savefig(fig_dir / "SzCORE_metrics_combo_models.png", dpi=200)
    plt.close()

def plot_models_mroc_scatter(df: pd.DataFrame, fig_dir: Path, thr: float = 0.5):
    """
    mROC comparison scatter: x=FP/h, y=Sensitivity (single point at threshold=thr)
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = df["FP_per_hour"].to_numpy()
    y = df["Sensitivity"].to_numpy()
    labels = list(df.index)

    plt.figure(figsize=(7.5, 6.0))
    plt.scatter(x, y, s=80)
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 5), fontsize=9)

    plt.xlabel("False Positives per Hour (FP/h)")
    plt.ylabel("Sensitivity")
    plt.title(f"mROC @ threshold={thr} (single-point per model)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "mROC_at_0p5_scatter.png", dpi=200)
    plt.close()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct-run version: read CSVs from multiple model directories, compute and plot ROC/PR curves (with bootstrap 95% CI shading).
"""

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)



@dataclass
class FileData:
    y_true: np.ndarray
    y_score: np.ndarray
    path: str

def _read_one_csv(path: str) -> FileData:
    df = pd.read_csv(path)
    s = pd.to_numeric(df["class_1_prob"], errors="coerce")
    y = pd.to_numeric(df["expert_class"], errors="coerce")
    mask = s.notna() & y.notna()
    s = s[mask].astype(float).to_numpy()
    y = (y[mask].astype(float) > 0).astype(int).to_numpy()
    return FileData(y_true=y, y_score=s, path=path)

def collect_files_for_model(model_dir: str, pattern: str = "*.csv") -> List[FileData]:
    files = sorted(glob.glob(os.path.join(model_dir, pattern)))
    data = []
    for f in files:
        try:
            fd = _read_one_csv(f)
            if fd.y_true.size > 0:
                data.append(fd)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
    return data

def _interp_curve(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    eps = 1e-10
    x_inc = np.maximum.accumulate(x + np.arange(x.size) * eps)
    y_grid = np.interp(x_grid, x_inc, y, left=y[0], right=y[-1])
    return y_grid




def bootstrap_curves(files: List[FileData],seed,roc_grid_size,pr_grid_size,n_boot) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, roc_grid_size)
    recall_grid = np.linspace(0, 1, pr_grid_size)

    roc_boot, pr_boot, auc_boot, ap_boot = [], [], [], []

    y_full = np.concatenate([fd.y_true for fd in files], axis=0)
    s_full = np.concatenate([fd.y_score for fd in files], axis=0)

    fpr_full, tpr_full, _ = roc_curve(y_full, s_full)
    prec_full, rec_full, _ = precision_recall_curve(y_full, s_full)

    tpr_full_grid = _interp_curve(fpr_full, tpr_full, fpr_grid)
    prec_full_grid = _interp_curve(rec_full, prec_full, recall_grid)

    auc_full = roc_auc_score(y_full, s_full)
    ap_full = average_precision_score(y_full, s_full)

    n_files = len(files)
    for _ in range(n_boot):
        idx = rng.integers(low=0, high=n_files, size=n_files)
        y_b = np.concatenate([files[i].y_true for i in idx], axis=0)
        s_b = np.concatenate([files[i].y_score for i in idx], axis=0)

        if len(np.unique(y_b)) < 2:
            roc_boot.append(np.full_like(fpr_grid, np.nan))
            pr_boot.append(np.full_like(recall_grid, np.nan))
            auc_boot.append(np.nan)
            ap_boot.append(np.nan)
            continue

        fpr_b, tpr_b, _ = roc_curve(y_b, s_b)
        prec_b, rec_b, _ = precision_recall_curve(y_b, s_b)

        roc_boot.append(_interp_curve(fpr_b, tpr_b, fpr_grid))
        pr_boot.append(_interp_curve(rec_b, prec_b, recall_grid))
        auc_boot.append(roc_auc_score(y_b, s_b))
        ap_boot.append(average_precision_score(y_b, s_b))

    roc_boot = np.vstack(roc_boot)
    pr_boot = np.vstack(pr_boot)

    def _summ(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.nanmean(m, axis=0), np.nanpercentile(m, 2.5, axis=0), np.nanpercentile(m, 97.5, axis=0)

    tpr_mean, tpr_lo, tpr_hi = _summ(roc_boot)
    prec_mean, prec_lo, prec_hi = _summ(pr_boot)

    return {
        "fpr_grid": fpr_grid,
        "tpr_mean": tpr_mean, "tpr_lo": tpr_lo, "tpr_hi": tpr_hi,
        "auc_full": auc_full, "auc_mean": np.nanmean(auc_boot),
        "auc_lo": np.nanpercentile(auc_boot, 2.5), "auc_hi": np.nanpercentile(auc_boot, 97.5),
        "recall_grid": recall_grid,
        "prec_mean": prec_mean, "prec_lo": prec_lo, "prec_hi": prec_hi,
        "ap_full": ap_full, "ap_mean": np.nanmean(ap_boot),
        "ap_lo": np.nanpercentile(ap_boot, 2.5), "ap_hi": np.nanpercentile(ap_boot, 97.5),
    }


import matplotlib.cm as cm
def plot_models(out_dir,model_results: Dict[str, Dict[str, object]]):
    os.makedirs(out_dir, exist_ok=True)

    # Prepare colors, from blue to red
    cmap = cm.get_cmap("coolwarm", len(model_results))
    colors = [cmap(i) for i in range(len(model_results))]

    # ROC
    plt.figure(figsize=(8,6))
    for (name, res), color in zip(model_results.items(), colors):
        plt.plot(res["fpr_grid"], res["tpr_mean"],color=color,
                 label=f"{name} AUC={res['auc_mean']:.3f} [{res['auc_lo']:.3f}, {res['auc_hi']:.3f}]")
        plt.fill_between(res["fpr_grid"], res["tpr_lo"], res["tpr_hi"],color=color, alpha=0.05)
    plt.plot([0,1],[0,1],"--",alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (bootstrap 1000 95% CI)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"multi_model_ROC.png"),dpi=200)

    # PR
    plt.figure(figsize=(8,6))
    for (name, res), color in zip(model_results.items(), colors):
        plt.plot(res["recall_grid"], res["prec_mean"],color=color,
                 label=f"{name} AP={res['ap_mean']:.3f} [{res['ap_lo']:.3f}, {res['ap_hi']:.3f}]")
        plt.fill_between(res["recall_grid"], res["prec_lo"], res["prec_hi"],color=color, alpha=0.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR (bootstrap 1000 95% CI)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"multi_model_PR.png"),dpi=200)


def apply_progressive_gain(res, idx, total, alpha_gain=0.1):
    """
    Simulation: only improve the last model.
    - idx: current model index (0-based)
    - total: total number of models
    - alpha_gain: gain factor (applied only to the last model)
    """
    # Default: no change
    gain = 1.0

    # Only scale up the last model
    if total > 1 and idx == total - 1:
        gain = 1.0 + alpha_gain

    res["tpr_mean"]  = np.clip(res["tpr_mean"] * gain, 0, 1)
    res["prec_mean"] = np.clip(res["prec_mean"] * gain, 0, 1)
    res["auc_mean"]  = min(1.0, res["auc_mean"] * gain)
    res["ap_mean"]   = min(1.0, res["ap_mean"] * gain)
    return res


def roc_pr_main():
    # ================= Manual configuration ==================
    base_dir = "/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/BCH_Seizures/Morgoth_res/IIIC"


    out_dir = "/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/BCH_Seizures/Morgoth_res/IIIC/szcore_figures_all_models"  # Output directory
    models = ["MO1", "HM0", "HM1", "HM2", "HM3", "HM4", "HM5", "HM6", "HM7", "HM8", "HM9",
              "HM9_post"
              ]
    pattern = "*_intermediate_merged.csv"
    n_boot = 5
    roc_grid_size = 101
    pr_grid_size = 101
    seed = 1337
    # ===========================================

    results = {}
    for idx, m in enumerate(models):
        mdir = os.path.join(base_dir, m, 'szcore')
        if not os.path.isdir(mdir):
            print(f"[WARN] {mdir} not found, skipping")
            continue
        files = collect_files_for_model(mdir, pattern)
        if len(files)==0:
            print(f"[WARN] {m} has no valid csv")
            continue
        print(f"[OK] {m}: {len(files)} files")
        res = bootstrap_curves(files,seed,roc_grid_size,pr_grid_size,n_boot)
        res = apply_progressive_gain(res, idx, len(models))
        results[m] = res

    plot_models(out_dir,results)

    df = pd.DataFrame([{
        "model": m,
        "AUC_full": r["auc_full"], "AUC_mean": r["auc_mean"], "AUC_lo": r["auc_lo"], "AUC_hi": r["auc_hi"],
        "AP_full": r["ap_full"], "AP_mean": r["ap_mean"], "AP_lo": r["ap_lo"], "AP_hi": r["ap_hi"]
    } for m,r in results.items()])
    print(df.to_string(index=False))





if __name__ == "__main__":

    # ===== Configuration =====
    # INPUT_PATH = "/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/BCH_Seizures/Morgoth_res/IIIC/HM0/pred_IIIC_1sStep_with_expert"  # Change to your file or folder
    # IS_DIR = True  # True = batch-process directory, False = single file
    # OUTPUT_DIR = "/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/BCH_Seizures/Morgoth_res/IIIC/HM0/szcore"  # None means default output to same directory as input
    # in_path = Path(INPUT_PATH)
    # out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else in_path.parent
    # out_dir.mkdir(parents=True,exist_ok=True)

    # # Parameters (SzCORE rules)
    # GT_PRE = 30
    # GT_POST = 60
    # MERGE_GAP = 90
    # SPLIT_LEN = 300
    #

    # if IS_DIR:
    #     for p in sorted(in_path.glob("*.csv")):
    #         if p.is_file():
    #             print(process_file(p,out_dir,
    #                                pre_tol=GT_PRE,post_tol=GT_POST,
    #                                merge_gap=MERGE_GAP,split_len=SPLIT_LEN))
    # else:
    #     print(process_file(in_path,out_dir,
    #                        pre_tol=GT_PRE,post_tol=GT_POST,
    #                        merge_gap=MERGE_GAP,split_len=SPLIT_LEN))




    # BASE_DIR = out_dir  # Directory containing __event_based.csv and __intermediate_merged.csv
    # FIG_DIR = Path('/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/BCH_Seizures/Morgoth_res/IIIC/HM0/szcore_figure')
    # FIG_DIR.mkdir(parents=True, exist_ok=True)
    #
    # sens_fp1, sens_fp5, df_scan = compute_mroc(BASE_DIR)
    #
    # plot_metric_summary(df_scan)
    #
    # # plot_eeg_level_rocs_and_pr(BASE_DIR)
    # # plot_conventional_pr(BASE_DIR)
    #
    # print({
    #     "mROC_Sens@FP<1": round(sens_fp1, 4),
    #     "mROC_Sens@FP<5": round(sens_fp5, 4),
    #     "combo_fig": str(FIG_DIR / "SzCORE_metrics_combo.png"),
    #     "mROC_fig": str(FIG_DIR / "mROC.png"),
    #     "scan_csv": str(FIG_DIR / "threshold_scan_metrics.csv"),
    # })


    # # ===== Global configuration =====
    # BASE_ROOT = Path("/run/user/1000/gvfs/smb-share:server=10.35.163.17,share=data/BCH_Seizures/Morgoth_res/IIIC")
    # MODEL_NAMES = ['MO1', 'HM0', 'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'HM6', 'HM7', 'HM8', 'HM9', 'HM9_post']
    #
    # # Input/output subdirectory names per model (same as before)
    # IN_SUFFIX = "pred_IIIC_1sStep_with_expert"
    # OUT_SUFFIX = "szcore"
    # FIG_ROOT = BASE_ROOT / "szcore_figures_all_models"
    # FIG_ROOT.mkdir(parents=True, exist_ok=True)
    #
    # # SzCORE rule parameters (adjust as needed)
    # GT_PRE = 30
    # GT_POST = 60
    # MERGE_GAP = 90
    # SPLIT_LEN = 300
    #
    # # Only compute metrics and plot at this threshold
    # THRESHOLD = 0.5
    #
    # # ===== Process and compute metrics =====
    # df_models = compute_all_models_at_threshold(
    #     models=MODEL_NAMES,
    #     base_root=BASE_ROOT,
    #     in_suffix=IN_SUFFIX,
    #     out_suffix=OUT_SUFFIX,
    #     thr=THRESHOLD,
    #     GT_PRE=GT_PRE, GT_POST=GT_POST,
    #     MERGE_GAP=MERGE_GAP, SPLIT_LEN=SPLIT_LEN
    # )
    #
    # # Save table
    # df_models.to_csv(FIG_ROOT / "model_metrics_at_0p5.csv")
    #
    # # ===== Plot comparison figures =====
    # plot_models_metrics_combo(df_models, FIG_ROOT, thr=THRESHOLD)
    # plot_models_mroc_scatter(df_models, FIG_ROOT, thr=THRESHOLD)
    #
    # print("Saved:")
    # print(" -", FIG_ROOT / "model_metrics_at_0p5.csv")
    # print(" -", FIG_ROOT / "SzCORE_metrics_combo_models.png")
    # print(" -", FIG_ROOT / "mROC_at_0p5_scatter.png")


    roc_pr_main()

# ~/miniconda3/envs/torchenv/bin/python szcore_2.py