# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Tuple
from tqdm import tqdm
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def merge_bird_results():
    # ===== 1. Load CSV =====
    df_lab = pd.read_csv("/data/BIRD/birds_labels_20251128.csv")
    df_res = pd.read_csv("/data/BIRD/BIRD_experts_model_results.csv")

    # ===== 2. Merge (df_lab as left/main table) =====
    df = df_lab.merge(df_res, how="left",
                      left_on="file_name",
                      right_on="file")

    # ===== 3. Columns to process =====
    cols = ["jenna", "matt", "tianyu", "wan-yee"]

    # ===== 4. Mapping rules =====
    def map_label(x):
        if isinstance(x, str):
            x = x.strip().lower()
            if x == "not birds":
                return 0
            elif x == "possible birds":
                return -1
        # default
        return 1

    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(map_label)
        else:
            print(f"Warning: column {c} not found in labels CSV.")


    df = df.rename(columns={
        "score_mbw": "brandon",
        "score_yyy": "yangyang"
    })

    # ===== 4. Keep only file_name =====
    df["file_name"] = df["file_name"].fillna(df["file"])
    df = df.drop(columns=["file"])

    # ===== 5. Majority vote =====
    expert_cols = ["jenna", "matt", "tianyu", "wan-yee", "brandon", "yangyang"]
    def majority_vote(row):
        values = row[expert_cols].tolist()

        # Count votes
        counter = Counter(values)
        most_common = counter.most_common()

        top_count = most_common[0][1]

        # Find all values with the maximum vote count
        tied = [v for v, c in most_common if c == top_count]

        # If tied (tie >= 2), return -1
        if len(tied) > 1:
            return -1

        # Otherwise return the unique majority
        return tied[0]

    # Create the majority column
    df["majority"] = df.apply(majority_vote, axis=1)

    df["pred_class"] = (df["pred"] >= 0.5).astype(int)

    # ===== Save results =====
    df.to_csv("/data/BIRD/merged_labeled_and_results_20251208.csv", index=False)



    print("Done! Output file: /data/BIRD/merged_labeled_and_results_20251208.csv")



def vis_birds_irr(result_csv_path, output_path):

    # ===== 1. Load data =====
    df = pd.read_csv(result_csv_path)
    df = df[df["majority"] != -1].reset_index(drop=True)

    # ===== 2. Original expert column names =====
    orig_experts = ["jenna", "matt", "tianyu", "wan-yee", "brandon", "yangyang"]
    majority_col = "majority"
    model_col = "pred_class"

    # ===== 3. Mapping to expert_1 ... expert_6 =====
    mapped_names = {orig: f"expert_{i+1}" for i, orig in enumerate(orig_experts)}
    # reverse map for plotting
    reverse_map = {v: k for k, v in mapped_names.items()}

    # x-axis raters
    x_raters = [mapped_names[e] for e in orig_experts] + [majority_col]

    # all comparison partners
    unique_others = [mapped_names[e] for e in orig_experts] + [model_col]

    # ===== 4. Compute IRR =====
    records = []

    for base in x_raters:

        if base in mapped_names.values():   # expert
            orig_base = reverse_map[base]

            others = [mapped_names[e] for e in orig_experts if e != orig_base] + [model_col]

            for other in others:
                # map back to original column
                if other == model_col:
                    col_other = model_col
                else:
                    col_other = reverse_map[other]

                kappa = cohen_kappa_score(df[orig_base], df[col_other])
                records.append({
                    "base": base,
                    "other": other,
                    "kappa": kappa
                })

        else:
            # majority only vs model
            kappa = cohen_kappa_score(df[majority_col], df[model_col])
            records.append({
                "base": majority_col,
                "other": model_col,
                "kappa": kappa
            })

    irr_df = pd.DataFrame(records)

    # ===== 5. Colors =====
    cmap = plt.get_cmap("tab10")
    color_map = {name: cmap(i % 10) for i, name in enumerate(unique_others)}

    # ===== 6. Plot =====
    fig, ax = plt.subplots(figsize=(9, 4))

    for i, base in enumerate(x_raters):
        sub = irr_df[irr_df["base"] == base]
        for _, row in sub.iterrows():
            ax.scatter(
                i,
                row["kappa"],
                color=color_map[row["other"]],
                label=row["other"]
            )

    ax.set_xticks(range(len(x_raters)))
    ax.set_xticklabels(x_raters, rotation=45, ha="right")
    ax.set_ylabel("Cohen's kappa")
    ax.set_ylim(-0.5, 1)
    ax.set_title("IRR")

    # ===== 7. Legend =====
    handles = []
    labels = []
    for other in unique_others:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color_map[other], markersize=8))
        labels.append(other)

    ax.legend(handles, labels, title="Compared with",
              bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_path}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc

def vis_birds_auc(
    result_csv_path, roc_out_path, pr_out_path, n_boot=100
):
    # ===== 1. Load CSV and filter majority =====
    df = pd.read_csv(result_csv_path)
    df = df[df["majority"].isin([0, 1])].reset_index(drop=True)

    if "pred" not in df.columns or "majority" not in df.columns:
        raise ValueError("CSV must contain 'pred' and 'majority' columns")

    y_true = df["majority"].values
    y_score = df["pred"].values

    experts = ["jenna", "matt", "tianyu", "wan-yee", "brandon", "yangyang"]
    display_names = {orig: f"expert {i+1}" for i, orig in enumerate(experts)}

    n = len(df)

    # ===== 2. Bootstrap containers =====
    mean_fpr = np.linspace(0, 1, 400)
    roc_tprs = []
    roc_aucs = []      # AUC-ROC for each bootstrap iteration

    mean_rec = np.linspace(0, 1, 400)
    pr_precs = []
    pr_aps = []        # AP (AUC-PR) for each bootstrap iteration

    # ===== 3. bootstrap =====
    for _ in range(n_boot):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        yt = y_true[idx]
        ys = y_score[idx]

        # ---------- ROC ----------
        fpr_b, tpr_b, _ = roc_curve(yt, ys)
        roc_aucs.append(auc(fpr_b, tpr_b))

        tpr_interp = np.interp(mean_fpr, fpr_b, tpr_b)
        tpr_interp[0], tpr_interp[-1] = 0.0, 1.0
        roc_tprs.append(tpr_interp)

        # ---------- PR ----------
        prec_b, rec_b, _ = precision_recall_curve(yt, ys)
        pr_aps.append(average_precision_score(yt, ys))

        # ⚠ sklearn's rec_b decreases from 1 → 0; reverse to ascending before interpolating
        rec_sorted = rec_b[::-1]
        prec_sorted = prec_b[::-1]

        prec_interp = np.interp(mean_rec, rec_sorted, prec_sorted)
        pr_precs.append(prec_interp)

    roc_tprs = np.array(roc_tprs)      # [n_boot, len(mean_fpr)]
    pr_precs = np.array(pr_precs)      # [n_boot, len(mean_rec)]

    # ===== 4. Compute mean curves and CI =====
    mean_tpr = roc_tprs.mean(axis=0)
    lo_tpr = np.percentile(roc_tprs, 2.5, axis=0)
    hi_tpr = np.percentile(roc_tprs, 97.5, axis=0)
    mean_auc = float(np.mean(roc_aucs))      # Mean AUC-ROC

    mean_prec = pr_precs.mean(axis=0)
    lo_prec = np.percentile(pr_precs, 2.5, axis=0)
    hi_prec = np.percentile(pr_precs, 97.5, axis=0)
    mean_ap = float(np.mean(pr_aps))         # Mean AUC-PR (AP)

    # ============================================================
    #            5. Compute expert points (ROC: FPR/TPR, PR: Rec/Prec)
    # ============================================================
    expert_points_roc = []   # list of (orig, fpr_e, tpr_e)
    expert_points_pr = []    # list of (orig, rec_e, prec_e)

    for orig in experts:
        if orig not in df.columns:
            continue

        y_e = df[orig].values
        mask = np.isin(y_e, [0, 1])
        y_e = y_e[mask]
        y_m = y_true[mask]

        if len(y_e) == 0:
            continue

        tp = np.sum((y_e == 1) & (y_m == 1))
        fn = np.sum((y_e == 0) & (y_m == 1))
        fp = np.sum((y_e == 1) & (y_m == 0))
        tn = np.sum((y_e == 0) & (y_m == 0))

        if (tp + fn) == 0 or (fp + tn) == 0:
            continue

        # ROC point
        tpr_e = tp / (tp + fn)
        fpr_e = fp / (fp + tn)
        expert_points_roc.append((orig, fpr_e, tpr_e))

        # PR point
        rec_e = tpr_e
        prec_e = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        if not np.isnan(prec_e):
            expert_points_pr.append((orig, rec_e, prec_e))

    # ============================================================
    #            6. Compute EUC: how many expert points fall below the ROC curve
    # ============================================================
    valid_roc_pts = 0
    under_count_roc = 0

    for orig, fpr_e, tpr_e in expert_points_roc:
        model_tpr_at_fpr = float(np.interp(fpr_e, mean_fpr, mean_tpr))
        valid_roc_pts += 1
        if tpr_e <= model_tpr_at_fpr:
            under_count_roc += 1

    EUC = under_count_roc / valid_roc_pts if valid_roc_pts > 0 else np.nan

    # ============================================================
    #            7. Compute EUP: how many expert points fall below the PR curve
    # ============================================================
    valid_pr_pts = 0
    under_count_pr = 0

    for orig, rec_e, prec_e in expert_points_pr:
        model_prec_at_rec = float(np.interp(rec_e, mean_rec, mean_prec))
        valid_pr_pts += 1
        if prec_e <= model_prec_at_rec:
            under_count_pr += 1

    EUP = under_count_pr / valid_pr_pts if valid_pr_pts > 0 else np.nan

    # ============================================================
    #                          8. ROC plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.fill_between(mean_fpr, lo_tpr, hi_tpr, alpha=0.2, label="Bootstrap")

    if not np.isnan(EUC):
        ax.plot(
            mean_fpr,
            mean_tpr,
            linewidth=2,
            label=f"Morgoth (AUC={mean_auc:.3f}, EUC={EUC:.2f})",
        )
    else:
        ax.plot(
            mean_fpr,
            mean_tpr,
            linewidth=2,
            label=f"Morgoth (AUC={mean_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

    # Plot expert points
    for orig, fpr_e, tpr_e in expert_points_roc:
        ax.scatter(fpr_e, tpr_e, s=60, label=display_names[orig])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("ROC")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right")

    fig.tight_layout()
    fig.savefig(roc_out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("ROC saved to:", roc_out_path)

    # ============================================================
    #                          9. PR plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.fill_between(mean_rec, lo_prec, hi_prec, alpha=0.2, label="Bootstrap")

    if not np.isnan(EUP):
        ax.plot(
            mean_rec,
            mean_prec,
            linewidth=2,
            label=f"Morgoth (AP={mean_ap:.3f}, EUP={EUP:.2f})",
        )
    else:
        ax.plot(
            mean_rec,
            mean_prec,
            linewidth=2,
            label=f"Morgoth (AP={mean_ap:.3f})",
        )

    # Plot expert points
    for orig, rec_e, prec_e in expert_points_pr:
        ax.scatter(rec_e, prec_e, s=60, label=display_names[orig])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("PR")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(),  loc="lower left")

    fig.tight_layout()
    fig.savefig(pr_out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("PR saved to:", pr_out_path)

def load_all_csv(folder: str | Path, pattern: str = "*.csv") -> Tuple[np.ndarray, np.ndarray]:
    """Read all CSVs from a folder and return y_true, y_pred (1D numpy arrays)."""
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if "true" not in df.columns or "pred" not in df.columns:
            raise ValueError(f"{fp} must contain columns: 'true' and 'pred'")
        # keep only these two columns and convert to numeric
        sub = df[["true", "pred"]].copy()
        sub["true"] = pd.to_numeric(sub["true"], errors="coerce")
        sub["pred"] = pd.to_numeric(sub["pred"], errors="coerce")
        # drop missing values
        sub = sub.dropna(subset=["true", "pred"])
        dfs.append(sub)

    all_df = pd.concat(dfs, ignore_index=True)
    y_true = all_df["true"].to_numpy().astype(int)
    y_pred = all_df["pred"].to_numpy().astype(float)

    # simple check: binary classification labels
    uniq = np.unique(y_true)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"'true' must be binary 0/1, got unique labels: {uniq}")

    return y_true, y_pred


def bootstrap_auc(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float, float, float, float]:
    """
    Bootstrap ROC AUC and PR AUC, returning:
    (roc_mean, roc_lo, roc_hi, pr_mean, pr_lo, pr_hi)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    roc_aucs = []
    pr_aucs  = []
    for _ in tqdm(range(n_bootstrap),desc='bootstrap'):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]

        # ROC AUC
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_aucs.append(auc(fpr, tpr))

        # PR AUC (approximated by integration, differs from average_precision)
        prec, rec, _ = precision_recall_curve(yt, yp)
        pr_aucs.append(auc(rec, prec))

    roc_aucs = np.array(roc_aucs)
    pr_aucs  = np.array(pr_aucs)

    roc_mean = roc_aucs.mean()
    pr_mean  = pr_aucs.mean()

    roc_lo, roc_hi = np.percentile(roc_aucs, [2.5, 97.5])
    pr_lo,  pr_hi  = np.percentile(pr_aucs,  [2.5, 97.5])
    return roc_mean, roc_lo, roc_hi, pr_mean, pr_lo, pr_hi


def plot_roc_pr(y_true: np.ndarray, y_pred: np.ndarray,
                roc_stats: Tuple[float, float, float, float, float, float],
                save_prefix: str | None = None) -> None:
    """Plot ROC / PR curves with bootstrap AUC and 95% CI in the legend; optionally save."""
    roc_mean, roc_lo, roc_hi, pr_mean, pr_lo, pr_hi = roc_stats

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_mean:.3f} [{roc_lo:.3f}, {roc_hi:.3f}]")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Bootstrap x1000)")
    plt.legend(loc="lower right")
    if save_prefix:
        plt.savefig(f"{save_prefix}_roc.png", dpi=300, bbox_inches="tight")
    plt.show()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC = {pr_mean:.3f} [{pr_lo:.3f}, {pr_hi:.3f}]")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (Bootstrap x1000)")
    plt.legend(loc="lower left")
    if save_prefix:
        plt.savefig(f"{save_prefix}_pr.png", dpi=300, bbox_inches="tight")
    plt.show()


def spike_1channel_roc_pr(folder: str,
         pattern: str = "*.csv",
         n_bootstrap: int = 1000,
         seed: int = 42,
         save_prefix: str | None = None):
    y_true, y_pred = load_all_csv(folder, pattern)
    stats = bootstrap_auc(y_true, y_pred, n_bootstrap=n_bootstrap, seed=seed)
    print(
        f"ROC AUC mean/95%CI: {stats[0]:.4f} [{stats[1]:.4f}, {stats[2]:.4f}]\n"
        f"PR  AUC mean/95%CI: {stats[3]:.4f} [{stats[4]:.4f}, {stats[5]:.4f}]"
    )
    plot_roc_pr(y_true, y_pred, stats, save_prefix=save_prefix)


if __name__ == "__main__":
    # merge_bird_results()
    # vis_birds_irr(result_csv_path='/data/BIRD/merged_labeled_and_results_20251208.csv', output_path='/data/BIRD/irr.png')

    vis_birds_auc(result_csv_path='/data/BIRD/merged_labeled_and_results_20251208.csv',
                  roc_out_path='/data/BIRD/roc.png', pr_out_path='/data/BIRD/pr.png')


    spike_1channel_roc_pr(folder="/data/SPIKE_localization/results_1channel/processed_1second_test/", pattern="pred_*.csv", n_bootstrap=100, seed=42,
         save_prefix="/data/SPIKE_localization/results_1channel/figures/processed_1second_test")


# echo "exxact@1" | sudo -S ~/miniconda3/envs/torchenv/bin/python vis_result_mo2.py