import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import ast

def get_result_file_sparcnettest(label_file,eeglevel_dir,output_csv):
    # Load the main table
    main_df = pd.read_csv(label_file)

    main_df=main_df[["file_name", "label ([other,seizure,lpd,gpd,lrda,grda])"]]

    # Find all csv files in the directory
    csv_files = glob.glob(os.path.join(eeglevel_dir, "*.csv"))

    print(f"Found {len(csv_files)} csv files.")

    for csv_file in csv_files:
        file_base = os.path.splitext(os.path.basename(csv_file))[0]

        # Get the last field of the filename split by "_"
        suffix = file_base.split("_")[-1]

        prob_col = f"{suffix}_eeglevel_probability"
        conf_col = f"{suffix}_eeglevel_confidence"

        df = pd.read_csv(csv_file)

        # Check that required columns exist
        required_cols = {"file_name", "probability", "confidence"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {csv_file}, missing columns: {required_cols - set(df.columns)}")
            continue

        # Keep only the needed columns and rename
        df = df[["file_name", "probability", "confidence"]].copy()
        df = df.rename(columns={
            "probability": prob_col,
            "confidence": conf_col
        })

        # Merge into the main table, keeping all rows
        main_df = main_df.merge(df, on="file_name", how="left")

        print(f"Merged {csv_file} -> {prob_col}, {conf_col}")

    def decide_validation(label):
        if isinstance(label, list) and len(label) > 0:
            return "no" if label[0] == max(label) else "yes"
        return "no"  # Prevent edge case exceptions

    main_df["label ([other,seizure,lpd,gpd,lrda,grda])"] = main_df["label ([other,seizure,lpd,gpd,lrda,grda])"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    main_df["used_for_validation"] = main_df["label ([other,seizure,lpd,gpd,lrda,grda])"].apply(decide_validation)
    main_df["source"] = "sparcnettest"

    # Save results
    main_df.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}")


def get_result_file_moe(label_file,eeglevel_dir,output_csv):
    # Load the main table
    label_df = pd.read_csv(label_file)
    label_df = label_df.rename(columns={
        "event": "file_name",
    })


    # Find all columns starting with "expert"
    expert_cols = [col for col in label_df.columns if col.startswith("expert")]

    print(f"Found {len(expert_cols)} expert columns")

    # Count occurrences of 0~5 for each row
    def count_labels(row):
        counts = [0] * 6  # Corresponding to 0~5
        for val in row:
            if pd.notna(val):
                val = int(val)
                if 0 <= val <= 5:
                    counts[val] += 1
        return counts

    # Generate the label column
    label_df["label ([other,seizure,lpd,gpd,lrda,grda])"] = label_df[expert_cols].apply(count_labels, axis=1)

    # Merge back into main_df (only file_name + label)
    main_df = label_df[["file_name", "label ([other,seizure,lpd,gpd,lrda,grda])"]]

    # Find all csv files in the directory
    csv_files = glob.glob(os.path.join(eeglevel_dir, "*.csv"))

    print(f"Found {len(csv_files)} csv files.")

    for csv_file in csv_files:
        file_base = os.path.splitext(os.path.basename(csv_file))[0]

        # Get the last field of the filename split by "_"
        suffix = file_base.split("_")[-1]

        prob_col = f"{suffix}_eeglevel_probability"
        conf_col = f"{suffix}_eeglevel_confidence"

        df = pd.read_csv(csv_file)

        # Check that required columns exist
        required_cols = {"file_name", "probability", "confidence"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {csv_file}, missing columns: {required_cols - set(df.columns)}")
            continue

        # Keep only the needed columns and rename
        df = df[["file_name", "probability", "confidence"]].copy()
        df = df.rename(columns={
            "probability": prob_col,
            "confidence": conf_col
        })

        # Merge into the main table, keeping all rows
        main_df = main_df.merge(df, on="file_name", how="left")

        print(f"Merged {csv_file} -> {prob_col}, {conf_col}")

    def decide_validation(label):
        if isinstance(label, list) and len(label) > 0:
            return "no" if label[0] == max(label) else "yes"
        return "no"  # Prevent edge case exceptions

    main_df["used_for_validation"] = main_df["label ([other,seizure,lpd,gpd,lrda,grda])"].apply(decide_validation)
    main_df["source"] = "moe"

    # Save results
    main_df.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}")


def get_result_file_representative(label_file,eeglevel_dir,output_csv):
    # Load the main table

    label_df = pd.read_csv(label_file)
    label_df = label_df[label_df["source"] == "representative"].copy()

    def to_one_hot(x):
        vec = [0] * 6
        if pd.notna(x):
            x = int(x)
            if 0 <= x <= 5:
                vec[x] = 1
        return vec

    label_df["label ([other,seizure,lpd,gpd,lrda,grda])"]= label_df["label"].apply(to_one_hot)

    # Merge back into main_df (only file_name + label)
    main_df = label_df[["file_name", "label ([other,seizure,lpd,gpd,lrda,grda])"]]

    # Find all csv files in the directory
    csv_files = glob.glob(os.path.join(eeglevel_dir, "*.csv"))

    print(f"Found {len(csv_files)} csv files.")

    for csv_file in csv_files:
        file_base = os.path.splitext(os.path.basename(csv_file))[0]

        # Get the last field of the filename split by "_"
        suffix = file_base.split("_")[-1]

        prob_col = f"{suffix}_eeglevel_probability"
        conf_col = f"{suffix}_eeglevel_confidence"

        df = pd.read_csv(csv_file)

        # Check that required columns exist
        required_cols = {"file_name", "probability", "confidence"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {csv_file}, missing columns: {required_cols - set(df.columns)}")
            continue

        # Keep only the needed columns and rename
        df = df[["file_name", "probability", "confidence"]].copy()
        df = df.rename(columns={
            "probability": prob_col,
            "confidence": conf_col
        })

        # Merge into the main table, keeping all rows
        main_df = main_df.merge(df, on="file_name", how="left")

        print(f"Merged {csv_file} -> {prob_col}, {conf_col}")


    main_df["used_for_validation"] = "yes"
    main_df["source"] = "representative"

    # Save results
    main_df.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}")

def get_result_file(result_files, output_csv):
    dfs = []

    for file in result_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded: {file}, shape={df.shape}")
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    if len(dfs) == 0:
        print("No valid files found.")
        return

    merged_df = pd.concat(dfs, axis=0, ignore_index=True)

    merged_df.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}, shape={merged_df.shape}")



def plot_roc_pr(result_csv_path, save_path, n_bootstrap=1000):
    import os, ast
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    tasks = ["SEIZURE", "LPD", "GPD", "LRDA", "GRDA"]
    task_idx = {
        "SEIZURE": 1,
        "LPD": 2,
        "GPD": 3,
        "LRDA": 4,
        "GRDA": 5
    }

    label_col = "label ([other,seizure,lpd,gpd,lrda,grda])"

    # ========= Global font scaling =========
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "legend.fontsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    # ========= Load data =========
    df = pd.read_csv(result_csv_path)

    df[label_col] = df[label_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df = df[df["used_for_validation"] == "yes"].copy()
    print("Data size:", len(df))

    def is_target(label, idx):
        return int(label[idx] == max(label))

    def bootstrap_curve(y_true, y_score, curve_type="roc", n_bootstrap=1000):
        xs = np.linspace(0, 1, 200)
        curves, aucs = [], []
        n = len(y_true)

        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(n, n, replace=True)
            y_t = y_true[boot_idx]
            y_s = y_score[boot_idx]

            if len(np.unique(y_t)) < 2:
                continue

            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(y_t, y_s)
                interp = np.interp(xs, fpr, tpr)
                interp[0], interp[-1] = 0.0, 1.0
                curves.append(interp)
                aucs.append(auc(fpr, tpr))

            else:
                prec, rec, _ = precision_recall_curve(y_t, y_s)
                interp = np.interp(xs, rec[::-1], prec[::-1])
                curves.append(interp)
                aucs.append(auc(rec, prec))

        if len(curves) == 0:
            return xs, None, None, None, np.nan

        curves = np.array(curves)
        return (
            xs,
            curves.mean(axis=0),
            np.percentile(curves, 2.5, axis=0),
            np.percentile(curves, 97.5, axis=0),
            float(np.mean(aucs)),
        )

    # ========= Create figure =========
    fig, axes = plt.subplots(2, 5, figsize=(28, 10))

    for col_idx, task in enumerate(tasks):
        idx = task_idx[task]

        y_true = np.array([is_target(l, idx) for l in df[label_col]])

        prob_col = f"{task}_eeglevel_probability"
        conf_col = f"{task}_eeglevel_confidence"

        if prob_col not in df.columns:
            continue

        y_prob = pd.to_numeric(df[prob_col], errors="coerce").values
        y_conf = pd.to_numeric(df[conf_col], errors="coerce").values

        mask_prob = ~np.isnan(y_prob)
        mask_conf = ~np.isnan(y_conf)

        # ===== ROC =====
        ax = axes[0, col_idx]

        for y_score, mask, name in [
            (y_prob, mask_prob, "prob"),
            (y_conf, mask_conf, "conf"),
        ]:
            if mask.sum() == 0:
                continue

            xs, mean, low, up, auc_mean = bootstrap_curve(
                y_true[mask], y_score[mask], "roc", n_bootstrap
            )

            if mean is None:
                continue

            ax.plot(xs, mean,
                    label=f"{task} {name} (AUC={auc_mean:.3f})")
            ax.fill_between(xs, low, up, alpha=0.2)

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ===== PR =====
        ax = axes[1, col_idx]

        for y_score, mask, name in [
            (y_prob, mask_prob, "prob"),
            (y_conf, mask_conf, "conf"),
        ]:
            if mask.sum() == 0:
                continue

            xs, mean, low, up, auc_mean = bootstrap_curve(
                y_true[mask], y_score[mask], "pr", n_bootstrap
            )

            if mean is None:
                continue

            ax.plot(xs, mean,
                    label=f"{task} {name} (AUC={auc_mean:.3f})")
            ax.fill_between(xs, low, up, alpha=0.2)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    get_result_file_sparcnettest(label_file="/data/IIIC/sparcnet_test_list.csv",
                    eeglevel_dir="/data/IIIC_EEG_level/event_res/IIIC_segments_raw_EEGlevel",
                    output_csv = "/data/IIIC_EEG_level/event_res/IIIC_segments_raw_EEGlevel/sparcnet_testset_eeglevel_results.csv"
    )

    get_result_file_moe(label_file="/data/MoE/labels/IIIC_merged_experts.csv",
                                 eeglevel_dir="/data/IIIC_EEG_level/event_res/MoE_segments_raw_EEGlevel",
                                 output_csv="/data/IIIC_EEG_level/event_res/MoE_segments_raw_EEGlevel/moe_eeglevel_results.csv"
                                 )

    get_result_file_representative(label_file="/data/IIIC_EEG_level/event_res/training_list.csv",
                                 eeglevel_dir="/data/IIIC_EEG_level/event_res/representative_res_EEGlevel",
                                 output_csv="/data/IIIC_EEG_level/event_res/representative_res_EEGlevel/representative_eeglevel_results.csv"
                                 )

    get_result_file(result_files=[
        "/data/IIIC_EEG_level/event_res/IIIC_segments_raw_EEGlevel/sparcnet_testset_eeglevel_results.csv",
        "/data/IIIC_EEG_level/event_res/MoE_segments_raw_EEGlevel/moe_eeglevel_results.csv",
        "/data/IIIC_EEG_level/event_res/representative_res_EEGlevel/representative_eeglevel_results.csv"
                                  ],
        output_csv="/data/IIIC_EEG_level/eeglevel_results.csv")

    plot_roc_pr(result_csv_path = "/data/IIIC_EEG_level/eeglevel_results.csv", save_path="/data/IIIC_EEG_level/eeglevel_results.png",n_bootstrap=100)

# echo "exxact@1" | sudo -S ~/miniconda3/envs/torchenv/bin/python vis_IIIC_EEG_level_results.py