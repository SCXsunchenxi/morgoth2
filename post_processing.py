import os
from pathlib import Path

import numpy as np
import pandas as pd


def post_process_probabilities(prob, mask, gap_threshold=2, min_duration=20):
    """
    Smooth a 1D probability sequence by:
      1) Merging neighboring positive segments if the gap between them is small
         (<= gap_threshold), and filling the merged gap by linear interpolation.
      2) Removing short positive segments (<= min_duration) by zeroing them out.

    Args:
        prob: 1D numpy array of continuous values, shape [T]
        mask: 1D numpy array of {0,1}, shape [T]
        gap_threshold: max gap length (samples) to merge
        min_duration: segments shorter than or equal to this are removed

    Returns:
        processed_prob: 1D numpy array, shape [T]
    """
    prob = prob.astype(float, copy=True)
    mask = mask.astype(np.int8, copy=True)

    if prob.ndim != 1 or mask.ndim != 1 or prob.shape[0] != mask.shape[0]:
        raise ValueError("`prob` and `mask` must be 1D arrays with the same length.")

    def find_segments(binary):
        starts = np.where(np.diff(np.r_[0, binary]) == 1)[0]
        ends = np.where(np.diff(np.r_[binary, 0]) == -1)[0]
        return starts, ends

    starts, ends = find_segments(mask)
    if len(starts) == 0:
        return prob

    # Merge close segments and interpolate
    for i in range(len(ends) - 1):
        gap = starts[i + 1] - ends[i] - 1
        if gap <= gap_threshold:
            left = ends[i]
            right = starts[i + 1]

            mask[left:right + 1] = 1

            left_val = prob[left]
            right_val = prob[right]
            prob[left:right + 1] = np.linspace(left_val, right_val, right - left + 1)

    # Recompute segments after merging
    starts, ends = find_segments(mask)
    if len(starts) == 0:
        return prob

    # Remove short segments
    for s, e in zip(starts, ends):
        if (e - s + 1) <= min_duration:
            mask[s:e + 1] = 0
            prob[s:e + 1] = 0.0

    return prob


def process_csv_directory(
    input_dir,
    output_dir,
    prob_col="class_1_prob",
    pred_col="pred_class",
    target_class=1,
    gap_threshold=2,
    min_duration=20,
    update_other_probs_to_zero=True,
):
    """
    Apply post-processing to all CSV files in a directory.

    Required columns:
        - prob_col (default: 'class_1_prob')
        - pred_col (default: 'pred_class')

    Behavior:
        - Smooth prob_col using post_process_probabilities()
        - For rows where prob_col changed:
            * set pred_col = target_class if processed_prob > 0.5 else 0
            * if class_0_prob exists, set it to 1 - processed_prob
            * optionally set all other columns to 0 for those rows

    Writes processed CSVs to output_dir with the same filenames.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in: {input_path}")

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)

            if prob_col not in df.columns or pred_col not in df.columns:
                print(f"  [WARN] Missing required columns in {csv_file.name}. Skipping.")
                continue

            prob = df[prob_col].to_numpy(dtype=float)
            pred = df[pred_col].to_numpy()

            mask = (pred == target_class).astype(np.int8)

            processed_prob = post_process_probabilities(
                prob=prob,
                mask=mask,
                gap_threshold=gap_threshold,
                min_duration=min_duration,
            )

            changed = ~np.isclose(prob, processed_prob, rtol=1e-6, atol=1e-8)

            df.loc[changed, prob_col] = processed_prob[changed]

            new_pred = np.where(processed_prob > 0.5, target_class, 0)
            df.loc[changed, pred_col] = new_pred[changed]

            if "class_0_prob" in df.columns:
                df.loc[changed, "class_0_prob"] = 1.0 - processed_prob[changed]

            if update_other_probs_to_zero:
                keep = {"class_0_prob", prob_col, pred_col}
                other_cols = [c for c in df.columns if c not in keep]
                if other_cols:
                    df.loc[changed, other_cols] = 0

            out_file = output_path / csv_file.name
            df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file}")

        except Exception as exc:
            print(f"  [ERROR] Failed on {csv_file.name}: {exc}")
            continue

    print("Done processing all files.")


if __name__ == "__main__":
    process_csv_directory(
        input_dir="/path/to/continuous_results",
        output_dir="/path/to/post_processed_results",
        prob_col="class_1_prob",
        pred_col="pred_class",
        target_class=1,
        gap_threshold=2,
        min_duration=20,
        update_other_probs_to_zero=True,
    )