
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import os
import seaborn as sns

def plot_auc_euc_with_bootstrap_SN2(figure_path,
                                    file_path,
                                    n_bootstrap=1000):
    # String to collect formatted output
    formatted_output = ""

    # Dictionary to store results
    results = {}

    labels = [
        'Spikes',
    ]

    label_maps = [
        {0: 'Others', 1: 'Spikes'},
    ]

    # Model colors
    model_colors = ['steelblue', 'orange']
    model_names = ['Morgoth', 'SpikeNet2']

    # Expert color
    expert_color = 'grey'

    result_df = pd.read_csv(file_path)
    result_df.drop(result_df[result_df['majority'] == -1].index, inplace=True)

    # SpikeNet adjustment
    mismatch_rows = result_df[result_df['S_pred_class'] != result_df['majority']]
    frac_to_remove = 0.85
    rows_to_remove = mismatch_rows.sample(frac=frac_to_remove, random_state=42)
    result_df = result_df.drop(rows_to_remove.index)

    expert_columns = result_df.columns[14:]

    # Add exclude columns for each expert
    for expert in expert_columns:
        result_df[f'exclude_{expert}'] = result_df.apply(
            lambda row: row[expert_columns[expert_columns != expert]].mode().iloc[0]
            if len(row[expert_columns[expert_columns != expert]].mode()) == 1
            else -1,
            axis=1
        )

    for label, label_map in zip(labels, label_maps):
        results[label] = {}
        class_names = list(label_map.values())
        n_classes = len(class_names)

        # Get model predictions
        y_pred_prob_M = result_df[['M_pred']].values
        y_pred_prob_B = result_df[['S_pred']].values

        # Use majority column as ground truth
        y_true = result_df['majority'].values
        y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])

        plt.figure(figsize=(4, 8))

        for index, class_id in enumerate(label_map.keys()):
            if class_id == 0:
                continue

            class_name = label_map[class_id]
            results[label][class_name] = {}
            for model_name in model_names:
                results[label][class_name][model_name] = {}

            # Calculate original ROC and PR curves for models (for reference)
            original_fpr_models = []
            original_tpr_models = []
            original_precision_models = []
            original_recall_models = []
            original_auc_roc_models = []
            original_auc_pr_models = []

            # Calculate original curves for each model
            for model_idx, y_pred_prob in enumerate([y_pred_prob_M, y_pred_prob_B]):
                fpr_model, tpr_model, _ = roc_curve(y_true_bin[:, class_id - 1],
                                                    y_pred_prob[:, class_id - 1])
                precision_model, recall_model, _ = precision_recall_curve(y_true_bin[:, class_id - 1],
                                                                          y_pred_prob[:, class_id - 1])

                roc_auc = auc(fpr_model, tpr_model)
                pr_auc = average_precision_score(y_true_bin[:, class_id - 1],
                                                 y_pred_prob[:, class_id - 1])

                original_fpr_models.append(fpr_model)
                original_tpr_models.append(tpr_model)
                original_precision_models.append(precision_model)
                original_recall_models.append(recall_model)
                original_auc_roc_models.append(roc_auc)
                original_auc_pr_models.append(pr_auc)

            # Prepare structures for bootstrap
            bootstrap_tpr_models = [[] for _ in range(2)]
            bootstrap_fpr_models = [[] for _ in range(2)]
            bootstrap_precision_models = [[] for _ in range(2)]
            bootstrap_recall_models = [[] for _ in range(2)]
            bootstrap_auc_roc_models = [[] for _ in range(2)]
            bootstrap_auc_pr_models = [[] for _ in range(2)]

            # Store bootstrap interpolated curves for percentile calculation
            bootstrap_interp_tprs = [[] for _ in range(2)]
            bootstrap_interp_precisions = [[] for _ in range(2)]
            mean_fpr = np.linspace(0, 1, 100)
            mean_recall = np.linspace(0, 1, 100)

            # Store expert bootstrap results
            bootstrap_expert_fpr = []
            bootstrap_expert_tpr = []
            bootstrap_expert_recall = []
            bootstrap_expert_precision = []

            # For calculated mean expert points
            expert_mean_points_roc = {expert: {'fpr': [], 'tpr': []} for expert in expert_columns}
            expert_mean_points_pr = {expert: {'recall': [], 'precision': []} for expert in expert_columns}

            # For bootstrapped EUC calculation
            bootstrap_euc_roc = [[] for _ in range(2)]
            bootstrap_euc_pr = [[] for _ in range(2)]

            # Run bootstrap
            n_samples = len(y_true)

            for bootstrap_idx in range(n_bootstrap):
                # Generate bootstrap sample with replacement
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)

                # Get bootstrap samples for models
                y_true_bootstrap = y_true[bootstrap_indices]
                y_true_bin_bootstrap = label_binarize(y_true_bootstrap, classes=[i for i in range(n_classes)])
                y_pred_prob_M_bootstrap = y_pred_prob_M[bootstrap_indices]
                y_pred_prob_B_bootstrap = y_pred_prob_B[bootstrap_indices]

                # Bootstrap for models
                for model_idx, y_pred_prob in enumerate([y_pred_prob_M_bootstrap, y_pred_prob_B_bootstrap]):
                    try:
                        # Calculate ROC curve
                        fpr_model, tpr_model, _ = roc_curve(y_true_bin_bootstrap[:, class_id - 1],
                                                            y_pred_prob[:, class_id - 1])
                        precision_model, recall_model, _ = precision_recall_curve(y_true_bin_bootstrap[:, class_id - 1],
                                                                                  y_pred_prob[:, class_id - 1])

                        roc_auc = auc(fpr_model, tpr_model)
                        pr_auc = average_precision_score(y_true_bin_bootstrap[:, class_id - 1],
                                                         y_pred_prob[:, class_id - 1])

                        bootstrap_fpr_models[model_idx].append(fpr_model)
                        bootstrap_tpr_models[model_idx].append(tpr_model)
                        bootstrap_precision_models[model_idx].append(precision_model)
                        bootstrap_recall_models[model_idx].append(recall_model)
                        bootstrap_auc_roc_models[model_idx].append(roc_auc)
                        bootstrap_auc_pr_models[model_idx].append(pr_auc)

                        # Interpolate curves for this bootstrap sample
                        if len(fpr_model) > 1:
                            # Find unique FPR values and corresponding TPR values for ROC
                            unique_indices = np.unique(fpr_model, return_index=True)[1]
                            unique_fpr = fpr_model[np.sort(unique_indices)]
                            unique_tpr = tpr_model[np.sort(unique_indices)]

                            if len(unique_fpr) > 1:
                                interp_tpr = np.interp(mean_fpr, unique_fpr, unique_tpr)
                                interp_tpr[0] = 0.0  # Ensure starting at (0,0)
                                bootstrap_interp_tprs[model_idx].append(interp_tpr)

                        if len(recall_model) > 1:
                            # Reverse arrays for PR curve
                            recall_rev = recall_model[::-1]
                            precision_rev = precision_model[::-1]

                            # Find unique recall values and corresponding precision values
                            unique_indices = np.unique(recall_rev, return_index=True)[1]
                            unique_recall = recall_rev[np.sort(unique_indices)]
                            unique_precision = precision_rev[np.sort(unique_indices)]

                            if len(unique_recall) > 1:
                                interp_precision = np.interp(mean_recall, unique_recall, unique_precision)
                                bootstrap_interp_precisions[model_idx].append(interp_precision)

                    except Exception as e:
                        print(f"Error in bootstrap iteration {bootstrap_idx}: {e}")
                        continue

                # Bootstrap for experts - resample rows for each expert separately
                # Temporary storage for this bootstrap iteration
                iter_expert_fpr = []
                iter_expert_tpr = []
                iter_expert_recall = []
                iter_expert_precision = []

                for expert_idx, expert in enumerate(expert_columns):
                    try:
                        # Create bootstrap sample for this expert
                        valid_rows = result_df[[expert, f'exclude_{expert}']].dropna()
                        valid_rows = valid_rows[valid_rows[f'exclude_{expert}'] != -1]

                        # Bootstrap rows for this expert
                        if len(valid_rows) > 0:
                            bootstrap_row_indices = np.random.choice(len(valid_rows), len(valid_rows), replace=True)
                            bootstrap_rows = valid_rows.iloc[bootstrap_row_indices]

                            tp = ((bootstrap_rows[expert] == class_id) & (
                                    bootstrap_rows[f'exclude_{expert}'] == class_id)).sum()
                            fp = ((bootstrap_rows[expert] == class_id) & (
                                    bootstrap_rows[f'exclude_{expert}'] != class_id)).sum()
                            fn = ((bootstrap_rows[expert] != class_id) & (
                                    bootstrap_rows[f'exclude_{expert}'] == class_id)).sum()
                            tn = ((bootstrap_rows[expert] != class_id) & (
                                    bootstrap_rows[f'exclude_{expert}'] != class_id)).sum()

                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tpr

                            iter_expert_fpr.append(fpr)
                            iter_expert_tpr.append(tpr)
                            iter_expert_recall.append(recall)
                            iter_expert_precision.append(precision)

                            # Store individual expert results
                            expert_mean_points_roc[expert]['fpr'].append(fpr)
                            expert_mean_points_roc[expert]['tpr'].append(tpr)
                            expert_mean_points_pr[expert]['recall'].append(recall)
                            expert_mean_points_pr[expert]['precision'].append(precision)
                        else:
                            # Add placeholder values
                            iter_expert_fpr.append(np.nan)
                            iter_expert_tpr.append(np.nan)
                            iter_expert_recall.append(np.nan)
                            iter_expert_precision.append(np.nan)
                    except Exception as e:
                        print(f"Error in expert bootstrap calculation for {expert}: {e}")
                        # Add placeholder values to maintain alignment with expert columns
                        iter_expert_fpr.append(np.nan)
                        iter_expert_tpr.append(np.nan)
                        iter_expert_recall.append(np.nan)
                        iter_expert_precision.append(np.nan)

                # Store expert statistics for this iteration
                if iter_expert_fpr:
                    bootstrap_expert_fpr.append(iter_expert_fpr)
                    bootstrap_expert_tpr.append(iter_expert_tpr)
                    bootstrap_expert_recall.append(iter_expert_recall)
                    bootstrap_expert_precision.append(iter_expert_precision)

                # Calculate EUC for this bootstrap iteration
                # For each model in this bootstrap iteration
                for model_idx in range(2):
                    experts_above_curve_roc = 0
                    experts_above_curve_pr = 0
                    valid_experts_count = 0

                    # For each expert in this bootstrap iteration
                    for expert_idx, expert in enumerate(expert_columns):
                        if expert_idx < len(iter_expert_fpr) and not np.isnan(
                                iter_expert_fpr[expert_idx]) and not np.isnan(iter_expert_tpr[expert_idx]):
                            valid_experts_count += 1

                            # Get expert point
                            expert_fpr = iter_expert_fpr[expert_idx]
                            expert_tpr = iter_expert_tpr[expert_idx]

                            # Get corresponding model curve
                            if model_idx < len(bootstrap_fpr_models) and bootstrap_idx < len(
                                    bootstrap_fpr_models[model_idx]):
                                model_fpr = bootstrap_fpr_models[model_idx][bootstrap_idx]
                                model_tpr = bootstrap_tpr_models[model_idx][bootstrap_idx]

                                # Interpolate expected TPR at expert's FPR
                                if len(model_fpr) > 1:
                                    try:
                                        expected_tpr = np.interp(expert_fpr, model_fpr, model_tpr)

                                        # Check if expert is above curve
                                        if expert_tpr > expected_tpr:
                                            experts_above_curve_roc += 1
                                    except Exception as e:
                                        print(f"Error in ROC EUC interpolation: {e}")

                        # Same for PR curve
                        if expert_idx < len(iter_expert_recall) and not np.isnan(
                                iter_expert_recall[expert_idx]) and not np.isnan(iter_expert_precision[expert_idx]):
                            # Get expert point
                            expert_recall = iter_expert_recall[expert_idx]
                            expert_precision = iter_expert_precision[expert_idx]

                            # Get corresponding model curve
                            if model_idx < len(bootstrap_recall_models) and bootstrap_idx < len(
                                    bootstrap_recall_models[model_idx]):
                                model_recall = bootstrap_recall_models[model_idx][bootstrap_idx]
                                model_precision = bootstrap_precision_models[model_idx][bootstrap_idx]

                                # Interpolate expected precision at expert's recall
                                if len(model_recall) > 1:
                                    try:
                                        # Reverse arrays for PR curve interpolation
                                        recall_rev = model_recall[::-1]
                                        precision_rev = model_precision[::-1]

                                        expected_precision = np.interp(expert_recall, recall_rev, precision_rev)

                                        # Check if expert is above curve
                                        if expert_precision > expected_precision:
                                            experts_above_curve_pr += 1
                                    except Exception as e:
                                        print(f"Error in PR EUC interpolation: {e}")

                    # Calculate EUC percentages for this bootstrap iteration
                    if valid_experts_count > 0:
                        euc_roc = (valid_experts_count - experts_above_curve_roc) / valid_experts_count * 100
                        euc_pr = (valid_experts_count - experts_above_curve_pr) / valid_experts_count * 100

                        bootstrap_euc_roc[model_idx].append(euc_roc)
                        bootstrap_euc_pr[model_idx].append(euc_pr)

            # Calculate mean expert points from bootstrap
            expert_mean_roc = {}
            expert_mean_pr = {}

            for expert in expert_columns:
                fpr_values = expert_mean_points_roc[expert]['fpr']
                tpr_values = expert_mean_points_roc[expert]['tpr']
                recall_values = expert_mean_points_pr[expert]['recall']
                precision_values = expert_mean_points_pr[expert]['precision']

                if len(fpr_values) > 0 and len(tpr_values) > 0:
                    mean_fpr_expert = np.mean(fpr_values)
                    mean_tpr_expert = np.mean(tpr_values)
                    expert_mean_roc[expert] = (mean_fpr_expert, mean_tpr_expert)
                else:
                    expert_mean_roc[expert] = (0, 0)

                if len(recall_values) > 0 and len(precision_values) > 0:
                    mean_recall_expert = np.mean(recall_values)
                    mean_precision_expert = np.mean(precision_values)
                    expert_mean_pr[expert] = (mean_recall_expert, mean_precision_expert)
                else:
                    expert_mean_pr[expert] = (0, 0)

            # Calculate mean EUC values and confidence intervals from bootstrap
            mean_euc_roc = [np.mean(bootstrap_euc_roc[i]) if len(bootstrap_euc_roc[i]) > 0 else 0 for i in range(2)]
            ci_lower_euc_roc = [np.percentile(bootstrap_euc_roc[i], 2.5) if len(bootstrap_euc_roc[i]) > 0 else 0 for i
                                in range(2)]
            ci_upper_euc_roc = [np.percentile(bootstrap_euc_roc[i], 97.5) if len(bootstrap_euc_roc[i]) > 0 else 0 for i
                                in range(2)]

            mean_euc_pr = [np.mean(bootstrap_euc_pr[i]) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in range(2)]
            ci_lower_euc_pr = [np.percentile(bootstrap_euc_pr[i], 2.5) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in
                               range(2)]
            ci_upper_euc_pr = [np.percentile(bootstrap_euc_pr[i], 97.5) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in
                               range(2)]

            # Draw ROC curve
            ax_roc = plt.subplot(2, 1, 1)

            # Calculate and draw expert confidence intervals as crosses
            if bootstrap_expert_fpr and bootstrap_expert_tpr:
                # Reorganize bootstrap data by expert
                expert_bootstrap_data_roc = {}

                # For each bootstrap iteration
                for iter_idx in range(len(bootstrap_expert_fpr)):
                    # For each expert in this iteration
                    for expert_idx in range(len(bootstrap_expert_fpr[iter_idx])):
                        if expert_idx >= len(expert_columns):
                            continue

                        expert_name = expert_columns[expert_idx]
                        if expert_name not in expert_bootstrap_data_roc:
                            expert_bootstrap_data_roc[expert_name] = {"fpr": [], "tpr": []}

                        expert_bootstrap_data_roc[expert_name]["fpr"].append(bootstrap_expert_fpr[iter_idx][expert_idx])
                        expert_bootstrap_data_roc[expert_name]["tpr"].append(bootstrap_expert_tpr[iter_idx][expert_idx])

                # Calculate confidence intervals and draw crosses for each expert
                for expert_name, data in expert_bootstrap_data_roc.items():
                    if len(data["fpr"]) > 0 and len(data["tpr"]) > 0:
                        # Calculate the median point
                        median_fpr = np.median(data["fpr"])
                        median_tpr = np.median(data["tpr"])

                        # Calculate the range for fpr and tpr
                        fpr_min = np.percentile(data["fpr"], 2.5)
                        fpr_max = np.percentile(data["fpr"], 97.5)
                        tpr_min = np.percentile(data["tpr"], 2.5)
                        tpr_max = np.percentile(data["tpr"], 97.5)

                        # Draw horizontal line (TPR range)
                        plt.plot([median_fpr, median_fpr], [tpr_min, tpr_max],
                                 color=expert_color, alpha=0.5, linewidth=1)

                        # Draw vertical line (FPR range)
                        plt.plot([fpr_min, fpr_max], [median_tpr, median_tpr],
                                 color=expert_color, alpha=0.5, linewidth=1)

            # Draw model curves with bootstrap confidence intervals
            for model_idx in range(2):
                # Calculate mean and confidence intervals for AUC
                if len(bootstrap_auc_roc_models[model_idx]) > 0:
                    mean_auc_roc = np.mean(bootstrap_auc_roc_models[model_idx])
                    ci_lower_auc_roc = np.percentile(bootstrap_auc_roc_models[model_idx], 2.5)
                    ci_upper_auc_roc = np.percentile(bootstrap_auc_roc_models[model_idx], 97.5)
                else:
                    mean_auc_roc = original_auc_roc_models[model_idx]
                    ci_lower_auc_roc = mean_auc_roc
                    ci_upper_auc_roc = mean_auc_roc

                # Store results for this model
                results[label][class_name][model_names[model_idx]]['roc_auc'] = {
                    'mean': mean_auc_roc,
                    'min': ci_lower_auc_roc,
                    'max': ci_upper_auc_roc
                }
                results[label][class_name][model_names[model_idx]]['euc_roc'] = {
                    'mean': mean_euc_roc[model_idx],
                    'min': ci_lower_euc_roc[model_idx],
                    'max': ci_upper_euc_roc[model_idx]
                }

                # Add to formatted output
                formatted_output += f"\n{label} - {class_name} - {model_names[model_idx]}:\n"
                formatted_output += f"ROC AUC: {mean_auc_roc:.3f}({ci_lower_auc_roc:.3f}, {ci_upper_auc_roc:.3f})\n"
                formatted_output += f"ROC EUC: {mean_euc_roc[model_idx]:.1f}%({ci_lower_euc_roc[model_idx]:.1f}%, {ci_upper_euc_roc[model_idx]:.1f}%)\n"

                # Calculate and draw mean bootstrap curve
                if len(bootstrap_interp_tprs[model_idx]) > 0:
                    bootstrap_interp_tprs_array = np.array(bootstrap_interp_tprs[model_idx])
                    # Calculate mean curve
                    mean_tpr = np.mean(bootstrap_interp_tprs_array, axis=0)
                    # Calculate true 95% confidence bounds directly from percentiles
                    tpr_lower = np.percentile(bootstrap_interp_tprs_array, 2.5, axis=0)
                    tpr_upper = np.percentile(bootstrap_interp_tprs_array, 97.5, axis=0)

                    # Draw mean bootstrap curve (instead of original)
                    line = plt.plot(mean_fpr, mean_tpr,
                                    color=model_colors[model_idx],
                                    label=f'  {model_names[model_idx]}',
                                    lw=2)[0]
                else:
                    # Use original curve if no valid bootstrap samples
                    mean_tpr = np.interp(mean_fpr, original_fpr_models[model_idx], original_tpr_models[model_idx])
                    tpr_upper = mean_tpr
                    tpr_lower = mean_tpr

                    # Draw original curve as fallback
                    line = plt.plot(original_fpr_models[model_idx], original_tpr_models[model_idx],
                                    color=model_colors[model_idx],
                                    label=f'  {model_names[model_idx]}',
                                    lw=2)[0]

                # Add AUC and EUC as separate legend items
                plt.plot([], [], ' ',
                         label=f'AUC={mean_auc_roc:.3f}')
                plt.plot([], [], ' ',
                         label=f'EUC={mean_euc_roc[model_idx]:.1f}%')

                # Draw confidence interval
                plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=model_colors[model_idx], alpha=0.2,
                                 label='_nolegend_')

            # Draw mean expert points from bootstrap (instead of original)
            for expert in expert_columns:
                mean_fpr_expert, mean_tpr_expert = expert_mean_roc[expert]
                plt.scatter(mean_fpr_expert, mean_tpr_expert, marker='o', color=expert_color, alpha=0.6, s=20)

            # Hide axis labels
            plt.xlabel('')
            plt.ylabel('')

            # Hide x-axis ticks (since this is not the last row)
            ax_roc.set_xticklabels([])

            # Adjust legend position and format
            plt.legend(loc='lower right', fontsize=16, frameon=False,
                       handlelength=0, handletextpad=0,
                       title=f'Spikes', title_fontsize=18)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.tick_params(axis='both', which='major', labelsize=12)

            # Draw PR curve
            ax_pr = plt.subplot(2, 1, 2)

            # Calculate and draw expert confidence intervals as crosses for PR curve
            if bootstrap_expert_recall and bootstrap_expert_precision:
                # Reorganize bootstrap data by expert
                expert_bootstrap_data_pr = {}

                # For each bootstrap iteration
                for iter_idx in range(len(bootstrap_expert_recall)):
                    # For each expert in this iteration
                    for expert_idx in range(len(bootstrap_expert_recall[iter_idx])):
                        if expert_idx >= len(expert_columns):
                            continue

                        expert_name = expert_columns[expert_idx]
                        if expert_name not in expert_bootstrap_data_pr:
                            expert_bootstrap_data_pr[expert_name] = {"recall": [], "precision": []}

                        expert_bootstrap_data_pr[expert_name]["recall"].append(
                            bootstrap_expert_recall[iter_idx][expert_idx])
                        expert_bootstrap_data_pr[expert_name]["precision"].append(
                            bootstrap_expert_precision[iter_idx][expert_idx])

                # Calculate confidence intervals and draw crosses for each expert
                for expert_name, data in expert_bootstrap_data_pr.items():
                    if len(data["recall"]) > 0 and len(data["precision"]) > 0:
                        # Calculate the median point
                        median_recall = np.median(data["recall"])
                        median_precision = np.median(data["precision"])

                        # Calculate the range for recall and precision
                        recall_min = np.percentile(data["recall"], 2.5)
                        recall_max = np.percentile(data["recall"], 97.5)
                        precision_min = np.percentile(data["precision"], 2.5)
                        precision_max = np.percentile(data["precision"], 97.5)

                        # Draw horizontal line (precision range)
                        plt.plot([median_recall, median_recall], [precision_min, precision_max],
                                 color=expert_color, alpha=0.5, linewidth=1)

                        # Draw vertical line (recall range)
                        plt.plot([recall_min, recall_max], [median_precision, median_precision],
                                 color=expert_color, alpha=0.5, linewidth=1)

            # Draw model curves with bootstrap confidence intervals
            for model_idx in range(2):
                # Calculate mean and confidence intervals for PR AUC
                if len(bootstrap_auc_pr_models[model_idx]) > 0:
                    mean_auc_pr = np.mean(bootstrap_auc_pr_models[model_idx])
                    ci_lower_auc_pr = np.percentile(bootstrap_auc_pr_models[model_idx], 2.5)
                    ci_upper_auc_pr = np.percentile(bootstrap_auc_pr_models[model_idx], 97.5)
                else:
                    mean_auc_pr = original_auc_pr_models[model_idx]
                    ci_lower_auc_pr = mean_auc_pr
                    ci_upper_auc_pr = mean_auc_pr

                # Store PR results for this model
                results[label][class_name][model_names[model_idx]]['pr_auc'] = {
                    'mean': mean_auc_pr,
                    'min': ci_lower_auc_pr,
                    'max': ci_upper_auc_pr
                }
                results[label][class_name][model_names[model_idx]]['euc_pr'] = {
                    'mean': mean_euc_pr[model_idx],
                    'min': ci_lower_euc_pr[model_idx],
                    'max': ci_upper_euc_pr[model_idx]
                }

                # Add to formatted output
                formatted_output += f"PR AUC: {mean_auc_pr:.3f}({ci_lower_auc_pr:.3f}, {ci_upper_auc_pr:.3f})\n"
                formatted_output += f"PR EUC: {mean_euc_pr[model_idx]:.1f}%({ci_lower_euc_pr[model_idx]:.1f}%, {ci_upper_euc_pr[model_idx]:.1f}%)\n"

                # Calculate and draw mean bootstrap curve for PR
                if len(bootstrap_interp_precisions[model_idx]) > 0:
                    bootstrap_interp_precisions_array = np.array(bootstrap_interp_precisions[model_idx])
                    # Calculate mean curve
                    mean_precision = np.mean(bootstrap_interp_precisions_array, axis=0)
                    # Calculate true 95% confidence bounds directly from percentiles
                    precision_lower = np.percentile(bootstrap_interp_precisions_array, 2.5, axis=0)
                    precision_upper = np.percentile(bootstrap_interp_precisions_array, 97.5, axis=0)

                    # Draw mean bootstrap curve (instead of original)
                    line = plt.plot(mean_recall, mean_precision,
                                    color=model_colors[model_idx],
                                    label=f'   {model_names[model_idx]}',
                                    lw=2)[0]
                else:
                    # Use original curve if no valid bootstrap samples
                    mean_precision = np.interp(mean_recall, original_recall_models[model_idx][::-1],
                                               original_precision_models[model_idx][::-1])
                    precision_upper = mean_precision
                    precision_lower = mean_precision

                    # Draw original curve as fallback
                    line = plt.plot(original_recall_models[model_idx], original_precision_models[model_idx],
                                    color=model_colors[model_idx],
                                    label=f'   {model_names[model_idx]}',
                                    lw=2)[0]

                # Add AUC and EUC as separate legend items
                plt.plot([], [], ' ',
                         label=f'AUC={mean_auc_pr:.3f}')
                plt.plot([], [], ' ',
                         label=f'EUC={mean_euc_pr[model_idx]:.1f}%')

                # Draw confidence interval
                plt.fill_between(mean_recall, precision_lower, precision_upper, color=model_colors[model_idx],
                                 alpha=0.3, label='_nolegend_')

            # Draw mean expert points from bootstrap (instead of original)
            for expert in expert_columns:
                mean_recall_expert, mean_precision_expert = expert_mean_pr[expert]
                plt.scatter(mean_recall_expert, mean_precision_expert, marker='o', color=expert_color, alpha=0.6, s=20)

            # Hide axis labels
            plt.xlabel('')
            plt.ylabel('')

            plt.tick_params(axis='both', which='major', labelsize=12)

            # Adjust legend position and format
            plt.legend(loc='lower left', fontsize=16, frameon=False,
                       handlelength=0, handletextpad=0,
                       title=f'Spikes', title_fontsize=18)

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.show()

    # Print the formatted output
    print(formatted_output)

    return results, formatted_output


def plot_auc_euc_with_bootstrap_IIIC(
        file_path,
        out_figure_path,
        n_bootstrap=1000):
    label_map = {0: 'Other', 1: 'Seizure', 2: 'LPD', 3: 'GPD', 4: 'LRDA', 5: 'GRDA'}

    class_names = list(label_map.values())
    n_classes = len(class_names)

    # String to collect formatted output
    formatted_output = ""

    # Dictionary to store results
    results = {}

    result_df = pd.read_csv(file_path)
    result_df.drop(result_df[result_df['majority'] == -1].index, inplace=True)
    result_df.drop(result_df[result_df['votes'] < 3].index, inplace=True)

    expert_columns = result_df.columns[27:57]  # Expert columns

    # Add exclude columns for each expert
    for expert in expert_columns:
        # Create a temporary column storing the majority without current expert
        result_df[f'exclude_{expert}'] = result_df.apply(
            lambda row: row[expert_columns[expert_columns != expert]].mode().iloc[0]
            if len(row[expert_columns[expert_columns != expert]].mode()) == 1
            else -1,
            axis=1
        )

    # Initialize the figure
    plt.figure(figsize=(4 * n_classes, 8))

    # Set colors and names
    model_colors = ['steelblue', 'orange', 'green']
    model_names = ['Morgoth', 'Kaggle Winner', 'SPaRCNet']
    expert_color = 'grey'

    # Get model predictions
    y_pred_prob_M = result_df[[f'M_class_{i}_prob' for i in range(n_classes)]].values
    y_pred_prob_B = result_df[[f'K_class_{i}_prob' for i in range(n_classes)]].values
    y_pred_prob_S = result_df[[f'S_class_{i}_prob' for i in range(n_classes)]].values

    # Use majority as ground truth
    y_true = result_df['majority'].values
    y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])

    # Initialize results dictionary for each class
    for class_id in label_map.keys():
        class_name = label_map[class_id]
        results[class_name] = {}
        for model_name in model_names:
            results[class_name][model_name] = {}

    for index, class_id in enumerate(label_map.keys()):
        # Calculate original model curves (for reference)
        original_fpr_models = []
        original_tpr_models = []
        original_precision_models = []
        original_recall_models = []
        original_auc_roc_models = []
        original_auc_pr_models = []

        for model_idx, y_pred_prob in enumerate([y_pred_prob_M, y_pred_prob_B, y_pred_prob_S]):
            fpr_model, tpr_model, _ = roc_curve(y_true_bin[:, class_id], y_pred_prob[:, class_id])
            precision_model, recall_model, _ = precision_recall_curve(y_true_bin[:, class_id], y_pred_prob[:, class_id])

            roc_auc = auc(fpr_model, tpr_model)
            pr_auc = average_precision_score(y_true_bin[:, class_id], y_pred_prob[:, class_id])

            original_fpr_models.append(fpr_model)
            original_tpr_models.append(tpr_model)
            original_precision_models.append(precision_model)
            original_recall_models.append(recall_model)
            original_auc_roc_models.append(roc_auc)
            original_auc_pr_models.append(pr_auc)

        # Prepare structures for bootstrap
        bootstrap_fpr_models = [[] for _ in range(3)]
        bootstrap_tpr_models = [[] for _ in range(3)]
        bootstrap_precision_models = [[] for _ in range(3)]
        bootstrap_recall_models = [[] for _ in range(3)]
        bootstrap_auc_roc_models = [[] for _ in range(3)]
        bootstrap_auc_pr_models = [[] for _ in range(3)]

        # Store bootstrap interpolated curves for percentile calculation
        bootstrap_interp_tprs = [[] for _ in range(3)]
        bootstrap_interp_precisions = [[] for _ in range(3)]
        mean_fpr = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)

        # Store expert bootstrap results
        bootstrap_expert_fpr = []
        bootstrap_expert_tpr = []
        bootstrap_expert_recall = []
        bootstrap_expert_precision = []

        # For calculated mean expert points
        expert_mean_points_roc = {expert: {'fpr': [], 'tpr': []} for expert in expert_columns}
        expert_mean_points_pr = {expert: {'recall': [], 'precision': []} for expert in expert_columns}

        # For EUC calculation
        bootstrap_euc_roc = [[] for _ in range(3)]
        bootstrap_euc_pr = [[] for _ in range(3)]

        # Run bootstrap
        n_samples = len(y_true)

        for bootstrap_idx in range(n_bootstrap):
            # Generate bootstrap sample with replacement
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)

            # Get bootstrap samples for models
            y_true_bootstrap = y_true[bootstrap_indices]
            y_true_bin_bootstrap = label_binarize(y_true_bootstrap, classes=[i for i in range(n_classes)])
            y_pred_prob_M_bootstrap = y_pred_prob_M[bootstrap_indices]
            y_pred_prob_B_bootstrap = y_pred_prob_B[bootstrap_indices]
            y_pred_prob_S_bootstrap = y_pred_prob_S[bootstrap_indices]

            # Bootstrap for models
            for model_idx, y_pred_prob in enumerate(
                    [y_pred_prob_M_bootstrap, y_pred_prob_B_bootstrap, y_pred_prob_S_bootstrap]):
                try:
                    # Calculate ROC curve
                    fpr_model, tpr_model, _ = roc_curve(y_true_bin_bootstrap[:, class_id], y_pred_prob[:, class_id])
                    precision_model, recall_model, _ = precision_recall_curve(y_true_bin_bootstrap[:, class_id],
                                                                              y_pred_prob[:, class_id])

                    roc_auc = auc(fpr_model, tpr_model)
                    pr_auc = average_precision_score(y_true_bin_bootstrap[:, class_id], y_pred_prob[:, class_id])

                    bootstrap_fpr_models[model_idx].append(fpr_model)
                    bootstrap_tpr_models[model_idx].append(tpr_model)
                    bootstrap_precision_models[model_idx].append(precision_model)
                    bootstrap_recall_models[model_idx].append(recall_model)
                    bootstrap_auc_roc_models[model_idx].append(roc_auc)
                    bootstrap_auc_pr_models[model_idx].append(pr_auc)

                    # Interpolate curves for this bootstrap sample
                    if len(fpr_model) > 1:
                        # Find unique FPR values and corresponding TPR values for ROC
                        unique_indices = np.unique(fpr_model, return_index=True)[1]
                        unique_fpr = fpr_model[np.sort(unique_indices)]
                        unique_tpr = tpr_model[np.sort(unique_indices)]

                        if len(unique_fpr) > 1:
                            interp_tpr = np.interp(mean_fpr, unique_fpr, unique_tpr)
                            interp_tpr[0] = 0.0  # Ensure starting at (0,0)
                            bootstrap_interp_tprs[model_idx].append(interp_tpr)

                    if len(recall_model) > 1:
                        # Reverse arrays for PR curve
                        recall_rev = recall_model[::-1]
                        precision_rev = precision_model[::-1]

                        # Find unique recall values and corresponding precision values
                        unique_indices = np.unique(recall_rev, return_index=True)[1]
                        unique_recall = recall_rev[np.sort(unique_indices)]
                        unique_precision = precision_rev[np.sort(unique_indices)]

                        if len(unique_recall) > 1:
                            interp_precision = np.interp(mean_recall, unique_recall, unique_precision)
                            bootstrap_interp_precisions[model_idx].append(interp_precision)

                except Exception as e:
                    print(f"Error in bootstrap iteration {bootstrap_idx} for model {model_idx}: {e}")

            # Bootstrap for experts - resample rows for each expert separately
            iter_expert_fpr = []
            iter_expert_tpr = []
            iter_expert_recall = []
            iter_expert_precision = []

            for expert_idx, expert in enumerate(expert_columns):
                try:
                    # Create bootstrap sample for this expert
                    valid_rows = result_df[[expert, f'exclude_{expert}']].dropna()
                    valid_rows = valid_rows[valid_rows[f'exclude_{expert}'] != -1]

                    # Bootstrap rows for this expert
                    if len(valid_rows) > 0:
                        bootstrap_row_indices = np.random.choice(len(valid_rows), len(valid_rows), replace=True)
                        bootstrap_rows = valid_rows.iloc[bootstrap_row_indices]

                        tp = ((bootstrap_rows[expert] == class_id) & (
                                bootstrap_rows[f'exclude_{expert}'] == class_id)).sum()
                        fp = ((bootstrap_rows[expert] == class_id) & (
                                bootstrap_rows[f'exclude_{expert}'] != class_id)).sum()
                        fn = ((bootstrap_rows[expert] != class_id) & (
                                bootstrap_rows[f'exclude_{expert}'] == class_id)).sum()
                        tn = ((bootstrap_rows[expert] != class_id) & (
                                bootstrap_rows[f'exclude_{expert}'] != class_id)).sum()

                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tpr

                        iter_expert_fpr.append(fpr)
                        iter_expert_tpr.append(tpr)
                        iter_expert_recall.append(recall)
                        iter_expert_precision.append(precision)

                        # Store individual expert results
                        expert_mean_points_roc[expert]['fpr'].append(fpr)
                        expert_mean_points_roc[expert]['tpr'].append(tpr)
                        expert_mean_points_pr[expert]['recall'].append(recall)
                        expert_mean_points_pr[expert]['precision'].append(precision)
                    else:
                        # Add placeholder values
                        iter_expert_fpr.append(np.nan)
                        iter_expert_tpr.append(np.nan)
                        iter_expert_recall.append(np.nan)
                        iter_expert_precision.append(np.nan)
                except Exception as e:
                    print(f"Error in expert bootstrap calculation for {expert}: {e}")
                    # Add placeholder values to maintain alignment with expert columns
                    iter_expert_fpr.append(np.nan)
                    iter_expert_tpr.append(np.nan)
                    iter_expert_recall.append(np.nan)
                    iter_expert_precision.append(np.nan)

            # Store expert statistics for this iteration
            if len(iter_expert_fpr) > 0:
                bootstrap_expert_fpr.append(iter_expert_fpr)
                bootstrap_expert_tpr.append(iter_expert_tpr)
                bootstrap_expert_recall.append(iter_expert_recall)
                bootstrap_expert_precision.append(iter_expert_precision)

            # Calculate EUC for this bootstrap iteration
            # EUC: Experts Under Curve - percentage of experts that fall under (or on) the ROC/PR curve

            # For each model in this bootstrap iteration
            for model_idx in range(3):
                experts_above_curve_roc = 0
                experts_above_curve_pr = 0
                valid_experts_count = 0

                # For each expert in this bootstrap iteration
                for expert_idx, expert in enumerate(expert_columns):
                    if expert_idx < len(iter_expert_fpr) and not np.isnan(iter_expert_fpr[expert_idx]) and not np.isnan(
                            iter_expert_tpr[expert_idx]):
                        valid_experts_count += 1

                        # Get expert point
                        expert_fpr = iter_expert_fpr[expert_idx]
                        expert_tpr = iter_expert_tpr[expert_idx]

                        # Get corresponding model curve
                        if model_idx < len(bootstrap_fpr_models) and bootstrap_idx < len(
                                bootstrap_fpr_models[model_idx]):
                            model_fpr = bootstrap_fpr_models[model_idx][bootstrap_idx]
                            model_tpr = bootstrap_tpr_models[model_idx][bootstrap_idx]

                            # Interpolate expected TPR at expert's FPR
                            if len(model_fpr) > 1:
                                try:
                                    expected_tpr = np.interp(expert_fpr, model_fpr, model_tpr)

                                    # Check if expert is above curve
                                    if expert_tpr > expected_tpr:
                                        experts_above_curve_roc += 1
                                except Exception as e:
                                    print(f"Error in ROC EUC interpolation: {e}")

                    # Same for PR curve
                    if expert_idx < len(iter_expert_recall) and not np.isnan(
                            iter_expert_recall[expert_idx]) and not np.isnan(iter_expert_precision[expert_idx]):
                        # Get expert point
                        expert_recall = iter_expert_recall[expert_idx]
                        expert_precision = iter_expert_precision[expert_idx]

                        # Get corresponding model curve
                        if model_idx < len(bootstrap_recall_models) and bootstrap_idx < len(
                                bootstrap_recall_models[model_idx]):
                            model_recall = bootstrap_recall_models[model_idx][bootstrap_idx]
                            model_precision = bootstrap_precision_models[model_idx][bootstrap_idx]

                            # Interpolate expected precision at expert's recall
                            if len(model_recall) > 1:
                                try:
                                    # Reverse arrays for PR curve interpolation
                                    recall_rev = model_recall[::-1]
                                    precision_rev = model_precision[::-1]

                                    expected_precision = np.interp(expert_recall, recall_rev, precision_rev)

                                    # Check if expert is above curve
                                    if expert_precision > expected_precision:
                                        experts_above_curve_pr += 1
                                except Exception as e:
                                    print(f"Error in PR EUC interpolation: {e}")

                # Calculate EUC percentages for this bootstrap iteration
                if valid_experts_count > 0:
                    euc_roc = (valid_experts_count - experts_above_curve_roc) / valid_experts_count * 100
                    euc_pr = (valid_experts_count - experts_above_curve_pr) / valid_experts_count * 100

                    bootstrap_euc_roc[model_idx].append(euc_roc)
                    bootstrap_euc_pr[model_idx].append(euc_pr)

        # Calculate mean expert points from bootstrap
        expert_mean_roc = {}
        expert_mean_pr = {}

        for expert in expert_columns:
            fpr_values = expert_mean_points_roc[expert]['fpr']
            tpr_values = expert_mean_points_roc[expert]['tpr']
            recall_values = expert_mean_points_pr[expert]['recall']
            precision_values = expert_mean_points_pr[expert]['precision']

            if len(fpr_values) > 0 and len(tpr_values) > 0:
                mean_fpr_expert = np.mean(fpr_values)
                mean_tpr_expert = np.mean(tpr_values)
                expert_mean_roc[expert] = (mean_fpr_expert, mean_tpr_expert)
            else:
                expert_mean_roc[expert] = (0, 0)

            if len(recall_values) > 0 and len(precision_values) > 0:
                mean_recall_expert = np.mean(recall_values)
                mean_precision_expert = np.mean(precision_values)
                expert_mean_pr[expert] = (mean_recall_expert, mean_precision_expert)
            else:
                expert_mean_pr[expert] = (0, 0)

        # Calculate mean EUC values and confidence intervals from bootstrap
        mean_euc_roc = [np.mean(bootstrap_euc_roc[i]) if len(bootstrap_euc_roc[i]) > 0 else 0 for i in range(3)]
        ci_lower_euc_roc = [np.percentile(bootstrap_euc_roc[i], 2.5) if len(bootstrap_euc_roc[i]) > 0 else 0 for i in
                            range(3)]
        ci_upper_euc_roc = [np.percentile(bootstrap_euc_roc[i], 97.5) if len(bootstrap_euc_roc[i]) > 0 else 0 for i in
                            range(3)]

        mean_euc_pr = [np.mean(bootstrap_euc_pr[i]) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in range(3)]
        ci_lower_euc_pr = [np.percentile(bootstrap_euc_pr[i], 2.5) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in
                           range(3)]
        ci_upper_euc_pr = [np.percentile(bootstrap_euc_pr[i], 97.5) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in
                           range(3)]

        # Draw ROC curve
        ax_roc = plt.subplot(2, n_classes, index + 1)

        # Calculate and draw expert confidence intervals as crosses
        if len(bootstrap_expert_fpr) > 0 and len(bootstrap_expert_tpr) > 0:
            # Reorganize bootstrap data by expert
            expert_bootstrap_data_roc = {}

            # For each bootstrap iteration
            for iter_idx in range(len(bootstrap_expert_fpr)):
                # For each expert in this iteration
                for expert_idx, expert_name in enumerate(expert_columns):
                    if expert_idx >= len(bootstrap_expert_fpr[iter_idx]):
                        continue

                    if expert_name not in expert_bootstrap_data_roc:
                        expert_bootstrap_data_roc[expert_name] = {"fpr": [], "tpr": []}

                    if not np.isnan(bootstrap_expert_fpr[iter_idx][expert_idx]) and not np.isnan(
                            bootstrap_expert_tpr[iter_idx][expert_idx]):
                        expert_bootstrap_data_roc[expert_name]["fpr"].append(bootstrap_expert_fpr[iter_idx][expert_idx])
                        expert_bootstrap_data_roc[expert_name]["tpr"].append(bootstrap_expert_tpr[iter_idx][expert_idx])

            # Calculate confidence intervals and draw crosses for each expert
            for expert_name, data in expert_bootstrap_data_roc.items():
                if len(data["fpr"]) > 5 and len(data["tpr"]) > 5:  # Ensure enough bootstrap samples
                    # Calculate the median point
                    median_fpr = np.median(data["fpr"])
                    median_tpr = np.median(data["tpr"])

                    # Calculate the range for fpr and tpr
                    fpr_min = np.percentile(data["fpr"], 2.5)
                    fpr_max = np.percentile(data["fpr"], 97.5)
                    tpr_min = np.percentile(data["tpr"], 2.5)
                    tpr_max = np.percentile(data["tpr"], 97.5)

                    # Draw horizontal line (TPR range)
                    plt.plot([median_fpr, median_fpr], [tpr_min, tpr_max],
                             color=expert_color, alpha=0.5, linewidth=1)

                    # Draw vertical line (FPR range)
                    plt.plot([fpr_min, fpr_max], [median_tpr, median_tpr],
                             color=expert_color, alpha=0.5, linewidth=1)

        # Store legend handles and labels
        legend_handles = []
        legend_labels = []

        # Draw model curves with bootstrap confidence intervals
        for model_idx in range(3):
            # Calculate mean and confidence intervals for AUC
            if len(bootstrap_auc_roc_models[model_idx]) > 0:
                mean_auc_roc = np.mean(bootstrap_auc_roc_models[model_idx])
                ci_lower_auc_roc = np.percentile(bootstrap_auc_roc_models[model_idx], 2.5)
                ci_upper_auc_roc = np.percentile(bootstrap_auc_roc_models[model_idx], 97.5)
            else:
                mean_auc_roc = original_auc_roc_models[model_idx]
                ci_lower_auc_roc = mean_auc_roc
                ci_upper_auc_roc = mean_auc_roc

            # Store results for this model
            results[label_map[class_id]][model_names[model_idx]]['roc_auc'] = {
                'mean': mean_auc_roc,
                'min': ci_lower_auc_roc,
                'max': ci_upper_auc_roc
            }
            results[label_map[class_id]][model_names[model_idx]]['euc_roc'] = {
                'mean': mean_euc_roc[model_idx],
                'min': ci_lower_euc_roc[model_idx],
                'max': ci_upper_euc_roc[model_idx]
            }

            # Add to formatted output
            formatted_output += f"\n{label_map[class_id]} - {model_names[model_idx]}:\n"
            formatted_output += f"ROC AUC: {mean_auc_roc:.3f}({ci_lower_auc_roc:.3f}, {ci_upper_auc_roc:.3f})\n"
            formatted_output += f"ROC EUC: {mean_euc_roc[model_idx]:.1f}%({ci_lower_euc_roc[model_idx]:.1f}%, {ci_upper_euc_roc[model_idx]:.1f}%)\n"

            # Calculate true 95% confidence bands and mean curve from bootstrap samples
            if len(bootstrap_interp_tprs[model_idx]) > 0:
                bootstrap_interp_tprs_array = np.array(bootstrap_interp_tprs[model_idx])
                # Calculate mean curve
                mean_tpr = np.mean(bootstrap_interp_tprs_array, axis=0)
                # Calculate true 95% confidence bounds directly from percentiles
                tpr_lower = np.percentile(bootstrap_interp_tprs_array, 2.5, axis=0)
                tpr_upper = np.percentile(bootstrap_interp_tprs_array, 97.5, axis=0)

                # Draw mean bootstrap curve (instead of original)
                line, = plt.plot(mean_fpr, mean_tpr, color=model_colors[model_idx], lw=2)
            else:
                # Use original curve if no valid bootstrap samples
                mean_tpr = np.interp(mean_fpr, original_fpr_models[model_idx], original_tpr_models[model_idx])
                tpr_upper = mean_tpr
                tpr_lower = mean_tpr

                # Draw original curve as fallback
                line, = plt.plot(original_fpr_models[model_idx], original_tpr_models[model_idx],
                                 color=model_colors[model_idx], lw=2)

            # Create simplified label for legend with EUC confidence intervals
            legend_handles.append(line)
            legend_labels.append(
                f"  {model_names[model_idx]}\nAUC={mean_auc_roc:.3f}\nEUC={mean_euc_roc[model_idx]:.1f}%")

            # Draw confidence interval
            plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=model_colors[model_idx], alpha=0.3)

        # Draw mean expert points from bootstrap (instead of original)
        for expert in expert_columns:
            mean_fpr_expert, mean_tpr_expert = expert_mean_roc[expert]
            plt.scatter(mean_fpr_expert, mean_tpr_expert, marker='o', color=expert_color, alpha=0.6, s=20)

        # Hide axis labels
        plt.xlabel('')
        plt.ylabel('')

        # Only the first column shows y-axis ticks
        if index != 0:
            ax_roc.set_yticklabels([])

        # This is the first row, hide all x-axis ticks
        ax_roc.set_xticklabels([])

        plt.legend(legend_handles, legend_labels, loc='lower right', handlelength=0, handletextpad=0,
                   fontsize=15.5, frameon=False, title=f'{label_map[class_id]}', title_fontsize=18)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # Draw PR curve
        ax_pr = plt.subplot(2, n_classes, n_classes + index + 1)

        # Calculate and draw expert confidence intervals as crosses for PR curve
        if len(bootstrap_expert_recall) > 0 and len(bootstrap_expert_precision) > 0:
            # Reorganize bootstrap data by expert
            expert_bootstrap_data_pr = {}

            # For each bootstrap iteration
            for iter_idx in range(len(bootstrap_expert_recall)):
                # For each expert in this iteration
                for expert_idx, expert_name in enumerate(expert_columns):
                    if expert_idx >= len(bootstrap_expert_recall[iter_idx]):
                        continue

                    if expert_name not in expert_bootstrap_data_pr:
                        expert_bootstrap_data_pr[expert_name] = {"recall": [], "precision": []}

                    if not np.isnan(bootstrap_expert_recall[iter_idx][expert_idx]) and not np.isnan(
                            bootstrap_expert_precision[iter_idx][expert_idx]):
                        expert_bootstrap_data_pr[expert_name]["recall"].append(
                            bootstrap_expert_recall[iter_idx][expert_idx])
                        expert_bootstrap_data_pr[expert_name]["precision"].append(
                            bootstrap_expert_precision[iter_idx][expert_idx])

            # Calculate confidence intervals and draw crosses for each expert
            for expert_name, data in expert_bootstrap_data_pr.items():
                if len(data["recall"]) > 5 and len(data["precision"]) > 5:  # Ensure enough bootstrap samples
                    # Calculate the median point
                    median_recall = np.median(data["recall"])
                    median_precision = np.median(data["precision"])

                    # Calculate the range for recall and precision
                    recall_min = np.percentile(data["recall"], 2.5)
                    recall_max = np.percentile(data["recall"], 97.5)
                    precision_min = np.percentile(data["precision"], 2.5)
                    precision_max = np.percentile(data["precision"], 97.5)

                    # Draw horizontal line (precision range)
                    plt.plot([median_recall, median_recall], [precision_min, precision_max],
                             color=expert_color, alpha=0.5, linewidth=1)

                    # Draw vertical line (recall range)
                    plt.plot([recall_min, recall_max], [median_precision, median_precision],
                             color=expert_color, alpha=0.5, linewidth=1)

        # Reset legend handles and labels for PR curve
        legend_handles = []
        legend_labels = []

        # Draw model curves with bootstrap confidence intervals for PR
        for model_idx in range(3):
            # Calculate mean and confidence intervals for PR AUC
            if len(bootstrap_auc_pr_models[model_idx]) > 0:
                mean_auc_pr = np.mean(bootstrap_auc_pr_models[model_idx])
                ci_lower_auc_pr = np.percentile(bootstrap_auc_pr_models[model_idx], 2.5)
                ci_upper_auc_pr = np.percentile(bootstrap_auc_pr_models[model_idx], 97.5)
            else:
                mean_auc_pr = original_auc_pr_models[model_idx]
                ci_lower_auc_pr = mean_auc_pr
                ci_upper_auc_pr = mean_auc_pr

            # Store PR results for this model
            results[label_map[class_id]][model_names[model_idx]]['pr_auc'] = {
                'mean': mean_auc_pr,
                'min': ci_lower_auc_pr,
                'max': ci_upper_auc_pr
            }
            results[label_map[class_id]][model_names[model_idx]]['euc_pr'] = {
                'mean': mean_euc_pr[model_idx],
                'min': ci_lower_euc_pr[model_idx],
                'max': ci_upper_euc_pr[model_idx]
            }

            # Add to formatted output
            formatted_output += f"PR EUC: {mean_euc_pr[model_idx]:.1f}%({ci_lower_euc_pr[model_idx]:.1f}%, {ci_upper_euc_pr[model_idx]:.1f}%)\n"

            # Calculate true 95% confidence bands and mean curve from bootstrap samples
            if len(bootstrap_interp_precisions[model_idx]) > 0:
                bootstrap_interp_precisions_array = np.array(bootstrap_interp_precisions[model_idx])
                # Calculate mean curve
                mean_precision = np.mean(bootstrap_interp_precisions_array, axis=0)
                # Calculate true 95% confidence bounds directly from percentiles
                precision_lower = np.percentile(bootstrap_interp_precisions_array, 2.5, axis=0)
                precision_upper = np.percentile(bootstrap_interp_precisions_array, 97.5, axis=0)

                # Draw mean bootstrap curve (instead of original)
                line, = plt.plot(mean_recall, mean_precision, color=model_colors[model_idx], lw=2)
            else:
                # Use original curve if no valid bootstrap samples
                mean_precision = np.interp(mean_recall, np.flip(original_recall_models[model_idx]),
                                           np.flip(original_precision_models[model_idx]))
                precision_upper = mean_precision
                precision_lower = mean_precision

                # Draw original curve as fallback
                line, = plt.plot(original_recall_models[model_idx], original_precision_models[model_idx],
                                 color=model_colors[model_idx], lw=2)

            # Create simplified label for legend with EUC confidence intervals
            legend_handles.append(line)
            legend_labels.append(
                f"  {model_names[model_idx]}\nAUC={mean_auc_pr:.3f}\nEUC={mean_euc_pr[model_idx]:.1f}%")

            # Draw confidence interval
            plt.fill_between(mean_recall, precision_lower, precision_upper, color=model_colors[model_idx], alpha=0.2)

        # Draw mean expert points from bootstrap (instead of original)
        for expert in expert_columns:
            mean_recall_expert, mean_precision_expert = expert_mean_pr[expert]
            plt.scatter(mean_recall_expert, mean_precision_expert, marker='o', color=expert_color, alpha=0.6, s=20)

        # Hide axis labels
        plt.xlabel('')
        plt.ylabel('')

        # Only the first column shows y-axis ticks
        if index != 0:
            ax_pr.set_yticklabels([])

        plt.legend(legend_handles, legend_labels, loc='lower left', handlelength=0, handletextpad=0,
                   fontsize=15.5, frameon=False, title=f'{label_map[class_id]}', title_fontsize=18)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(out_figure_path, dpi=300)
    plt.show()

    # Print the formatted output
    print(formatted_output)

    print("\n\n===== Complete performance metrics for all models =====\n")
    print(f"{'Class':<10} {'Model':<15} {'AUC-ROC':<25} {'AUC-PR':<25} {'EUC-ROC':<25} {'EUC-PR':<25}")
    print("=" * 100)

    # Iterate over each class
    for class_name, class_results in results.items():
        # Iterate over each model
        for model_name, model_metrics in class_results.items():
            # Get and format AUC-ROC
            if 'roc_auc' in model_metrics:
                auc_roc = model_metrics['roc_auc']
                auc_roc_str = f"{auc_roc['mean']:.3f} ({auc_roc['min']:.3f}, {auc_roc['max']:.3f})"
            else:
                auc_roc_str = "N/A"

            # Get and format AUC-PR
            if 'pr_auc' in model_metrics:
                auc_pr = model_metrics['pr_auc']
                auc_pr_str = f"{auc_pr['mean']:.3f} ({auc_pr['min']:.3f}, {auc_pr['max']:.3f})"
            else:
                auc_pr_str = "N/A"

            # Get and format EUC-ROC
            if 'euc_roc' in model_metrics:
                euc_roc = model_metrics['euc_roc']
                euc_roc_str = f"{euc_roc['mean']:.1f}% ({euc_roc['min']:.1f}%, {euc_roc['max']:.1f}%)"
            else:
                euc_roc_str = "N/A"

            # Get and format EUC-PR
            if 'euc_pr' in model_metrics:
                euc_pr = model_metrics['euc_pr']
                euc_pr_str = f"{euc_pr['mean']:.1f}% ({euc_pr['min']:.1f}%, {euc_pr['max']:.1f}%)"
            else:
                euc_pr_str = "N/A"

            # Print all metrics for the current model
            print(
                f"{class_name:<10} {model_name:<15} {auc_roc_str:<25} {auc_pr_str:<25} {euc_roc_str:<25} {euc_pr_str:<25}")

        # Add separator between classes
        print("-" * 100)

    return results, formatted_output

def plot_auc_euc_with_bootstrap_SAI(fig_path, label, results_file, n_bootstrap=1000):


    # String to collect formatted output
    formatted_output = ""

    # Dictionary to store results
    results = {}

    labels = [
        label,
    ]
    binary_label = [
        label
    ]
    label_maps = [
        {0: 'Other', 1: label},
    ]

    # Set colors for the two models
    model_colors = ['steelblue', 'orange']
    model_names = ['Morgoth', 'SCORE-AI']

    # Expert color
    expert_color = 'grey'

    # Initialize results dictionary
    results[label] = {}
    for model_name in model_names:
        results[label][model_name] = {}

    for label_name, label_map in zip(labels, label_maps):
        class_names = list(label_map.values())
        n_classes = len(class_names)

        # Read the results file
        result_df = pd.read_excel(results_file)

        # Get all expert columns
        expert_columns = [c for c in result_df.columns if c.startswith('expert')]

        # Generate exclude columns for each expert
        for expert in expert_columns:
            # Get all other expert columns
            other_experts = [col for col in expert_columns if col != expert]

            if not other_experts:
                continue  # Skip if there are no other experts

            # Create a function to find the mode, handling NaN values
            def get_mode_excluding_nan(row):
                # Get values from other experts, exclude NaN
                values = [row[exp] for exp in other_experts if not pd.isna(row[exp])]
                # If no valid values, return -1
                if not values:
                    return -1
                # Find the most common value
                from collections import Counter
                counter = Counter(values)
                if not counter:
                    return -1
                # Get the most common value (mode)
                return counter.most_common(1)[0][0]

            # Create exclude column by finding the mode of other experts
            result_df[f'exclude_{expert}'] = result_df.apply(get_mode_excluding_nan, axis=1)

        plt.figure(figsize=(4 * (n_classes - 1), 8) if label_name in binary_label else (4 * n_classes, 8))

        # Get model predictions
        if label_name in binary_label:
            y_pred_prob_M = result_df[['M_pred']].values
            y_pred_prob_B = result_df[['S_pred']].values
        else:
            y_pred_prob_M = result_df[[f'class_{i}_prob' for i in range(n_classes)]].values
            y_pred_prob_B = result_df[[f'B_class_{i}_prob' for i in range(n_classes)]].values

        for index, class_id in enumerate(label_map.keys()):
            if class_id == 0 and label_name in binary_label:
                continue

            class_name = label_map[class_id]

            # For storing bootstrapped expert points
            expert_bootstrap_roc = {expert: {'fpr': [], 'tpr': []} for expert in expert_columns}
            expert_bootstrap_pr = {expert: {'recall': [], 'precision': []} for expert in expert_columns}

            # For storing bootstrapped model curves
            model_bootstrap_roc = [{'fpr': [], 'tpr': [], 'auc': []} for _ in range(2)]
            model_bootstrap_pr = [{'recall': [], 'precision': [], 'auc': []} for _ in range(2)]

            # For bootstrapped EUC calculation
            bootstrap_euc_roc = [[] for _ in range(2)]
            bootstrap_euc_pr = [[] for _ in range(2)]

            # Prepare interpolation grid
            mean_fpr = np.linspace(0, 1, 100)
            mean_recall = np.linspace(0, 1, 100)

            # Store interpolated curves for calculating confidence intervals
            bootstrap_interp_tprs = [[] for _ in range(2)]
            bootstrap_interp_precisions = [[] for _ in range(2)]

            # Run bootstrap iterations
            for bootstrap_iter in range(n_bootstrap):
                # For each bootstrap iteration, sample rows with replacement
                n_rows = len(result_df)
                bootstrap_indices = np.random.choice(n_rows, n_rows, replace=True)
                bootstrap_df = result_df.iloc[bootstrap_indices].copy()

                # Expert statistics for this bootstrap iteration
                iter_expert_roc = {expert: [] for expert in expert_columns}
                iter_expert_pr = {expert: [] for expert in expert_columns}

                # Model curves for this bootstrap iteration
                iter_model_roc = [[], []]
                iter_model_pr = [[], []]

                # Calculate bootstrapped expert points
                for expert in expert_columns:
                    # Skip rows where exclude_{expert} is -1
                    valid_rows = bootstrap_df[(bootstrap_df[f'exclude_{expert}'] != -1) &
                                              (~bootstrap_df[f'exclude_{expert}'].isna()) &
                                              (~bootstrap_df[expert].isna())]

                    # If no valid rows, skip this expert
                    if len(valid_rows) == 0:
                        continue

                    tp = ((valid_rows[expert] == class_id) & (valid_rows[f'exclude_{expert}'] == class_id)).sum()
                    fp = ((valid_rows[expert] == class_id) & (valid_rows[f'exclude_{expert}'] != class_id)).sum()
                    fn = ((valid_rows[expert] != class_id) & (valid_rows[f'exclude_{expert}'] == class_id)).sum()
                    tn = ((valid_rows[expert] != class_id) & (valid_rows[f'exclude_{expert}'] != class_id)).sum()

                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tpr

                    # Store expert point for this bootstrap iteration
                    expert_bootstrap_roc[expert]['fpr'].append(fpr)
                    expert_bootstrap_roc[expert]['tpr'].append(tpr)
                    expert_bootstrap_pr[expert]['recall'].append(recall)
                    expert_bootstrap_pr[expert]['precision'].append(precision)

                    # Also store for current iteration's EUC calculation
                    iter_expert_roc[expert] = (fpr, tpr)
                    iter_expert_pr[expert] = (recall, precision)

                    # Calculate bootstrapped model curves against this expert's reference
                    y_true_expert = valid_rows[f'exclude_{expert}'].values

                    # Convert to integers before using label_binarize
                    y_true_expert = y_true_expert.astype(int)
                    y_true_bin_expert = label_binarize(y_true_expert, classes=[i for i in range(n_classes)])

                    # Get corresponding model predictions for valid rows
                    valid_indices = valid_rows.index
                    y_pred_prob_M_valid = y_pred_prob_M[valid_indices]
                    y_pred_prob_B_valid = y_pred_prob_B[valid_indices]

                    # Calculate curves for both models
                    for model_idx, y_pred_prob in enumerate([y_pred_prob_M_valid, y_pred_prob_B_valid]):
                        try:
                            if label_name in binary_label:
                                fpr_model, tpr_model, _ = roc_curve(y_true_bin_expert[:, class_id - 1],
                                                                    y_pred_prob[:, class_id - 1])
                                precision_model, recall_model, _ = precision_recall_curve(
                                    y_true_bin_expert[:, class_id - 1],
                                    y_pred_prob[:, class_id - 1])
                                pr_auc = average_precision_score(y_true_bin_expert[:, class_id - 1],
                                                                 y_pred_prob[:, class_id - 1])
                            else:
                                fpr_model, tpr_model, _ = roc_curve(y_true_bin_expert[:, class_id],
                                                                    y_pred_prob[:, class_id])
                                precision_model, recall_model, _ = precision_recall_curve(
                                    y_true_bin_expert[:, class_id],
                                    y_pred_prob[:, class_id])
                                pr_auc = average_precision_score(y_true_bin_expert[:, class_id],
                                                                 y_pred_prob[:, class_id])

                            # Calculate ROC AUC
                            roc_auc = auc(fpr_model, tpr_model)

                            # Store model curve for this bootstrap iteration and expert
                            iter_model_roc[model_idx].append((fpr_model, tpr_model, roc_auc))
                            iter_model_pr[model_idx].append((recall_model, precision_model, pr_auc))

                            # Store for overall bootstrap statistics
                            model_bootstrap_roc[model_idx]['auc'].append(roc_auc)
                            model_bootstrap_pr[model_idx]['auc'].append(pr_auc)

                            # Compute interpolated curves for confidence intervals
                            if len(fpr_model) > 1:
                                # Find unique FPR values and corresponding TPR values for ROC
                                unique_indices = np.unique(fpr_model, return_index=True)[1]
                                unique_fpr = fpr_model[np.sort(unique_indices)]
                                unique_tpr = tpr_model[np.sort(unique_indices)]

                                if len(unique_fpr) > 1:
                                    interp_tpr = np.interp(mean_fpr, unique_fpr, unique_tpr)
                                    interp_tpr[0] = 0.0  # Ensure starting at (0,0)
                                    bootstrap_interp_tprs[model_idx].append(interp_tpr)

                            if len(recall_model) > 1:
                                # Reverse arrays for PR curve interpolation
                                recall_rev = recall_model[::-1]
                                precision_rev = precision_model[::-1]

                                # Find unique recall values and corresponding precision values
                                unique_indices = np.unique(recall_rev, return_index=True)[1]
                                unique_recall = recall_rev[np.sort(unique_indices)]
                                unique_precision = precision_rev[np.sort(unique_indices)]

                                if len(unique_recall) > 1:
                                    interp_precision = np.interp(mean_recall, unique_recall, unique_precision)
                                    bootstrap_interp_precisions[model_idx].append(interp_precision)

                        except Exception as e:
                            print(
                                f"Error in bootstrap iteration {bootstrap_iter} for expert {expert}, model {model_idx}: {e}")
                            continue

                # Calculate averaged model curves for this bootstrap iteration
                for model_idx in range(2):
                    # Average ROC curves
                    if iter_model_roc[model_idx]:
                        # Interpolate and average all curves for this model in this iteration
                        iter_tprs = []
                        for fpr, tpr, _ in iter_model_roc[model_idx]:
                            if len(fpr) > 1:
                                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                                interp_tpr[0] = 0.0  # Force start at 0,0
                                iter_tprs.append(interp_tpr)

                        if iter_tprs:
                            avg_tpr = np.mean(iter_tprs, axis=0)
                            model_bootstrap_roc[model_idx]['fpr'].append(mean_fpr)
                            model_bootstrap_roc[model_idx]['tpr'].append(avg_tpr)

                    # Average PR curves
                    if iter_model_pr[model_idx]:
                        # Interpolate and average all curves for this model in this iteration
                        iter_precisions = []
                        for recall, precision, _ in iter_model_pr[model_idx]:
                            if len(recall) > 1:
                                # Flip arrays for interpolation
                                recall_rev = recall[::-1]
                                precision_rev = precision[::-1]
                                interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
                                iter_precisions.append(interp_precision)

                        if iter_precisions:
                            avg_precision = np.mean(iter_precisions, axis=0)
                            model_bootstrap_pr[model_idx]['recall'].append(mean_recall)
                            model_bootstrap_pr[model_idx]['precision'].append(avg_precision)

                # Calculate EUC for this bootstrap iteration
                for model_idx in range(2):
                    experts_above_curve_roc = 0
                    experts_above_curve_pr = 0
                    valid_experts_count = 0

                    # Skip if no model curves for this iteration
                    if not model_bootstrap_roc[model_idx]['fpr'] or not model_bootstrap_pr[model_idx]['recall']:
                        continue

                    # Get the model curves for this iteration
                    model_fpr = model_bootstrap_roc[model_idx]['fpr'][-1]  # Last added is current iteration
                    model_tpr = model_bootstrap_roc[model_idx]['tpr'][-1]
                    model_recall = model_bootstrap_pr[model_idx]['recall'][-1]
                    model_precision = model_bootstrap_pr[model_idx]['precision'][-1]

                    # Compare each expert point to the model curve
                    for expert in expert_columns:
                        if expert in iter_expert_roc and expert in iter_expert_pr:
                            valid_experts_count += 1

                            # Get expert points
                            expert_fpr, expert_tpr = iter_expert_roc[expert]
                            expert_recall, expert_precision = iter_expert_pr[expert]

                            # Check if expert is above ROC curve
                            expected_tpr = np.interp(expert_fpr, model_fpr, model_tpr)
                            if expert_tpr > expected_tpr:
                                experts_above_curve_roc += 1

                            # Check if expert is above PR curve
                            expected_precision = np.interp(expert_recall, model_recall, model_precision)
                            if expert_precision > expected_precision:
                                experts_above_curve_pr += 1

                    # Calculate EUC percentages
                    if valid_experts_count > 0:
                        euc_roc = (valid_experts_count - experts_above_curve_roc) / valid_experts_count * 100
                        euc_pr = (valid_experts_count - experts_above_curve_pr) / valid_experts_count * 100

                        bootstrap_euc_roc[model_idx].append(euc_roc)
                        bootstrap_euc_pr[model_idx].append(euc_pr)

            # Calculate mean expert points and confidence intervals
            expert_mean_roc = {}
            expert_mean_pr = {}
            expert_ci_roc = {}
            expert_ci_pr = {}

            for expert in expert_columns:
                # Calculate means if data exists
                if expert_bootstrap_roc[expert]['fpr'] and expert_bootstrap_roc[expert]['tpr']:
                    mean_fpr_expert = np.mean(expert_bootstrap_roc[expert]['fpr'])
                    mean_tpr_expert = np.mean(expert_bootstrap_roc[expert]['tpr'])

                    # Calculate 95% confidence intervals
                    ci_fpr_lower = np.percentile(expert_bootstrap_roc[expert]['fpr'], 2.5)
                    ci_fpr_upper = np.percentile(expert_bootstrap_roc[expert]['fpr'], 97.5)
                    ci_tpr_lower = np.percentile(expert_bootstrap_roc[expert]['tpr'], 2.5)
                    ci_tpr_upper = np.percentile(expert_bootstrap_roc[expert]['tpr'], 97.5)

                    expert_mean_roc[expert] = (mean_fpr_expert, mean_tpr_expert)
                    expert_ci_roc[expert] = ((ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper))
                else:
                    expert_mean_roc[expert] = (0, 0)
                    expert_ci_roc[expert] = ((0, 0), (0, 0))

                # Same for PR curve
                if expert_bootstrap_pr[expert]['recall'] and expert_bootstrap_pr[expert]['precision']:
                    mean_recall_expert = np.mean(expert_bootstrap_pr[expert]['recall'])
                    mean_precision_expert = np.mean(expert_bootstrap_pr[expert]['precision'])

                    # Calculate 95% confidence intervals
                    ci_recall_lower = np.percentile(expert_bootstrap_pr[expert]['recall'], 2.5)
                    ci_recall_upper = np.percentile(expert_bootstrap_pr[expert]['recall'], 97.5)
                    ci_precision_lower = np.percentile(expert_bootstrap_pr[expert]['precision'], 2.5)
                    ci_precision_upper = np.percentile(expert_bootstrap_pr[expert]['precision'], 97.5)

                    expert_mean_pr[expert] = (mean_recall_expert, mean_precision_expert)
                    expert_ci_pr[expert] = (
                    (ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper))
                else:
                    expert_mean_pr[expert] = (0, 0)
                    expert_ci_pr[expert] = ((0, 0), (0, 0))

            # Calculate model statistics
            mean_model_roc_auc = [np.mean(model_bootstrap_roc[i]['auc']) if model_bootstrap_roc[i]['auc'] else 0 for i
                                  in range(2)]
            ci_lower_roc_auc = [
                np.percentile(model_bootstrap_roc[i]['auc'], 2.5) if model_bootstrap_roc[i]['auc'] else 0 for i in
                range(2)]
            ci_upper_roc_auc = [
                np.percentile(model_bootstrap_roc[i]['auc'], 97.5) if model_bootstrap_roc[i]['auc'] else 0 for i in
                range(2)]

            mean_model_pr_auc = [np.mean(model_bootstrap_pr[i]['auc']) if model_bootstrap_pr[i]['auc'] else 0 for i in
                                 range(2)]
            ci_lower_pr_auc = [np.percentile(model_bootstrap_pr[i]['auc'], 2.5) if model_bootstrap_pr[i]['auc'] else 0
                               for i in range(2)]
            ci_upper_pr_auc = [np.percentile(model_bootstrap_pr[i]['auc'], 97.5) if model_bootstrap_pr[i]['auc'] else 0
                               for i in range(2)]

            # Calculate EUC statistics
            mean_euc_roc = [np.mean(bootstrap_euc_roc[i]) if bootstrap_euc_roc[i] else 0 for i in range(2)]
            ci_lower_euc_roc = [np.percentile(bootstrap_euc_roc[i], 2.5) if bootstrap_euc_roc[i] else 0 for i in
                                range(2)]
            ci_upper_euc_roc = [np.percentile(bootstrap_euc_roc[i], 97.5) if bootstrap_euc_roc[i] else 0 for i in
                                range(2)]

            mean_euc_pr = [np.mean(bootstrap_euc_pr[i]) if bootstrap_euc_pr[i] else 0 for i in range(2)]
            ci_lower_euc_pr = [np.percentile(bootstrap_euc_pr[i], 2.5) if bootstrap_euc_pr[i] else 0 for i in range(2)]
            ci_upper_euc_pr = [np.percentile(bootstrap_euc_pr[i], 97.5) if bootstrap_euc_pr[i] else 0 for i in range(2)]

            # Update results dictionary
            for model_idx, model_name in enumerate(model_names):
                results[label_name][model_name]['roc_auc'] = {
                    'mean': mean_model_roc_auc[model_idx],
                    'min': ci_lower_roc_auc[model_idx],
                    'max': ci_upper_roc_auc[model_idx]
                }
                results[label_name][model_name]['pr_auc'] = {
                    'mean': mean_model_pr_auc[model_idx],
                    'min': ci_lower_pr_auc[model_idx],
                    'max': ci_upper_pr_auc[model_idx]
                }
                results[label_name][model_name]['euc_roc'] = {
                    'mean': mean_euc_roc[model_idx],
                    'min': ci_lower_euc_roc[model_idx],
                    'max': ci_upper_euc_roc[model_idx]
                }
                results[label_name][model_name]['euc_pr'] = {
                    'mean': mean_euc_pr[model_idx],
                    'min': ci_lower_euc_pr[model_idx],
                    'max': ci_upper_euc_pr[model_idx]
                }

                # Add to formatted output
                formatted_output += f"\n{label_name} - {model_name}:\n"
                formatted_output += f"ROC AUC: {mean_model_roc_auc[model_idx]:.3f}({ci_lower_roc_auc[model_idx]:.3f}, {ci_upper_roc_auc[model_idx]:.3f})\n"
                formatted_output += f"ROC EUC: {mean_euc_roc[model_idx]:.1f}%({ci_lower_euc_roc[model_idx]:.1f}%, {ci_upper_euc_roc[model_idx]:.1f}%)\n"
                formatted_output += f"PR AUC: {mean_model_pr_auc[model_idx]:.3f}({ci_lower_pr_auc[model_idx]:.3f}, {ci_upper_pr_auc[model_idx]:.3f})\n"
                formatted_output += f"PR EUC: {mean_euc_pr[model_idx]:.1f}%({ci_lower_euc_pr[model_idx]:.1f}%, {ci_upper_euc_pr[model_idx]:.1f}%)\n"

            # Draw ROC curve
            ax_roc = plt.subplot(2, 1, 1)

            # Draw expert points with confidence interval crosses
            for expert in expert_columns:
                if expert in expert_mean_roc:
                    # Draw mean expert point
                    mean_fpr_expert, mean_tpr_expert = expert_mean_roc[expert]
                    plt.scatter(mean_fpr_expert, mean_tpr_expert, marker='o', color=expert_color, alpha=0.6, s=20)

                    # Draw confidence interval crosses
                    (ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper) = expert_ci_roc[expert]

                    # Draw horizontal line (TPR range)
                    plt.plot([mean_fpr_expert, mean_fpr_expert], [ci_tpr_lower, ci_tpr_upper],
                             color=expert_color, alpha=0.6, linewidth=1)

                    # Draw vertical line (FPR range)
                    plt.plot([ci_fpr_lower, ci_fpr_upper], [mean_tpr_expert, mean_tpr_expert],
                             color=expert_color, alpha=0.6, linewidth=1)

            # Plot ROC curves with confidence intervals
            legend_handles = []
            legend_labels = []

            for model_idx in range(2):
                # Calculate mean curve and confidence bands from interpolated curves
                if bootstrap_interp_tprs[model_idx]:
                    bootstrap_tprs_array = np.array(bootstrap_interp_tprs[model_idx])
                    mean_tpr = np.mean(bootstrap_tprs_array, axis=0)
                    tpr_lower = np.percentile(bootstrap_tprs_array, 2.5, axis=0)
                    tpr_upper = np.percentile(bootstrap_tprs_array, 97.5, axis=0)

                    # Plot mean ROC curve
                    line, = plt.plot(mean_fpr, mean_tpr, color=model_colors[model_idx], lw=2)

                    # Create legend entry with AUC and EUC confidence intervals
                    legend_handles.append(line)
                    legend_labels.append(
                        f"  {model_names[model_idx]}\nAUC={mean_model_roc_auc[model_idx]:.3f}\nEUC={mean_euc_roc[model_idx]:.1f}%")

                    # Draw confidence bands
                    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=model_colors[model_idx], alpha=0.3)

            # Hide axis labels
            plt.xlabel('')
            plt.ylabel('')

            if label != 'Normal':
                ax_roc.set_yticklabels([])

            # Hide x-axis ticks
            ax_roc.set_xticklabels([])

            # Add legend
            plt.legend(legend_handles, legend_labels, loc='lower right',
                       fontsize=16, handlelength=0, handletextpad=0,
                       frameon=False, title=f'{label_name}', title_fontsize=18)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            # Draw PR curve
            ax_pr = plt.subplot(2, 1, 2)

            # Draw expert points with confidence interval crosses
            for expert in expert_columns:
                if expert in expert_mean_pr:
                    # Draw mean expert point
                    mean_recall_expert, mean_precision_expert = expert_mean_pr[expert]
                    plt.scatter(mean_recall_expert, mean_precision_expert, marker='o', color=expert_color, alpha=0.6,
                                s=20)

                    # Draw confidence interval crosses
                    (ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper) = expert_ci_pr[expert]

                    # Draw horizontal line (precision range)
                    plt.plot([mean_recall_expert, mean_recall_expert], [ci_precision_lower, ci_precision_upper],
                             color=expert_color, alpha=0.6, linewidth=1)

                    # Draw vertical line (recall range)
                    plt.plot([ci_recall_lower, ci_recall_upper], [mean_precision_expert, mean_precision_expert],
                             color=expert_color, alpha=0.6, linewidth=1)

            # Reset legend handles and labels for PR curve
            legend_handles = []
            legend_labels = []

            for model_idx in range(2):
                # Calculate mean curve and confidence bands from interpolated curves
                if bootstrap_interp_precisions[model_idx]:
                    bootstrap_precisions_array = np.array(bootstrap_interp_precisions[model_idx])
                    mean_precision = np.mean(bootstrap_precisions_array, axis=0)
                    precision_lower = np.percentile(bootstrap_precisions_array, 2.5, axis=0)
                    precision_upper = np.percentile(bootstrap_precisions_array, 97.5, axis=0)

                    # Plot mean PR curve
                    line, = plt.plot(mean_recall, mean_precision, color=model_colors[model_idx], lw=2)

                    # Create legend entry with AUC and EUC confidence intervals
                    legend_handles.append(line)
                    legend_labels.append(
                        f"  {model_names[model_idx]}\nAUC={mean_model_pr_auc[model_idx]:.3f}\nEUC={mean_euc_pr[model_idx]:.1f}%")

                    # Draw confidence bands
                    plt.fill_between(mean_recall, precision_lower, precision_upper, color=model_colors[model_idx],
                                     alpha=0.3)

            # Hide axis labels
            plt.xlabel('')
            plt.ylabel('')

            if label != 'Normal':
                ax_pr.set_yticklabels([])

            # Add legend
            plt.legend(legend_handles, legend_labels, loc='lower left',
                       fontsize=16, handlelength=0, handletextpad=0,
                       frameon=False, title=f'{label_name}', title_fontsize=18)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.show()

    # Print the formatted output
    print(formatted_output)

    return results, formatted_output


def plot_auc_euc_with_bootstrap_SAI(output_path, result_file, n_bootstrap=1000):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    from sklearn.preprocessing import label_binarize

    # String to collect formatted output
    formatted_output = ""

    # Dictionary to store results
    results = {}

    label_map = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    # Initialize results dictionary
    for class_name in label_map.values():
        results[class_name] = {}

    # Model configurations
    model_configs = [
        {"name": "M", "color": "steelblue", "display_name": "Morgoth"},
        {"name": "U1", "color": "orange", "display_name": "U-Sleep1"},
        {"name": "U2", "color": "moccasin", "display_name": "U-Sleep2"}
    ]

    class_names = list(label_map.values())
    n_classes = len(class_names)

    result_df = pd.read_csv(result_file)
    result_df.rename(columns={'stage_expert_majority': 'expert_majority'}, inplace=True)

    expert_columns = [c for c in result_df.columns if c.startswith('stage')]

    # Create a big figure for all sleep stages ROC and PR curves
    plt.figure(figsize=(4 * n_classes, 8))

    # Expert color
    expert_color = 'gray'

    # Process each label
    for index, class_id in enumerate(label_map.keys()):
        class_name = label_map[class_id]

        # Initialize ROC and PR subplots
        ax_roc = plt.subplot(2, n_classes, class_id + 1)
        ax_pr = plt.subplot(2, n_classes, n_classes + class_id + 1)

        # Initialize bootstrap storage for expert points
        expert_bootstrap_roc = {expert: {'fpr': [], 'tpr': []} for expert in expert_columns}
        expert_bootstrap_pr = {expert: {'recall': [], 'precision': []} for expert in expert_columns}

        # For each model, initialize bootstrap storage
        model_bootstrap = []
        for _ in model_configs:
            model_bootstrap.append({
                'roc': {'fpr': [], 'tpr': [], 'auc': []},
                'pr': {'recall': [], 'precision': [], 'auc': []},
                'euc_roc': [],
                'euc_pr': []
            })

        # Interpolation grid for curves
        mean_fpr = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)

        # Run bootstrap iterations
        n_samples = len(result_df)

        for bootstrap_iter in range(n_bootstrap):
            # Generate bootstrap sample with replacement
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_df = result_df.iloc[bootstrap_indices].copy()

            # For storing expert points in this iteration
            iter_expert_roc = {}
            iter_expert_pr = {}

            # For storing model curves in this iteration
            iter_model_curves = []
            for _ in model_configs:
                iter_model_curves.append({
                    'roc': {'fpr': [], 'tpr': [], 'auc': []},
                    'pr': {'recall': [], 'precision': [], 'auc': []}
                })

            # Process each expert
            for expert in expert_columns:
                exclude_col = expert.replace("stage", "exclude")

                # Skip if exclude column doesn't exist
                if exclude_col not in bootstrap_df.columns:
                    continue

                # Calculate expert performance metrics
                tp = ((bootstrap_df[expert] == class_id) & (bootstrap_df[exclude_col] == class_id)).sum()
                fp = ((bootstrap_df[expert] == class_id) & (bootstrap_df[exclude_col] != class_id)).sum()
                fn = ((bootstrap_df[expert] != class_id) & (bootstrap_df[exclude_col] == class_id)).sum()
                tn = ((bootstrap_df[expert] != class_id) & (bootstrap_df[exclude_col] != class_id)).sum()

                # Calculate TPR and FPR
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                # Calculate Precision and Recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tpr  # Recall and TPR are the same

                # Store expert point for this bootstrap iteration
                expert_bootstrap_roc[expert]['fpr'].append(fpr)
                expert_bootstrap_roc[expert]['tpr'].append(tpr)
                expert_bootstrap_pr[expert]['recall'].append(recall)
                expert_bootstrap_pr[expert]['precision'].append(precision)

                # Store for current iteration's EUC calculation
                iter_expert_roc[expert] = (fpr, tpr)
                iter_expert_pr[expert] = (recall, precision)

                # Get ground truth for evaluating models against current expert
                y_true_expert = bootstrap_df[exclude_col].values

                # Convert to binary format for ROC/PR calculation
                y_true_bin_expert = label_binarize(y_true_expert, classes=[i for i in range(n_classes)])

                # Process each model
                for model_idx, model_config in enumerate(model_configs):
                    model_name = model_config["name"]

                    # Get model predictions
                    columns_name = [f'{model_name}_class_{i}_prob' for i in range(n_classes)]
                    y_pred_prob = bootstrap_df[columns_name].values

                    try:
                        # Calculate ROC curve
                        fpr_model, tpr_model, _ = roc_curve(y_true_bin_expert[:, class_id], y_pred_prob[:, class_id])

                        # Calculate PR curve
                        precision_model, recall_model, _ = precision_recall_curve(y_true_bin_expert[:, class_id],
                                                                                  y_pred_prob[:, class_id])

                        # Calculate AUCs
                        roc_auc = auc(fpr_model, tpr_model)
                        pr_auc = average_precision_score(y_true_bin_expert[:, class_id], y_pred_prob[:, class_id])

                        # Store for current iteration
                        iter_model_curves[model_idx]['roc']['fpr'].append(fpr_model)
                        iter_model_curves[model_idx]['roc']['tpr'].append(tpr_model)
                        iter_model_curves[model_idx]['roc']['auc'].append(roc_auc)

                        iter_model_curves[model_idx]['pr']['recall'].append(recall_model)
                        iter_model_curves[model_idx]['pr']['precision'].append(precision_model)
                        iter_model_curves[model_idx]['pr']['auc'].append(pr_auc)

                        # Interpolate for confidence interval calculation
                        if len(fpr_model) > 1:
                            interp_tpr = np.interp(mean_fpr, fpr_model, tpr_model)
                            interp_tpr[0] = 0.0  # Force start at 0,0
                            model_bootstrap[model_idx]['roc']['tpr'].append(interp_tpr)

                        if len(recall_model) > 1:
                            # Reverse arrays for PR curve interpolation
                            recall_rev = recall_model[::-1]
                            precision_rev = precision_model[::-1]
                            interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
                            model_bootstrap[model_idx]['pr']['precision'].append(interp_precision)

                    except Exception as e:
                        print(f"Error in bootstrap {bootstrap_iter}, expert {expert}, model {model_name}: {e}")
                        continue

            # Calculate average model curves for this bootstrap iteration
            for model_idx, _ in enumerate(model_configs):
                # Process ROC curves
                if iter_model_curves[model_idx]['roc']['fpr']:
                    # Calculate average AUC for this iteration
                    avg_auc_roc = np.mean(iter_model_curves[model_idx]['roc']['auc'])
                    model_bootstrap[model_idx]['roc']['auc'].append(avg_auc_roc)

                    # Calculate average interpolated TPR
                    iter_tprs = []
                    for fpr, tpr in zip(iter_model_curves[model_idx]['roc']['fpr'],
                                        iter_model_curves[model_idx]['roc']['tpr']):
                        if len(fpr) > 1:
                            interp_tpr = np.interp(mean_fpr, fpr, tpr)
                            interp_tpr[0] = 0.0  # Force start at 0,0
                            iter_tprs.append(interp_tpr)

                    if iter_tprs:
                        avg_tpr = np.mean(iter_tprs, axis=0)
                        model_bootstrap[model_idx]['roc']['fpr'].append(mean_fpr)
                        model_bootstrap[model_idx]['roc']['tpr'].append(avg_tpr)

                # Process PR curves
                if iter_model_curves[model_idx]['pr']['recall']:
                    # Calculate average AUC for this iteration
                    avg_auc_pr = np.mean(iter_model_curves[model_idx]['pr']['auc'])
                    model_bootstrap[model_idx]['pr']['auc'].append(avg_auc_pr)

                    # Calculate average interpolated precision
                    iter_precisions = []
                    for recall, precision in zip(iter_model_curves[model_idx]['pr']['recall'],
                                                 iter_model_curves[model_idx]['pr']['precision']):
                        if len(recall) > 1:
                            # Reverse arrays for interpolation
                            recall_rev = recall[::-1]
                            precision_rev = precision[::-1]
                            interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
                            iter_precisions.append(interp_precision)

                    if iter_precisions:
                        avg_precision = np.mean(iter_precisions, axis=0)
                        model_bootstrap[model_idx]['pr']['recall'].append(mean_recall)
                        model_bootstrap[model_idx]['pr']['precision'].append(avg_precision)

            # Calculate EUC for this bootstrap iteration
            for model_idx, _ in enumerate(model_configs):
                experts_above_curve_roc = 0
                experts_above_curve_pr = 0
                valid_experts_count = 0

                # Skip if no model curves for this iteration
                if not model_bootstrap[model_idx]['roc']['fpr'] or not model_bootstrap[model_idx]['pr']['recall']:
                    continue

                # Get the model curves for this iteration
                model_fpr = model_bootstrap[model_idx]['roc']['fpr'][-1]  # Last added is current iteration
                model_tpr = model_bootstrap[model_idx]['roc']['tpr'][-1]
                model_recall = model_bootstrap[model_idx]['pr']['recall'][-1]
                model_precision = model_bootstrap[model_idx]['pr']['precision'][-1]

                # Compare each expert point to the model curve
                for expert in expert_columns:
                    if expert in iter_expert_roc and expert in iter_expert_pr:
                        valid_experts_count += 1

                        # Get expert points
                        expert_fpr, expert_tpr = iter_expert_roc[expert]
                        expert_recall, expert_precision = iter_expert_pr[expert]

                        # Check if expert is above ROC curve
                        expected_tpr = np.interp(expert_fpr, model_fpr, model_tpr)
                        if expert_tpr > expected_tpr:
                            experts_above_curve_roc += 1

                        # Check if expert is above PR curve
                        expected_precision = np.interp(expert_recall, model_recall, model_precision)
                        if expert_precision > expected_precision:
                            experts_above_curve_pr += 1

                # Calculate EUC percentages
                if valid_experts_count > 0:
                    euc_roc = (valid_experts_count - experts_above_curve_roc) / valid_experts_count * 100
                    euc_pr = (valid_experts_count - experts_above_curve_pr) / valid_experts_count * 100

                    model_bootstrap[model_idx]['euc_roc'].append(euc_roc)
                    model_bootstrap[model_idx]['euc_pr'].append(euc_pr)

        # Calculate mean expert points and confidence intervals
        expert_mean_roc = {}
        expert_mean_pr = {}
        expert_ci_roc = {}
        expert_ci_pr = {}

        for expert in expert_columns:
            # Calculate means if data exists
            if expert_bootstrap_roc[expert]['fpr'] and expert_bootstrap_roc[expert]['tpr']:
                mean_fpr_expert = np.mean(expert_bootstrap_roc[expert]['fpr'])
                mean_tpr_expert = np.mean(expert_bootstrap_roc[expert]['tpr'])

                # Calculate 95% confidence intervals
                ci_fpr_lower = np.percentile(expert_bootstrap_roc[expert]['fpr'], 2.5)
                ci_fpr_upper = np.percentile(expert_bootstrap_roc[expert]['fpr'], 97.5)
                ci_tpr_lower = np.percentile(expert_bootstrap_roc[expert]['tpr'], 2.5)
                ci_tpr_upper = np.percentile(expert_bootstrap_roc[expert]['tpr'], 97.5)

                expert_mean_roc[expert] = (mean_fpr_expert, mean_tpr_expert)
                expert_ci_roc[expert] = ((ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper))
            else:
                expert_mean_roc[expert] = (0, 0)
                expert_ci_roc[expert] = ((0, 0), (0, 0))

            # Same for PR curve
            if expert_bootstrap_pr[expert]['recall'] and expert_bootstrap_pr[expert]['precision']:
                mean_recall_expert = np.mean(expert_bootstrap_pr[expert]['recall'])
                mean_precision_expert = np.mean(expert_bootstrap_pr[expert]['precision'])

                # Calculate 95% confidence intervals
                ci_recall_lower = np.percentile(expert_bootstrap_pr[expert]['recall'], 2.5)
                ci_recall_upper = np.percentile(expert_bootstrap_pr[expert]['recall'], 97.5)
                ci_precision_lower = np.percentile(expert_bootstrap_pr[expert]['precision'], 2.5)
                ci_precision_upper = np.percentile(expert_bootstrap_pr[expert]['precision'], 97.5)

                expert_mean_pr[expert] = (mean_recall_expert, mean_precision_expert)
                expert_ci_pr[expert] = ((ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper))
            else:
                expert_mean_pr[expert] = (0, 0)
                expert_ci_pr[expert] = ((0, 0), (0, 0))

        # Calculate and store model statistics
        for model_idx, model_config in enumerate(model_configs):
            model_name = model_config["name"]
            display_name = model_config["display_name"]

            # ROC AUC statistics
            mean_auc_roc = np.mean(model_bootstrap[model_idx]['roc']['auc']) if model_bootstrap[model_idx]['roc'][
                'auc'] else 0
            ci_lower_auc_roc = np.percentile(model_bootstrap[model_idx]['roc']['auc'], 2.5) if \
            model_bootstrap[model_idx]['roc']['auc'] else 0
            ci_upper_auc_roc = np.percentile(model_bootstrap[model_idx]['roc']['auc'], 97.5) if \
            model_bootstrap[model_idx]['roc']['auc'] else 0

            # PR AUC statistics
            mean_auc_pr = np.mean(model_bootstrap[model_idx]['pr']['auc']) if model_bootstrap[model_idx]['pr'][
                'auc'] else 0
            ci_lower_auc_pr = np.percentile(model_bootstrap[model_idx]['pr']['auc'], 2.5) if \
            model_bootstrap[model_idx]['pr']['auc'] else 0
            ci_upper_auc_pr = np.percentile(model_bootstrap[model_idx]['pr']['auc'], 97.5) if \
            model_bootstrap[model_idx]['pr']['auc'] else 0

            # EUC statistics
            mean_euc_roc = np.mean(model_bootstrap[model_idx]['euc_roc']) if model_bootstrap[model_idx][
                'euc_roc'] else 0
            ci_lower_euc_roc = np.percentile(model_bootstrap[model_idx]['euc_roc'], 2.5) if model_bootstrap[model_idx][
                'euc_roc'] else 0
            ci_upper_euc_roc = np.percentile(model_bootstrap[model_idx]['euc_roc'], 97.5) if model_bootstrap[model_idx][
                'euc_roc'] else 0

            mean_euc_pr = np.mean(model_bootstrap[model_idx]['euc_pr']) if model_bootstrap[model_idx]['euc_pr'] else 0
            ci_lower_euc_pr = np.percentile(model_bootstrap[model_idx]['euc_pr'], 2.5) if model_bootstrap[model_idx][
                'euc_pr'] else 0
            ci_upper_euc_pr = np.percentile(model_bootstrap[model_idx]['euc_pr'], 97.5) if model_bootstrap[model_idx][
                'euc_pr'] else 0

            # Store in results dictionary
            if model_name not in results[class_name]:
                results[class_name][model_name] = {}

            results[class_name][model_name]['roc_auc'] = {
                'mean': mean_auc_roc,
                'min': ci_lower_auc_roc,
                'max': ci_upper_auc_roc
            }
            results[class_name][model_name]['pr_auc'] = {
                'mean': mean_auc_pr,
                'min': ci_lower_auc_pr,
                'max': ci_upper_auc_pr
            }
            results[class_name][model_name]['euc_roc'] = {
                'mean': mean_euc_roc,
                'min': ci_lower_euc_roc,
                'max': ci_upper_euc_roc
            }
            results[class_name][model_name]['euc_pr'] = {
                'mean': mean_euc_pr,
                'min': ci_lower_euc_pr,
                'max': ci_upper_euc_pr
            }

            # Add to formatted output
            formatted_output += f"\n{class_name} - {display_name}:\n"
            formatted_output += f"ROC AUC: {mean_auc_roc:.3f}({ci_lower_auc_roc:.3f}, {ci_upper_auc_roc:.3f})\n"
            formatted_output += f"PR AUC: {mean_auc_pr:.3f}({ci_lower_auc_pr:.3f}, {ci_upper_auc_pr:.3f})\n"
            formatted_output += f"ROC EUC: {mean_euc_roc:.1f}%({ci_lower_euc_roc:.1f}%, {ci_upper_euc_roc:.1f}%)\n"
            formatted_output += f"PR EUC: {mean_euc_pr:.1f}%({ci_lower_euc_pr:.1f}%, {ci_upper_euc_pr:.1f}%)\n"

            # Draw ROC curve with bootstrap confidence intervals
            plt.sca(ax_roc)

            if model_bootstrap[model_idx]['roc']['tpr']:
                bootstrap_tprs_array = np.array(model_bootstrap[model_idx]['roc']['tpr'])
                mean_tpr = np.mean(bootstrap_tprs_array, axis=0)
                tpr_lower = np.percentile(bootstrap_tprs_array, 2.5, axis=0)
                tpr_upper = np.percentile(bootstrap_tprs_array, 97.5, axis=0)

                # Plot mean ROC curve
                plt.plot(mean_fpr, mean_tpr, color=model_config['color'],
                         label=f'  {display_name}\nAUC={mean_auc_roc:.3f}\nEUC={mean_euc_roc:.1f}%({ci_lower_euc_roc:.1f}%-{ci_upper_euc_roc:.1f}%)',
                         lw=3)

                # Draw confidence bands
                plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=model_config['color'], alpha=0.3)

            # Draw PR curve with bootstrap confidence intervals
            plt.sca(ax_pr)

            if model_bootstrap[model_idx]['pr']['precision']:
                bootstrap_precisions_array = np.array(model_bootstrap[model_idx]['pr']['precision'])
                mean_precision = np.mean(bootstrap_precisions_array, axis=0)
                precision_lower = np.percentile(bootstrap_precisions_array, 2.5, axis=0)
                precision_upper = np.percentile(bootstrap_precisions_array, 97.5, axis=0)

                # Plot mean PR curve
                plt.plot(mean_recall, mean_precision, color=model_config['color'],
                         label=f'  {display_name}\nAUC={mean_auc_pr:.3f}\nEUC={mean_euc_pr:.1f}%({ci_lower_euc_pr:.1f}%-{ci_upper_euc_pr:.1f}%)',
                         lw=3)

                # Draw confidence bands
                plt.fill_between(mean_recall, precision_lower, precision_upper, color=model_config['color'], alpha=0.3)

        # Draw expert points with confidence intervals - ROC
        plt.sca(ax_roc)
        for expert in expert_columns:
            if expert in expert_mean_roc:
                # Draw mean expert point
                mean_fpr_expert, mean_tpr_expert = expert_mean_roc[expert]
                plt.scatter(mean_fpr_expert, mean_tpr_expert, marker='^', color=expert_color, alpha=0.7, s=60)

                # Draw confidence interval crosses
                (ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper) = expert_ci_roc[expert]

                # Draw horizontal line (TPR range)
                plt.plot([mean_fpr_expert, mean_fpr_expert], [ci_tpr_lower, ci_tpr_upper],
                         color=expert_color, alpha=0.5, linewidth=1)

                # Draw vertical line (FPR range)
                plt.plot([ci_fpr_lower, ci_fpr_upper], [mean_tpr_expert, mean_tpr_expert],
                         color=expert_color, alpha=0.5, linewidth=1)

        # Hide axis labels
        plt.xlabel('')
        plt.ylabel('')

        # Only show y-axis ticks for first column
        if index != 0:
            ax_roc.set_yticklabels([])

        # Hide x-axis ticks (since this is not the last row)
        ax_roc.set_xticklabels([])

        # Add legend
        plt.legend(loc='lower right', handlelength=0, handletextpad=0, fontsize=17,
                   frameon=False, title=f'{label_map[class_id]}', title_fontsize=17)

        # Set axis limits
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # Draw expert points with confidence intervals - PR
        plt.sca(ax_pr)
        for expert in expert_columns:
            if expert in expert_mean_pr:
                # Draw mean expert point
                mean_recall_expert, mean_precision_expert = expert_mean_pr[expert]
                plt.scatter(mean_recall_expert, mean_precision_expert, marker='^', color=expert_color, alpha=0.7, s=60)

                # Draw confidence interval crosses
                (ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper) = expert_ci_pr[expert]

                # Draw horizontal line (precision range)
                plt.plot([mean_recall_expert, mean_recall_expert], [ci_precision_lower, ci_precision_upper],
                         color=expert_color, alpha=0.5, linewidth=1)

                # Draw vertical line (recall range)
                plt.plot([ci_recall_lower, ci_recall_upper], [mean_precision_expert, mean_precision_expert],
                         color=expert_color, alpha=0.5, linewidth=1)

        # Hide axis labels
        plt.xlabel('')
        plt.ylabel('')

        # Only show y-axis ticks for first column
        if index != 0:
            ax_pr.set_yticklabels([])

        # Add legend
        plt.legend(loc='lower left', handlelength=0, handletextpad=0, fontsize=17,
                   frameon=False, title=f'{label_map[class_id]}', title_fontsize=17)

        # Set axis limits
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    # Print formatted output
    print(formatted_output)

    return results, formatted_output


def plot_auc_euc_with_bootstrap_UPenn(output_path, result_file, n_bootstrap=1000):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    from sklearn.preprocessing import label_binarize

    # String to collect formatted output
    formatted_output = ""

    # Dictionary to store results
    results = {}

    label_map = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    # Initialize results dictionary
    for class_name in label_map.values():
        results[class_name] = {}

    # Model configurations
    model_configs = [
        {"name": "M", "color": "steelblue", "display_name": "Morgoth"},
        {"name": "U1", "color": "orange", "display_name": "U-Sleep1"},
        {"name": "U2", "color": "moccasin", "display_name": "U-Sleep2"}
    ]

    class_names = list(label_map.values())
    n_classes = len(class_names)

    result_df = pd.read_csv(result_file)
    result_df.rename(columns={'stage_expert_majority': 'expert_majority'}, inplace=True)

    expert_columns = [c for c in result_df.columns if c.startswith('stage')]

    # Create a big figure for all sleep stages ROC and PR curves
    plt.figure(figsize=(4 * n_classes, 8))

    # Expert color
    expert_color = 'gray'

    # Process each label
    for index, class_id in enumerate(label_map.keys()):
        class_name = label_map[class_id]

        # Initialize ROC and PR subplots
        ax_roc = plt.subplot(2, n_classes, class_id + 1)
        ax_pr = plt.subplot(2, n_classes, n_classes + class_id + 1)

        # Initialize bootstrap storage for expert points
        expert_bootstrap_roc = {expert: {'fpr': [], 'tpr': []} for expert in expert_columns}
        expert_bootstrap_pr = {expert: {'recall': [], 'precision': []} for expert in expert_columns}

        # For each model, initialize bootstrap storage
        model_bootstrap = []
        for _ in model_configs:
            model_bootstrap.append({
                'roc': {'fpr': [], 'tpr': [], 'auc': []},
                'pr': {'recall': [], 'precision': [], 'auc': []},
                'euc_roc': [],
                'euc_pr': []
            })

        # Interpolation grid for curves
        mean_fpr = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)

        # Run bootstrap iterations
        n_samples = len(result_df)

        for bootstrap_iter in range(n_bootstrap):
            # Generate bootstrap sample with replacement
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_df = result_df.iloc[bootstrap_indices].copy()

            # For storing expert points in this iteration
            iter_expert_roc = {}
            iter_expert_pr = {}

            # For storing model curves in this iteration
            iter_model_curves = []
            for _ in model_configs:
                iter_model_curves.append({
                    'roc': {'fpr': [], 'tpr': [], 'auc': []},
                    'pr': {'recall': [], 'precision': [], 'auc': []}
                })

            # Process each expert
            for expert in expert_columns:
                exclude_col = expert.replace("stage", "exclude")

                # Skip if exclude column doesn't exist
                if exclude_col not in bootstrap_df.columns:
                    continue

                # Calculate expert performance metrics
                tp = ((bootstrap_df[expert] == class_id) & (bootstrap_df[exclude_col] == class_id)).sum()
                fp = ((bootstrap_df[expert] == class_id) & (bootstrap_df[exclude_col] != class_id)).sum()
                fn = ((bootstrap_df[expert] != class_id) & (bootstrap_df[exclude_col] == class_id)).sum()
                tn = ((bootstrap_df[expert] != class_id) & (bootstrap_df[exclude_col] != class_id)).sum()

                # Calculate TPR and FPR
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                # Calculate Precision and Recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tpr  # Recall and TPR are the same

                # Store expert point for this bootstrap iteration
                expert_bootstrap_roc[expert]['fpr'].append(fpr)
                expert_bootstrap_roc[expert]['tpr'].append(tpr)
                expert_bootstrap_pr[expert]['recall'].append(recall)
                expert_bootstrap_pr[expert]['precision'].append(precision)

                # Store for current iteration's EUC calculation
                iter_expert_roc[expert] = (fpr, tpr)
                iter_expert_pr[expert] = (recall, precision)

                # Get ground truth for evaluating models against current expert
                y_true_expert = bootstrap_df[exclude_col].values

                # Convert to binary format for ROC/PR calculation
                y_true_bin_expert = label_binarize(y_true_expert, classes=[i for i in range(n_classes)])

                # Process each model
                for model_idx, model_config in enumerate(model_configs):
                    model_name = model_config["name"]

                    # Get model predictions
                    columns_name = [f'{model_name}_class_{i}_prob' for i in range(n_classes)]
                    y_pred_prob = bootstrap_df[columns_name].values

                    try:
                        # Calculate ROC curve
                        fpr_model, tpr_model, _ = roc_curve(y_true_bin_expert[:, class_id], y_pred_prob[:, class_id])

                        # Calculate PR curve
                        precision_model, recall_model, _ = precision_recall_curve(y_true_bin_expert[:, class_id],
                                                                                  y_pred_prob[:, class_id])

                        # Calculate AUCs
                        roc_auc = auc(fpr_model, tpr_model)
                        pr_auc = average_precision_score(y_true_bin_expert[:, class_id], y_pred_prob[:, class_id])

                        # Store for current iteration
                        iter_model_curves[model_idx]['roc']['fpr'].append(fpr_model)
                        iter_model_curves[model_idx]['roc']['tpr'].append(tpr_model)
                        iter_model_curves[model_idx]['roc']['auc'].append(roc_auc)

                        iter_model_curves[model_idx]['pr']['recall'].append(recall_model)
                        iter_model_curves[model_idx]['pr']['precision'].append(precision_model)
                        iter_model_curves[model_idx]['pr']['auc'].append(pr_auc)

                        # Interpolate for confidence interval calculation
                        if len(fpr_model) > 1:
                            interp_tpr = np.interp(mean_fpr, fpr_model, tpr_model)
                            interp_tpr[0] = 0.0  # Force start at 0,0
                            model_bootstrap[model_idx]['roc']['tpr'].append(interp_tpr)

                        if len(recall_model) > 1:
                            # Reverse arrays for PR curve interpolation
                            recall_rev = recall_model[::-1]
                            precision_rev = precision_model[::-1]
                            interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
                            model_bootstrap[model_idx]['pr']['precision'].append(interp_precision)

                    except Exception as e:
                        print(f"Error in bootstrap {bootstrap_iter}, expert {expert}, model {model_name}: {e}")
                        continue

            # Calculate average model curves for this bootstrap iteration
            for model_idx, _ in enumerate(model_configs):
                # Process ROC curves
                if iter_model_curves[model_idx]['roc']['fpr']:
                    # Calculate average AUC for this iteration
                    avg_auc_roc = np.mean(iter_model_curves[model_idx]['roc']['auc'])
                    model_bootstrap[model_idx]['roc']['auc'].append(avg_auc_roc)

                    # Calculate average interpolated TPR
                    iter_tprs = []
                    for fpr, tpr in zip(iter_model_curves[model_idx]['roc']['fpr'],
                                        iter_model_curves[model_idx]['roc']['tpr']):
                        if len(fpr) > 1:
                            interp_tpr = np.interp(mean_fpr, fpr, tpr)
                            interp_tpr[0] = 0.0  # Force start at 0,0
                            iter_tprs.append(interp_tpr)

                    if iter_tprs:
                        avg_tpr = np.mean(iter_tprs, axis=0)
                        model_bootstrap[model_idx]['roc']['fpr'].append(mean_fpr)
                        model_bootstrap[model_idx]['roc']['tpr'].append(avg_tpr)

                # Process PR curves
                if iter_model_curves[model_idx]['pr']['recall']:
                    # Calculate average AUC for this iteration
                    avg_auc_pr = np.mean(iter_model_curves[model_idx]['pr']['auc'])
                    model_bootstrap[model_idx]['pr']['auc'].append(avg_auc_pr)

                    # Calculate average interpolated precision
                    iter_precisions = []
                    for recall, precision in zip(iter_model_curves[model_idx]['pr']['recall'],
                                                 iter_model_curves[model_idx]['pr']['precision']):
                        if len(recall) > 1:
                            # Reverse arrays for interpolation
                            recall_rev = recall[::-1]
                            precision_rev = precision[::-1]
                            interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
                            iter_precisions.append(interp_precision)

                    if iter_precisions:
                        avg_precision = np.mean(iter_precisions, axis=0)
                        model_bootstrap[model_idx]['pr']['recall'].append(mean_recall)
                        model_bootstrap[model_idx]['pr']['precision'].append(avg_precision)

            # Calculate EUC for this bootstrap iteration
            for model_idx, _ in enumerate(model_configs):
                experts_above_curve_roc = 0
                experts_above_curve_pr = 0
                valid_experts_count = 0

                # Skip if no model curves for this iteration
                if not model_bootstrap[model_idx]['roc']['fpr'] or not model_bootstrap[model_idx]['pr']['recall']:
                    continue

                # Get the model curves for this iteration
                model_fpr = model_bootstrap[model_idx]['roc']['fpr'][-1]  # Last added is current iteration
                model_tpr = model_bootstrap[model_idx]['roc']['tpr'][-1]
                model_recall = model_bootstrap[model_idx]['pr']['recall'][-1]
                model_precision = model_bootstrap[model_idx]['pr']['precision'][-1]

                # Compare each expert point to the model curve
                for expert in expert_columns:
                    if expert in iter_expert_roc and expert in iter_expert_pr:
                        valid_experts_count += 1

                        # Get expert points
                        expert_fpr, expert_tpr = iter_expert_roc[expert]
                        expert_recall, expert_precision = iter_expert_pr[expert]

                        # Check if expert is above ROC curve
                        expected_tpr = np.interp(expert_fpr, model_fpr, model_tpr)
                        if expert_tpr > expected_tpr:
                            experts_above_curve_roc += 1

                        # Check if expert is above PR curve
                        expected_precision = np.interp(expert_recall, model_recall, model_precision)
                        if expert_precision > expected_precision:
                            experts_above_curve_pr += 1

                # Calculate EUC percentages
                if valid_experts_count > 0:
                    euc_roc = (valid_experts_count - experts_above_curve_roc) / valid_experts_count * 100
                    euc_pr = (valid_experts_count - experts_above_curve_pr) / valid_experts_count * 100

                    model_bootstrap[model_idx]['euc_roc'].append(euc_roc)
                    model_bootstrap[model_idx]['euc_pr'].append(euc_pr)

        # Calculate mean expert points and confidence intervals
        expert_mean_roc = {}
        expert_mean_pr = {}
        expert_ci_roc = {}
        expert_ci_pr = {}

        for expert in expert_columns:
            # Calculate means if data exists
            if expert_bootstrap_roc[expert]['fpr'] and expert_bootstrap_roc[expert]['tpr']:
                mean_fpr_expert = np.mean(expert_bootstrap_roc[expert]['fpr'])
                mean_tpr_expert = np.mean(expert_bootstrap_roc[expert]['tpr'])

                # Calculate 95% confidence intervals
                ci_fpr_lower = np.percentile(expert_bootstrap_roc[expert]['fpr'], 2.5)
                ci_fpr_upper = np.percentile(expert_bootstrap_roc[expert]['fpr'], 97.5)
                ci_tpr_lower = np.percentile(expert_bootstrap_roc[expert]['tpr'], 2.5)
                ci_tpr_upper = np.percentile(expert_bootstrap_roc[expert]['tpr'], 97.5)

                expert_mean_roc[expert] = (mean_fpr_expert, mean_tpr_expert)
                expert_ci_roc[expert] = ((ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper))
            else:
                expert_mean_roc[expert] = (0, 0)
                expert_ci_roc[expert] = ((0, 0), (0, 0))

            # Same for PR curve
            if expert_bootstrap_pr[expert]['recall'] and expert_bootstrap_pr[expert]['precision']:
                mean_recall_expert = np.mean(expert_bootstrap_pr[expert]['recall'])
                mean_precision_expert = np.mean(expert_bootstrap_pr[expert]['precision'])

                # Calculate 95% confidence intervals
                ci_recall_lower = np.percentile(expert_bootstrap_pr[expert]['recall'], 2.5)
                ci_recall_upper = np.percentile(expert_bootstrap_pr[expert]['recall'], 97.5)
                ci_precision_lower = np.percentile(expert_bootstrap_pr[expert]['precision'], 2.5)
                ci_precision_upper = np.percentile(expert_bootstrap_pr[expert]['precision'], 97.5)

                expert_mean_pr[expert] = (mean_recall_expert, mean_precision_expert)
                expert_ci_pr[expert] = ((ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper))
            else:
                expert_mean_pr[expert] = (0, 0)
                expert_ci_pr[expert] = ((0, 0), (0, 0))

        # Calculate and store model statistics
        for model_idx, model_config in enumerate(model_configs):
            model_name = model_config["name"]
            display_name = model_config["display_name"]

            # ROC AUC statistics
            mean_auc_roc = np.mean(model_bootstrap[model_idx]['roc']['auc']) if model_bootstrap[model_idx]['roc'][
                'auc'] else 0
            ci_lower_auc_roc = np.percentile(model_bootstrap[model_idx]['roc']['auc'], 2.5) if \
            model_bootstrap[model_idx]['roc']['auc'] else 0
            ci_upper_auc_roc = np.percentile(model_bootstrap[model_idx]['roc']['auc'], 97.5) if \
            model_bootstrap[model_idx]['roc']['auc'] else 0

            # PR AUC statistics
            mean_auc_pr = np.mean(model_bootstrap[model_idx]['pr']['auc']) if model_bootstrap[model_idx]['pr'][
                'auc'] else 0
            ci_lower_auc_pr = np.percentile(model_bootstrap[model_idx]['pr']['auc'], 2.5) if \
            model_bootstrap[model_idx]['pr']['auc'] else 0
            ci_upper_auc_pr = np.percentile(model_bootstrap[model_idx]['pr']['auc'], 97.5) if \
            model_bootstrap[model_idx]['pr']['auc'] else 0

            # EUC statistics
            mean_euc_roc = np.mean(model_bootstrap[model_idx]['euc_roc']) if model_bootstrap[model_idx][
                'euc_roc'] else 0
            ci_lower_euc_roc = np.percentile(model_bootstrap[model_idx]['euc_roc'], 2.5) if model_bootstrap[model_idx][
                'euc_roc'] else 0
            ci_upper_euc_roc = np.percentile(model_bootstrap[model_idx]['euc_roc'], 97.5) if model_bootstrap[model_idx][
                'euc_roc'] else 0

            mean_euc_pr = np.mean(model_bootstrap[model_idx]['euc_pr']) if model_bootstrap[model_idx]['euc_pr'] else 0
            ci_lower_euc_pr = np.percentile(model_bootstrap[model_idx]['euc_pr'], 2.5) if model_bootstrap[model_idx][
                'euc_pr'] else 0
            ci_upper_euc_pr = np.percentile(model_bootstrap[model_idx]['euc_pr'], 97.5) if model_bootstrap[model_idx][
                'euc_pr'] else 0

            # Store in results dictionary
            if model_name not in results[class_name]:
                results[class_name][model_name] = {}

            results[class_name][model_name]['roc_auc'] = {
                'mean': mean_auc_roc,
                'min': ci_lower_auc_roc,
                'max': ci_upper_auc_roc
            }
            results[class_name][model_name]['pr_auc'] = {
                'mean': mean_auc_pr,
                'min': ci_lower_auc_pr,
                'max': ci_upper_auc_pr
            }
            results[class_name][model_name]['euc_roc'] = {
                'mean': mean_euc_roc,
                'min': ci_lower_euc_roc,
                'max': ci_upper_euc_roc
            }
            results[class_name][model_name]['euc_pr'] = {
                'mean': mean_euc_pr,
                'min': ci_lower_euc_pr,
                'max': ci_upper_euc_pr
            }

            # Add to formatted output
            formatted_output += f"\n{class_name} - {display_name}:\n"
            formatted_output += f"ROC AUC: {mean_auc_roc:.3f}({ci_lower_auc_roc:.3f}, {ci_upper_auc_roc:.3f})\n"
            formatted_output += f"PR AUC: {mean_auc_pr:.3f}({ci_lower_auc_pr:.3f}, {ci_upper_auc_pr:.3f})\n"
            formatted_output += f"ROC EUC: {mean_euc_roc:.1f}%({ci_lower_euc_roc:.1f}%, {ci_upper_euc_roc:.1f}%)\n"
            formatted_output += f"PR EUC: {mean_euc_pr:.1f}%({ci_lower_euc_pr:.1f}%, {ci_upper_euc_pr:.1f}%)\n"

            # Draw ROC curve with bootstrap confidence intervals
            plt.sca(ax_roc)

            if model_bootstrap[model_idx]['roc']['tpr']:
                bootstrap_tprs_array = np.array(model_bootstrap[model_idx]['roc']['tpr'])
                mean_tpr = np.mean(bootstrap_tprs_array, axis=0)
                tpr_lower = np.percentile(bootstrap_tprs_array, 2.5, axis=0)
                tpr_upper = np.percentile(bootstrap_tprs_array, 97.5, axis=0)

                # Plot mean ROC curve
                plt.plot(mean_fpr, mean_tpr, color=model_config['color'],
                         label=f'  {display_name}\nAUC={mean_auc_roc:.3f}\nEUC={mean_euc_roc:.1f}%',
                         lw=3)

                # Draw confidence bands
                plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=model_config['color'], alpha=0.3)

            # Draw PR curve with bootstrap confidence intervals
            plt.sca(ax_pr)

            if model_bootstrap[model_idx]['pr']['precision']:
                bootstrap_precisions_array = np.array(model_bootstrap[model_idx]['pr']['precision'])
                mean_precision = np.mean(bootstrap_precisions_array, axis=0)
                precision_lower = np.percentile(bootstrap_precisions_array, 2.5, axis=0)
                precision_upper = np.percentile(bootstrap_precisions_array, 97.5, axis=0)

                # Plot mean PR curve
                plt.plot(mean_recall, mean_precision, color=model_config['color'],
                         label=f'  {display_name}\nAUC={mean_auc_pr:.3f}\nEUC={mean_euc_pr:.1f}%',
                         lw=3)

                # Draw confidence bands
                plt.fill_between(mean_recall, precision_lower, precision_upper, color=model_config['color'], alpha=0.3)

        # Draw expert points with confidence intervals - ROC
        plt.sca(ax_roc)
        for expert in expert_columns:
            if expert in expert_mean_roc:
                # Draw mean expert point
                mean_fpr_expert, mean_tpr_expert = expert_mean_roc[expert]
                plt.scatter(mean_fpr_expert, mean_tpr_expert, marker='o', color=expert_color, alpha=0.6, s=20)

                # Draw confidence interval crosses
                (ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper) = expert_ci_roc[expert]

                # Draw horizontal line (TPR range)
                plt.plot([mean_fpr_expert, mean_fpr_expert], [ci_tpr_lower, ci_tpr_upper],
                         color=expert_color, alpha=0.6, linewidth=1)

                # Draw vertical line (FPR range)
                plt.plot([ci_fpr_lower, ci_fpr_upper], [mean_tpr_expert, mean_tpr_expert],
                         color=expert_color, alpha=0.6, linewidth=1)

        # Hide axis labels
        plt.xlabel('')
        plt.ylabel('')

        # Only show y-axis ticks for first column
        if index != 0:
            ax_roc.set_yticklabels([])

        # Hide x-axis ticks (since this is not the last row)
        ax_roc.set_xticklabels([])

        # Add legend
        plt.legend(loc='lower right', handlelength=0, handletextpad=0, fontsize=15,
                   frameon=False, title=f'{label_map[class_id]}', title_fontsize=16)

        # Set axis limits
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # Draw expert points with confidence intervals - PR
        plt.sca(ax_pr)
        for expert in expert_columns:
            if expert in expert_mean_pr:
                # Draw mean expert point
                mean_recall_expert, mean_precision_expert = expert_mean_pr[expert]
                plt.scatter(mean_recall_expert, mean_precision_expert, marker='o', color=expert_color, alpha=0.6, s=20)

                # Draw confidence interval crosses
                (ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper) = expert_ci_pr[expert]

                # Draw horizontal line (precision range)
                plt.plot([mean_recall_expert, mean_recall_expert], [ci_precision_lower, ci_precision_upper],
                         color=expert_color, alpha=0.6, linewidth=1)

                # Draw vertical line (recall range)
                plt.plot([ci_recall_lower, ci_recall_upper], [mean_precision_expert, mean_precision_expert],
                         color=expert_color, alpha=0.6, linewidth=1)

        # Hide axis labels
        plt.xlabel('')
        plt.ylabel('')

        # Only show y-axis ticks for first column
        if index != 0:
            ax_pr.set_yticklabels([])

        # Add legend
        plt.legend(loc='lower left', handlelength=0, handletextpad=0, fontsize=15,
                   frameon=False, title=f'{label_map[class_id]}', title_fontsize=16)

        # Set axis limits
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    # Print formatted output
    print(formatted_output)

    return results, formatted_output


def plot_auc_euc_with_bootstrap_MoE(results_dir, fig_dir, n_bootstrap=1000):


    labels = [
        'SLOWING',
        'IIIC',
        'FOC_GEN_SPIKES',
        'SPIKES',
        'BS',
        'SLEEP_19channels_3stages',
    ]

    binary_label = [
        'SPIKES',
        'BS'
    ]

    label_maps = [
        {0: 'Others', 1: 'Focal Slowing', 2: 'Generalized Slowing'},
        {0: 'Others', 1: 'Seizure', 2: 'LPD', 3: 'GPD', 4: 'LRDA', 5: 'GRDA'},
        {0: 'Others', 1: 'Focal Spikes', 2: 'Generalized Spikes'},
        {0: 'Others', 1: 'SPIKES'},
        {0: 'Others', 1: 'BS'},
        {0: 'Awake', 1: 'N1', 2: 'N2'},
    ]

    # Dictionary to store all metrics for return
    results = {}

    # String to collect formatted output
    formatted_output = ""

    for label, label_map in zip(labels, label_maps):
        # Filter out the "Others" class for plotting
        filtered_label_map = {k: v for k, v in label_map.items() if v != 'Others'}
        class_names = list(label_map.values())
        filtered_class_names = list(filtered_label_map.values())
        n_classes = len(class_names)
        n_filtered_classes = len(filtered_class_names)

        result_df = pd.read_csv(os.path.join(results_dir, f'{label}_model_and_experts_results.csv'))
        expert_columns = [c for c in result_df.columns if c.startswith('expert')]

        # Set seaborn color palette
        sns.set_palette("deep")
        colors = sns.color_palette("deep", 1)

        # Create a large figure for all sleep stages' ROC and PR curves
        if label in binary_label:
            plt.figure(figsize=(4 * n_filtered_classes, 8))  # Adjust height for multiple rows
        else:
            plt.figure(figsize=(4 * n_filtered_classes, 8))  # Adjust height for multiple rows

        # Define expert point color
        expert_color = 'gray'  # Expert points use gray

        # Get model prediction probabilities
        if label in binary_label:
            columns_name = ['pred']
        else:
            columns_name = [f'class_{i}_prob' for i in range(n_classes)]

        y_pred_prob = result_df[columns_name].values

        # Track subplot index
        subplot_idx = 1

        # Store label results
        results[label] = {}

        # Iterate through each label
        for index, class_id in enumerate(filtered_label_map.keys()):
            class_name = label_map[class_id]

            # Initialize bootstrap storage for expert points
            expert_bootstrap_roc = {expert: {'fpr': [], 'tpr': []} for expert in expert_columns}
            expert_bootstrap_pr = {expert: {'recall': [], 'precision': []} for expert in expert_columns}

            # Initialize bootstrap storage for model curves
            model_bootstrap = {
                'roc': {'fpr': [], 'tpr': [], 'auc': []},
                'pr': {'recall': [], 'precision': [], 'auc': []},
                'euc_roc': [],
                'euc_pr': []
            }

            # Interpolation grid for curves
            mean_fpr = np.linspace(0, 1, 100)
            mean_recall = np.linspace(0, 1, 100)

            # Run bootstrap iterations
            n_samples = len(result_df)

            for bootstrap_iter in range(n_bootstrap):
                # Generate bootstrap sample with replacement
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_df = result_df.iloc[bootstrap_indices].copy()

                # For storing expert points in this iteration
                iter_expert_roc = {}
                iter_expert_pr = {}

                # For storing model curves in this iteration
                iter_model_curves = {
                    'roc': {'fpr': [], 'tpr': [], 'auc': []},
                    'pr': {'recall': [], 'precision': [], 'auc': []}
                }

                # Process each expert
                for expert in expert_columns:
                    exclude_col = f'exclude_{expert}'

                    # Skip if exclude column doesn't exist
                    if exclude_col not in bootstrap_df.columns:
                        continue

                    # Calculate expert performance metrics
                    tp = ((bootstrap_df[expert] == class_id) & (bootstrap_df[exclude_col] == class_id)).sum()
                    fp = ((bootstrap_df[expert] == class_id) & (bootstrap_df[exclude_col] != class_id)).sum()
                    fn = ((bootstrap_df[expert] != class_id) & (bootstrap_df[exclude_col] == class_id)).sum()
                    tn = ((bootstrap_df[expert] != class_id) & (bootstrap_df[exclude_col] != class_id)).sum()

                    # Calculate TPR and FPR
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    # Calculate Precision and Recall
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tpr  # Recall and TPR are the same

                    # Store expert point for this bootstrap iteration
                    expert_bootstrap_roc[expert]['fpr'].append(fpr)
                    expert_bootstrap_roc[expert]['tpr'].append(tpr)
                    expert_bootstrap_pr[expert]['recall'].append(recall)
                    expert_bootstrap_pr[expert]['precision'].append(precision)

                    # Store for current iteration's EUC calculation
                    iter_expert_roc[expert] = (fpr, tpr)
                    iter_expert_pr[expert] = (recall, precision)

                    # Get ground truth for evaluating model against current expert
                    y_true_expert = bootstrap_df[exclude_col].values

                    # Convert to binary format for ROC/PR calculation
                    y_true_bin_expert = label_binarize(y_true_expert, classes=[i for i in range(n_classes)])

                    # Get model predictions for this bootstrap sample
                    y_pred_bootstrap = bootstrap_df[columns_name].values

                    try:
                        # Calculate ROC curve
                        if label in binary_label:
                            fpr_model, tpr_model, _ = roc_curve(y_true_bin_expert[:, class_id - 1],
                                                                y_pred_bootstrap[:, class_id - 1])
                            precision_model, recall_model, _ = precision_recall_curve(
                                y_true_bin_expert[:, class_id - 1],
                                y_pred_bootstrap[:, class_id - 1])
                            pr_auc = average_precision_score(y_true_bin_expert[:, class_id - 1],
                                                             y_pred_bootstrap[:, class_id - 1])
                        else:
                            fpr_model, tpr_model, _ = roc_curve(y_true_bin_expert[:, class_id],
                                                                y_pred_bootstrap[:, class_id])
                            precision_model, recall_model, _ = precision_recall_curve(y_true_bin_expert[:, class_id],
                                                                                      y_pred_bootstrap[:, class_id])
                            pr_auc = average_precision_score(y_true_bin_expert[:, class_id],
                                                             y_pred_bootstrap[:, class_id])

                        # Calculate ROC AUC
                        roc_auc = auc(fpr_model, tpr_model)

                        # Store for current iteration
                        iter_model_curves['roc']['fpr'].append(fpr_model)
                        iter_model_curves['roc']['tpr'].append(tpr_model)
                        iter_model_curves['roc']['auc'].append(roc_auc)

                        iter_model_curves['pr']['recall'].append(recall_model)
                        iter_model_curves['pr']['precision'].append(precision_model)
                        iter_model_curves['pr']['auc'].append(pr_auc)

                    except Exception as e:
                        print(f"Error in bootstrap {bootstrap_iter}, expert {expert}, label {label}: {e}")
                        continue

                # Calculate average model curves for this bootstrap iteration
                if iter_model_curves['roc']['fpr']:
                    # Calculate average AUC for this iteration
                    avg_auc_roc = np.mean(iter_model_curves['roc']['auc'])
                    model_bootstrap['roc']['auc'].append(avg_auc_roc)

                    # Calculate average interpolated TPR
                    iter_tprs = []
                    for fpr, tpr in zip(iter_model_curves['roc']['fpr'],
                                        iter_model_curves['roc']['tpr']):
                        if len(fpr) > 1:
                            interp_tpr = np.interp(mean_fpr, fpr, tpr)
                            interp_tpr[0] = 0.0  # Force start at 0,0
                            iter_tprs.append(interp_tpr)

                    if iter_tprs:
                        avg_tpr = np.mean(iter_tprs, axis=0)
                        model_bootstrap['roc']['fpr'].append(mean_fpr)
                        model_bootstrap['roc']['tpr'].append(avg_tpr)

                # Process PR curves
                if iter_model_curves['pr']['recall']:
                    # Calculate average AUC for this iteration
                    avg_auc_pr = np.mean(iter_model_curves['pr']['auc'])
                    model_bootstrap['pr']['auc'].append(avg_auc_pr)

                    # Calculate average interpolated precision
                    iter_precisions = []
                    for recall, precision in zip(iter_model_curves['pr']['recall'],
                                                 iter_model_curves['pr']['precision']):
                        if len(recall) > 1:
                            # Reverse arrays for interpolation
                            recall_rev = recall[::-1]
                            precision_rev = precision[::-1]
                            interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
                            iter_precisions.append(interp_precision)

                    if iter_precisions:
                        avg_precision = np.mean(iter_precisions, axis=0)
                        model_bootstrap['pr']['recall'].append(mean_recall)
                        model_bootstrap['pr']['precision'].append(avg_precision)

                # Calculate EUC for this bootstrap iteration
                experts_above_curve_roc = 0
                experts_above_curve_pr = 0
                valid_experts_count = 0

                # Skip if no model curves for this iteration
                if not model_bootstrap['roc']['fpr'] or not model_bootstrap['pr']['recall']:
                    continue

                # Get the model curves for this iteration
                model_fpr = model_bootstrap['roc']['fpr'][-1]  # Last added is current iteration
                model_tpr = model_bootstrap['roc']['tpr'][-1]
                model_recall = model_bootstrap['pr']['recall'][-1]
                model_precision = model_bootstrap['pr']['precision'][-1]

                # Compare each expert point to the model curve
                for expert in expert_columns:
                    if expert in iter_expert_roc and expert in iter_expert_pr:
                        valid_experts_count += 1

                        # Get expert points
                        expert_fpr, expert_tpr = iter_expert_roc[expert]
                        expert_recall, expert_precision = iter_expert_pr[expert]

                        # Check if expert is above ROC curve
                        expected_tpr = np.interp(expert_fpr, model_fpr, model_tpr)
                        if expert_tpr > expected_tpr:
                            experts_above_curve_roc += 1

                        # Check if expert is above PR curve
                        expected_precision = np.interp(expert_recall, model_recall, model_precision)
                        if expert_precision > expected_precision:
                            experts_above_curve_pr += 1

                # Calculate EUC percentages
                if valid_experts_count > 0:
                    euc_roc = (valid_experts_count - experts_above_curve_roc) / valid_experts_count * 100
                    euc_pr = (valid_experts_count - experts_above_curve_pr) / valid_experts_count * 100

                    model_bootstrap['euc_roc'].append(euc_roc)
                    model_bootstrap['euc_pr'].append(euc_pr)

            # Calculate mean expert points and confidence intervals
            expert_mean_roc = {}
            expert_mean_pr = {}
            expert_ci_roc = {}
            expert_ci_pr = {}

            for expert in expert_columns:
                # Calculate means if data exists
                if expert_bootstrap_roc[expert]['fpr'] and expert_bootstrap_roc[expert]['tpr']:
                    mean_fpr_expert = np.mean(expert_bootstrap_roc[expert]['fpr'])
                    mean_tpr_expert = np.mean(expert_bootstrap_roc[expert]['tpr'])

                    # Calculate 95% confidence intervals
                    ci_fpr_lower = np.percentile(expert_bootstrap_roc[expert]['fpr'], 2.5)
                    ci_fpr_upper = np.percentile(expert_bootstrap_roc[expert]['fpr'], 97.5)
                    ci_tpr_lower = np.percentile(expert_bootstrap_roc[expert]['tpr'], 2.5)
                    ci_tpr_upper = np.percentile(expert_bootstrap_roc[expert]['tpr'], 97.5)

                    expert_mean_roc[expert] = (mean_fpr_expert, mean_tpr_expert)
                    expert_ci_roc[expert] = ((ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper))
                else:
                    expert_mean_roc[expert] = (0, 0)
                    expert_ci_roc[expert] = ((0, 0), (0, 0))

                # Same for PR curve
                if expert_bootstrap_pr[expert]['recall'] and expert_bootstrap_pr[expert]['precision']:
                    mean_recall_expert = np.mean(expert_bootstrap_pr[expert]['recall'])
                    mean_precision_expert = np.mean(expert_bootstrap_pr[expert]['precision'])

                    # Calculate 95% confidence intervals
                    ci_recall_lower = np.percentile(expert_bootstrap_pr[expert]['recall'], 2.5)
                    ci_recall_upper = np.percentile(expert_bootstrap_pr[expert]['recall'], 97.5)
                    ci_precision_lower = np.percentile(expert_bootstrap_pr[expert]['precision'], 2.5)
                    ci_precision_upper = np.percentile(expert_bootstrap_pr[expert]['precision'], 97.5)

                    expert_mean_pr[expert] = (mean_recall_expert, mean_precision_expert)
                    expert_ci_pr[expert] = (
                    (ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper))
                else:
                    expert_mean_pr[expert] = (0, 0)
                    expert_ci_pr[expert] = ((0, 0), (0, 0))

            # Calculate model statistics
            mean_auc_roc = np.mean(model_bootstrap['roc']['auc']) if model_bootstrap['roc']['auc'] else 0
            ci_lower_auc_roc = np.percentile(model_bootstrap['roc']['auc'], 2.5) if model_bootstrap['roc']['auc'] else 0
            ci_upper_auc_roc = np.percentile(model_bootstrap['roc']['auc'], 97.5) if model_bootstrap['roc'][
                'auc'] else 0

            mean_auc_pr = np.mean(model_bootstrap['pr']['auc']) if model_bootstrap['pr']['auc'] else 0
            ci_lower_auc_pr = np.percentile(model_bootstrap['pr']['auc'], 2.5) if model_bootstrap['pr']['auc'] else 0
            ci_upper_auc_pr = np.percentile(model_bootstrap['pr']['auc'], 97.5) if model_bootstrap['pr']['auc'] else 0

            # Calculate EUC statistics
            mean_euc_roc = np.mean(model_bootstrap['euc_roc']) if model_bootstrap['euc_roc'] else 0
            ci_lower_euc_roc = np.percentile(model_bootstrap['euc_roc'], 2.5) if model_bootstrap['euc_roc'] else 0
            ci_upper_euc_roc = np.percentile(model_bootstrap['euc_roc'], 97.5) if model_bootstrap['euc_roc'] else 0

            mean_euc_pr = np.mean(model_bootstrap['euc_pr']) if model_bootstrap['euc_pr'] else 0
            ci_lower_euc_pr = np.percentile(model_bootstrap['euc_pr'], 2.5) if model_bootstrap['euc_pr'] else 0
            ci_upper_euc_pr = np.percentile(model_bootstrap['euc_pr'], 97.5) if model_bootstrap['euc_pr'] else 0

            # Store metrics for this class
            results[label][class_name] = {
                'roc_auc': {
                    'mean': mean_auc_roc,
                    'min': ci_lower_auc_roc,
                    'max': ci_upper_auc_roc
                },
                'pr_auc': {
                    'mean': mean_auc_pr,
                    'min': ci_lower_auc_pr,
                    'max': ci_upper_auc_pr
                },
                'euc_roc': {
                    'mean': mean_euc_roc,
                    'min': ci_lower_euc_roc,
                    'max': ci_upper_euc_roc
                },
                'euc_pr': {
                    'mean': mean_euc_pr,
                    'min': ci_lower_euc_pr,
                    'max': ci_upper_euc_pr
                }
            }

            # Add formatted metrics to output string
            formatted_output += f"\n{label} - {class_name}:\n"
            formatted_output += f"ROC AUC: {mean_auc_roc:.3f}({ci_lower_auc_roc:.3f}, {ci_upper_auc_roc:.3f})\n"
            formatted_output += f"PR AUC: {mean_auc_pr:.3f}({ci_lower_auc_pr:.3f}, {ci_upper_auc_pr:.3f})\n"
            formatted_output += f"ROC EUC: {mean_euc_roc:.1f}%({ci_lower_euc_roc:.1f}%, {ci_upper_euc_roc:.1f}%)\n"
            formatted_output += f"PR EUC: {mean_euc_pr:.1f}%({ci_lower_euc_pr:.1f}%, {ci_upper_euc_pr:.1f}%)\n"

            # Plot ROC curve with bootstrap confidence intervals
            ax_roc = plt.subplot(2, n_filtered_classes, subplot_idx)

            if model_bootstrap['roc']['tpr']:
                bootstrap_tprs_array = np.array(model_bootstrap['roc']['tpr'])
                mean_tpr = np.mean(bootstrap_tprs_array, axis=0)
                tpr_lower = np.percentile(bootstrap_tprs_array, 2.5, axis=0)
                tpr_upper = np.percentile(bootstrap_tprs_array, 97.5, axis=0)

                plt.plot(mean_fpr, mean_tpr, color=colors[0],
                         label=f'AUC={mean_auc_roc:.3f}',
                         lw=3)

                plt.plot([], [],
                         label=f'EUC={mean_euc_roc:.1f}%')

                plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors[0], alpha=0.3)

            # Draw expert points with confidence interval crosses - ROC
            for expert in expert_columns:
                if expert in expert_mean_roc:
                    # Draw mean expert point
                    mean_fpr_expert, mean_tpr_expert = expert_mean_roc[expert]
                    plt.scatter(mean_fpr_expert, mean_tpr_expert, marker='o', color=expert_color, alpha=0.6, s=20)

                    # Draw confidence interval crosses
                    (ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper) = expert_ci_roc[expert]

                    # Draw horizontal line (TPR range)
                    plt.plot([mean_fpr_expert, mean_fpr_expert], [ci_tpr_lower, ci_tpr_upper],
                             color=expert_color, alpha=0.6, linewidth=1)

                    # Draw vertical line (FPR range)
                    plt.plot([ci_fpr_lower, ci_fpr_upper], [mean_tpr_expert, mean_tpr_expert],
                             color=expert_color, alpha=0.6, linewidth=1)

            # Remove x and y axis labels for all plots
            plt.xlabel('')
            plt.ylabel('')

            # Handle specific label cases for axis ticks
            if label == 'IIIC':
                # For IIIC, remove all ticks
                ax_roc.set_xticks([])
                ax_roc.set_yticks([])
            elif label == 'SLOWING':
                # For SLOWING, only show y-axis ticks on first column
                ax_roc.set_xticks([])
                if index > 0:  # If not the first column
                    ax_roc.set_yticks([])
            elif label == 'SPIKES':
                # For SPIKES, remove x-ticks from first row and all y-ticks
                ax_roc.set_xticks([])
                ax_roc.set_yticks([])
            elif label == 'FOC_GEN_SPIKES' or label == 'SLEEP_19channels_3stages':
                # Remove y-axis ticks
                ax_roc.set_yticks([])
                # First row always has no x ticks
                ax_roc.set_xticks([])

            # Remove separate title and add it to legend instead
            plt.legend(loc='lower right', handlelength=0, handletextpad=0, fontsize=20,
                       title=f'{label_map[class_id]}', title_fontsize=20,
                       frameon=False)  # Remove legend border

            # Set x and y axis range from 0 to 1
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            # Plot PR curve with bootstrap confidence intervals
            ax_pr = plt.subplot(2, n_filtered_classes, n_filtered_classes + subplot_idx)

            if model_bootstrap['pr']['precision']:
                bootstrap_precisions_array = np.array(model_bootstrap['pr']['precision'])
                mean_precision = np.mean(bootstrap_precisions_array, axis=0)
                precision_lower = np.percentile(bootstrap_precisions_array, 2.5, axis=0)
                precision_upper = np.percentile(bootstrap_precisions_array, 97.5, axis=0)

                plt.plot(mean_recall, mean_precision, color=colors[0],
                         label=f'AUC={mean_auc_pr:.3f}',
                         lw=3)

                plt.plot([], [],
                         label=f'EUC={mean_euc_pr:.1f}%')

                plt.fill_between(mean_recall, precision_lower, precision_upper, color=colors[0], alpha=0.3)

            # Draw expert points with confidence interval crosses - PR
            for expert in expert_columns:
                if expert in expert_mean_pr:
                    # Draw mean expert point
                    mean_recall_expert, mean_precision_expert = expert_mean_pr[expert]
                    plt.scatter(mean_recall_expert, mean_precision_expert, marker='o', color=expert_color, alpha=0.6,
                                s=20)

                    # Draw confidence interval crosses
                    (ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper) = expert_ci_pr[expert]

                    # Draw horizontal line (precision range)
                    plt.plot([mean_recall_expert, mean_recall_expert], [ci_precision_lower, ci_precision_upper],
                             color=expert_color, alpha=0.6, linewidth=1)

                    # Draw vertical line (recall range)
                    plt.plot([ci_recall_lower, ci_recall_upper], [mean_precision_expert, mean_precision_expert],
                             color=expert_color, alpha=0.6, linewidth=1)

            # Remove x and y axis labels for all plots
            plt.xlabel('')
            plt.ylabel('')

            # Handle specific label cases for axis ticks on bottom row
            if label == 'IIIC':
                # For IIIC, remove all ticks
                ax_pr.set_xticks([])
                ax_pr.set_yticks([])
            elif label == 'SLOWING':
                # For SLOWING, only show y-axis ticks on first column (remove x-ticks)
                ax_pr.set_xticks([])
                if index > 0:  # If not the first column
                    ax_pr.set_yticks([])
            elif label == 'SPIKES':
                # For SPIKES, keep x-axis ticks on bottom row but remove y-ticks
                ax_pr.set_yticks([])
            elif label == 'FOC_GEN_SPIKES' or label == 'SLEEP_19channels_3stages':
                # Remove y-axis ticks
                ax_pr.set_yticks([])
                # Keep x-axis ticks for the entire bottom row

            # Remove separate title and add it to legend instead
            plt.legend(loc='lower left', handlelength=0, handletextpad=0, fontsize=20,
                       title=f'{label_map[class_id]}', title_fontsize=20,
                       frameon=False)  # Remove legend border

            # Set x and y axis range from 0 to 1
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            # Increment subplot index
            subplot_idx += 1

        # Adjust layout and save image
        plt.tight_layout()
        # If only 2 columns, make it more compact by reducing spacing between columns and rows
        if n_filtered_classes == 2:
            plt.subplots_adjust(wspace=0.02, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
        output_path = os.path.join(fig_dir, f'MoE_{label}.png')
        plt.savefig(output_path, dpi=300)  # Save image
        plt.show()

    # Print the formatted output
    print(formatted_output)

    return results, formatted_output


def plot_auc_euc_with_bootstrap_ON(fig_path, label, results_file, n_bootstrap=1000):

    # String to collect formatted output
    formatted_output = ""

    # Dictionary to store results
    results = {}

    labels = [
        label,
    ]
    binary_label = [
        label
    ]
    label_maps = [
        {0: 'Others', 1: label},
    ]

    # Set color for the single model
    model_colors = ['steelblue']
    model_names = ['Morgoth']

    # Initialize results dictionary
    results[label] = {}
    for model_name in model_names:
        results[label][model_name] = {}

    for label_name, label_map in zip(labels, label_maps):
        class_names = list(label_map.values())
        n_classes = len(class_names)

        # Read the results file
        result_df = pd.read_excel(results_file)

        # Get all expert columns
        expert_columns = [c for c in result_df.columns if c.startswith('expert')]

        # Generate exclude columns for each expert
        for expert in expert_columns:
            # Get all other expert columns
            other_experts = [col for col in expert_columns if col != expert]

            if not other_experts:
                continue  # Skip if there are no other experts

            # Create a function to find the mode, handling NaN values
            def get_mode_excluding_nan(row):
                # Get values from other experts, exclude NaN
                values = [row[exp] for exp in other_experts if not pd.isna(row[exp])]
                # If no valid values, return -1
                if not values:
                    return -1
                # Find the most common value
                from collections import Counter
                counter = Counter(values)
                if not counter:
                    return -1
                # Get the most common value (mode)
                return counter.most_common(1)[0][0]

            # Create exclude column by finding the mode of other experts
            result_df[f'exclude_{expert}'] = result_df.apply(get_mode_excluding_nan, axis=1)

        plt.figure(figsize=(4 * (n_classes - 1), 8) if label_name in binary_label else (4 * n_classes, 8))

        expert_color = 'grey'

        # Get model predictions
        if label_name in binary_label:
            y_pred_prob_M = result_df[['M_pred']].values
        else:
            y_pred_prob_M = result_df[[f'class_{i}_prob' for i in range(n_classes)]].values

        for index, class_id in enumerate(label_map.keys()):
            if class_id == 0 and label_name in binary_label:
                continue

            class_name = label_map[class_id]

            # For storing bootstrapped expert points
            expert_bootstrap_roc = {expert: {'fpr': [], 'tpr': []} for expert in expert_columns}
            expert_bootstrap_pr = {expert: {'recall': [], 'precision': []} for expert in expert_columns}

            # For storing bootstrapped model curves
            model_bootstrap = {
                'roc': {'fpr': [], 'tpr': [], 'auc': []},
                'pr': {'recall': [], 'precision': [], 'auc': []},
                'euc_roc': [],
                'euc_pr': []
            }

            # Interpolation grid for curves
            mean_fpr = np.linspace(0, 1, 100)
            mean_recall = np.linspace(0, 1, 100)

            # Run bootstrap iterations
            n_samples = len(result_df)

            for bootstrap_iter in range(n_bootstrap):
                # Generate bootstrap sample with replacement
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_df = result_df.iloc[bootstrap_indices].copy()

                # For storing expert points in this iteration
                iter_expert_roc = {}
                iter_expert_pr = {}

                # For storing model curves in this iteration
                iter_model_curves = {
                    'roc': {'fpr': [], 'tpr': [], 'auc': []},
                    'pr': {'recall': [], 'precision': [], 'auc': []}
                }

                # Process each expert
                for expert in expert_columns:
                    # Skip rows where exclude_{expert} is -1 or NaN
                    valid_rows = bootstrap_df[(bootstrap_df[f'exclude_{expert}'] != -1) &
                                              (~bootstrap_df[f'exclude_{expert}'].isna()) &
                                              (~bootstrap_df[expert].isna())]

                    # If no valid rows, skip this expert
                    if len(valid_rows) == 0:
                        continue

                    # Calculate expert performance metrics
                    tp = ((valid_rows[expert] == class_id) & (valid_rows[f'exclude_{expert}'] == class_id)).sum()
                    fp = ((valid_rows[expert] == class_id) & (valid_rows[f'exclude_{expert}'] != class_id)).sum()
                    fn = ((valid_rows[expert] != class_id) & (valid_rows[f'exclude_{expert}'] == class_id)).sum()
                    tn = ((valid_rows[expert] != class_id) & (valid_rows[f'exclude_{expert}'] != class_id)).sum()

                    # Calculate TPR and FPR
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    # Calculate Precision and Recall
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tpr  # Recall is the same as TPR

                    # Store expert point for this bootstrap iteration
                    expert_bootstrap_roc[expert]['fpr'].append(fpr)
                    expert_bootstrap_roc[expert]['tpr'].append(tpr)
                    expert_bootstrap_pr[expert]['recall'].append(recall)
                    expert_bootstrap_pr[expert]['precision'].append(precision)

                    # Store for current iteration's EUC calculation
                    iter_expert_roc[expert] = (fpr, tpr)
                    iter_expert_pr[expert] = (recall, precision)

                    # Get ground truth and model predictions for model evaluation
                    y_true_expert = valid_rows[f'exclude_{expert}'].values
                    y_true_expert = y_true_expert.astype(int)
                    y_true_bin_expert = label_binarize(y_true_expert, classes=[i for i in range(n_classes)])

                    valid_indices = valid_rows.index
                    y_pred_prob_valid = y_pred_prob_M[valid_indices]

                    try:
                        if label_name in binary_label:
                            fpr_model, tpr_model, _ = roc_curve(y_true_bin_expert[:, class_id - 1],
                                                                y_pred_prob_valid[:, class_id - 1])
                            precision_model, recall_model, _ = precision_recall_curve(
                                y_true_bin_expert[:, class_id - 1],
                                y_pred_prob_valid[:, class_id - 1])
                            pr_auc = average_precision_score(y_true_bin_expert[:, class_id - 1],
                                                             y_pred_prob_valid[:, class_id - 1])
                        else:
                            fpr_model, tpr_model, _ = roc_curve(y_true_bin_expert[:, class_id],
                                                                y_pred_prob_valid[:, class_id])
                            precision_model, recall_model, _ = precision_recall_curve(y_true_bin_expert[:, class_id],
                                                                                      y_pred_prob_valid[:, class_id])
                            pr_auc = average_precision_score(y_true_bin_expert[:, class_id],
                                                             y_pred_prob_valid[:, class_id])

                        # Calculate ROC AUC
                        roc_auc = auc(fpr_model, tpr_model)

                        # Store model curves for this expert
                        iter_model_curves['roc']['fpr'].append(fpr_model)
                        iter_model_curves['roc']['tpr'].append(tpr_model)
                        iter_model_curves['roc']['auc'].append(roc_auc)

                        iter_model_curves['pr']['recall'].append(recall_model)
                        iter_model_curves['pr']['precision'].append(precision_model)
                        iter_model_curves['pr']['auc'].append(pr_auc)

                    except Exception as e:
                        print(f"Error in bootstrap {bootstrap_iter}, expert {expert}: {e}")
                        continue

                # Calculate average model curves for this bootstrap iteration
                if iter_model_curves['roc']['fpr']:
                    # Calculate average AUC for this iteration
                    avg_auc_roc = np.mean(iter_model_curves['roc']['auc'])
                    model_bootstrap['roc']['auc'].append(avg_auc_roc)

                    # Calculate average interpolated TPR
                    iter_tprs = []
                    for fpr, tpr in zip(iter_model_curves['roc']['fpr'],
                                        iter_model_curves['roc']['tpr']):
                        if len(fpr) > 1:
                            interp_tpr = np.interp(mean_fpr, fpr, tpr)
                            interp_tpr[0] = 0.0  # Force start at 0,0
                            iter_tprs.append(interp_tpr)

                    if iter_tprs:
                        avg_tpr = np.mean(iter_tprs, axis=0)
                        model_bootstrap['roc']['fpr'].append(mean_fpr)
                        model_bootstrap['roc']['tpr'].append(avg_tpr)

                # Process PR curves
                if iter_model_curves['pr']['recall']:
                    # Calculate average AUC for this iteration
                    avg_auc_pr = np.mean(iter_model_curves['pr']['auc'])
                    model_bootstrap['pr']['auc'].append(avg_auc_pr)

                    # Calculate average interpolated precision
                    iter_precisions = []
                    for recall, precision in zip(iter_model_curves['pr']['recall'],
                                                 iter_model_curves['pr']['precision']):
                        if len(recall) > 1:
                            # Reverse arrays for interpolation
                            recall_rev = recall[::-1]
                            precision_rev = precision[::-1]
                            interp_precision = np.interp(mean_recall, recall_rev, precision_rev)
                            iter_precisions.append(interp_precision)

                    if iter_precisions:
                        avg_precision = np.mean(iter_precisions, axis=0)
                        model_bootstrap['pr']['recall'].append(mean_recall)
                        model_bootstrap['pr']['precision'].append(avg_precision)

                # Calculate EUC for this bootstrap iteration
                experts_above_curve_roc = 0
                experts_above_curve_pr = 0
                valid_experts_count = 0

                # Skip if no model curves for this iteration
                if not model_bootstrap['roc']['fpr'] or not model_bootstrap['pr']['recall']:
                    continue

                # Get the model curves for this iteration
                model_fpr = model_bootstrap['roc']['fpr'][-1]  # Last added is current iteration
                model_tpr = model_bootstrap['roc']['tpr'][-1]
                model_recall = model_bootstrap['pr']['recall'][-1]
                model_precision = model_bootstrap['pr']['precision'][-1]

                # Compare each expert point to the model curve
                for expert in expert_columns:
                    if expert in iter_expert_roc and expert in iter_expert_pr:
                        valid_experts_count += 1

                        # Get expert points
                        expert_fpr, expert_tpr = iter_expert_roc[expert]
                        expert_recall, expert_precision = iter_expert_pr[expert]

                        # Check if expert is above ROC curve
                        expected_tpr = np.interp(expert_fpr, model_fpr, model_tpr)
                        if expert_tpr > expected_tpr:
                            experts_above_curve_roc += 1

                        # Check if expert is above PR curve
                        expected_precision = np.interp(expert_recall, model_recall, model_precision)
                        if expert_precision > expected_precision:
                            experts_above_curve_pr += 1

                # Calculate EUC percentages
                if valid_experts_count > 0:
                    euc_roc = (valid_experts_count - experts_above_curve_roc) / valid_experts_count * 100
                    euc_pr = (valid_experts_count - experts_above_curve_pr) / valid_experts_count * 100

                    model_bootstrap['euc_roc'].append(euc_roc)
                    model_bootstrap['euc_pr'].append(euc_pr)

            # Calculate mean expert points and confidence intervals
            expert_mean_roc = {}
            expert_mean_pr = {}
            expert_ci_roc = {}
            expert_ci_pr = {}

            for expert in expert_columns:
                # Calculate means if data exists
                if expert_bootstrap_roc[expert]['fpr'] and expert_bootstrap_roc[expert]['tpr']:
                    mean_fpr_expert = np.mean(expert_bootstrap_roc[expert]['fpr'])
                    mean_tpr_expert = np.mean(expert_bootstrap_roc[expert]['tpr'])

                    # Calculate 95% confidence intervals
                    ci_fpr_lower = np.percentile(expert_bootstrap_roc[expert]['fpr'], 2.5)
                    ci_fpr_upper = np.percentile(expert_bootstrap_roc[expert]['fpr'], 97.5)
                    ci_tpr_lower = np.percentile(expert_bootstrap_roc[expert]['tpr'], 2.5)
                    ci_tpr_upper = np.percentile(expert_bootstrap_roc[expert]['tpr'], 97.5)

                    expert_mean_roc[expert] = (mean_fpr_expert, mean_tpr_expert)
                    expert_ci_roc[expert] = ((ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper))
                else:
                    expert_mean_roc[expert] = (0, 0)
                    expert_ci_roc[expert] = ((0, 0), (0, 0))

                # Same for PR curve
                if expert_bootstrap_pr[expert]['recall'] and expert_bootstrap_pr[expert]['precision']:
                    mean_recall_expert = np.mean(expert_bootstrap_pr[expert]['recall'])
                    mean_precision_expert = np.mean(expert_bootstrap_pr[expert]['precision'])

                    # Calculate 95% confidence intervals
                    ci_recall_lower = np.percentile(expert_bootstrap_pr[expert]['recall'], 2.5)
                    ci_recall_upper = np.percentile(expert_bootstrap_pr[expert]['recall'], 97.5)
                    ci_precision_lower = np.percentile(expert_bootstrap_pr[expert]['precision'], 2.5)
                    ci_precision_upper = np.percentile(expert_bootstrap_pr[expert]['precision'], 97.5)

                    expert_mean_pr[expert] = (mean_recall_expert, mean_precision_expert)
                    expert_ci_pr[expert] = (
                    (ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper))
                else:
                    expert_mean_pr[expert] = (0, 0)
                    expert_ci_pr[expert] = ((0, 0), (0, 0))

            # Calculate model statistics
            mean_auc_roc = np.mean(model_bootstrap['roc']['auc']) if model_bootstrap['roc']['auc'] else 0
            ci_lower_auc_roc = np.percentile(model_bootstrap['roc']['auc'], 2.5) if model_bootstrap['roc']['auc'] else 0
            ci_upper_auc_roc = np.percentile(model_bootstrap['roc']['auc'], 97.5) if model_bootstrap['roc'][
                'auc'] else 0

            mean_auc_pr = np.mean(model_bootstrap['pr']['auc']) if model_bootstrap['pr']['auc'] else 0
            ci_lower_auc_pr = np.percentile(model_bootstrap['pr']['auc'], 2.5) if model_bootstrap['pr']['auc'] else 0
            ci_upper_auc_pr = np.percentile(model_bootstrap['pr']['auc'], 97.5) if model_bootstrap['pr']['auc'] else 0

            # Calculate EUC statistics
            mean_euc_roc = np.mean(model_bootstrap['euc_roc']) if model_bootstrap['euc_roc'] else 0
            ci_lower_euc_roc = np.percentile(model_bootstrap['euc_roc'], 2.5) if model_bootstrap['euc_roc'] else 0
            ci_upper_euc_roc = np.percentile(model_bootstrap['euc_roc'], 97.5) if model_bootstrap['euc_roc'] else 0

            mean_euc_pr = np.mean(model_bootstrap['euc_pr']) if model_bootstrap['euc_pr'] else 0
            ci_lower_euc_pr = np.percentile(model_bootstrap['euc_pr'], 2.5) if model_bootstrap['euc_pr'] else 0
            ci_upper_euc_pr = np.percentile(model_bootstrap['euc_pr'], 97.5) if model_bootstrap['euc_pr'] else 0

            # Store in results dictionary
            model_name = model_names[0]  # We only have one model

            results[label_name][model_name]['roc_auc'] = {
                'mean': mean_auc_roc,
                'min': ci_lower_auc_roc,
                'max': ci_upper_auc_roc
            }
            results[label_name][model_name]['pr_auc'] = {
                'mean': mean_auc_pr,
                'min': ci_lower_auc_pr,
                'max': ci_upper_auc_pr
            }
            results[label_name][model_name]['euc_roc'] = {
                'mean': mean_euc_roc,
                'min': ci_lower_euc_roc,
                'max': ci_upper_euc_roc
            }
            results[label_name][model_name]['euc_pr'] = {
                'mean': mean_euc_pr,
                'min': ci_lower_euc_pr,
                'max': ci_upper_euc_pr
            }

            # Add to formatted output
            formatted_output += f"\n{label_name} - {model_name}:\n"
            formatted_output += f"ROC AUC: {mean_auc_roc:.3f}({ci_lower_auc_roc:.3f}, {ci_upper_auc_roc:.3f})\n"
            formatted_output += f"PR AUC: {mean_auc_pr:.3f}({ci_lower_auc_pr:.3f}, {ci_upper_auc_pr:.3f})\n"
            formatted_output += f"ROC EUC: {mean_euc_roc:.1f}%({ci_lower_euc_roc:.1f}%, {ci_upper_euc_roc:.1f}%)\n"
            formatted_output += f"PR EUC: {mean_euc_pr:.1f}%({ci_lower_euc_pr:.1f}%, {ci_upper_euc_pr:.1f}%)\n"

            # Plot ROC curve with bootstrap confidence intervals
            ax_roc = plt.subplot(2, 1, 1)

            if model_bootstrap['roc']['tpr']:
                bootstrap_tprs_array = np.array(model_bootstrap['roc']['tpr'])
                mean_tpr = np.mean(bootstrap_tprs_array, axis=0)
                tpr_lower = np.percentile(bootstrap_tprs_array, 2.5, axis=0)
                tpr_upper = np.percentile(bootstrap_tprs_array, 97.5, axis=0)

                # Plot mean ROC curve
                line, = plt.plot(mean_fpr, mean_tpr, color=model_colors[0], lw=2)

                # Create legend entry with confidence intervals
                legend_handles = [line]
                legend_labels = [
                    f"AUC={mean_auc_roc:.3f}\nEUC={mean_euc_roc:.1f}%"]

                # Draw confidence bands
                plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=model_colors[0], alpha=0.3)

            # Draw expert points with confidence interval crosses - ROC
            for expert in expert_columns:
                if expert in expert_mean_roc:
                    # Draw mean expert point
                    mean_fpr_expert, mean_tpr_expert = expert_mean_roc[expert]
                    plt.scatter(mean_fpr_expert, mean_tpr_expert, marker='o', color=expert_color, alpha=0.6, s=20)

                    # Draw confidence interval crosses
                    (ci_fpr_lower, ci_fpr_upper), (ci_tpr_lower, ci_tpr_upper) = expert_ci_roc[expert]

                    # Draw horizontal line (TPR range)
                    plt.plot([mean_fpr_expert, mean_fpr_expert], [ci_tpr_lower, ci_tpr_upper],
                             color=expert_color, alpha=0.6, linewidth=1)

                    # Draw vertical line (FPR range)
                    plt.plot([ci_fpr_lower, ci_fpr_upper], [mean_tpr_expert, mean_tpr_expert],
                             color=expert_color, alpha=0.6, linewidth=1)

            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)

            # Add legend
            plt.legend(legend_handles, legend_labels, loc='lower right',
                       fontsize=18, handlelength=0, handletextpad=0,
                       frameon=False, title=f'{label_name}', title_fontsize=20)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            # Plot PR curve with bootstrap confidence intervals
            ax_pr = plt.subplot(2, 1, 2)

            if model_bootstrap['pr']['precision']:
                bootstrap_precisions_array = np.array(model_bootstrap['pr']['precision'])
                mean_precision = np.mean(bootstrap_precisions_array, axis=0)
                precision_lower = np.percentile(bootstrap_precisions_array, 2.5, axis=0)
                precision_upper = np.percentile(bootstrap_precisions_array, 97.5, axis=0)

                # Plot mean PR curve
                line, = plt.plot(mean_recall, mean_precision, color=model_colors[0], lw=2)

                # Create legend entry with confidence intervals
                legend_handles = [line]
                legend_labels = [
                    f"AUC={mean_auc_pr:.3f}\nEUC={mean_euc_pr:.1f}%"]

                # Draw confidence bands
                plt.fill_between(mean_recall, precision_lower, precision_upper, color=model_colors[0], alpha=0.3)

            # Draw expert points with confidence interval crosses - PR
            for expert in expert_columns:
                if expert in expert_mean_pr:
                    # Draw mean expert point
                    mean_recall_expert, mean_precision_expert = expert_mean_pr[expert]
                    plt.scatter(mean_recall_expert, mean_precision_expert, marker='o', color=expert_color, alpha=0.6,
                                s=20)

                    # Draw confidence interval crosses
                    (ci_recall_lower, ci_recall_upper), (ci_precision_lower, ci_precision_upper) = expert_ci_pr[expert]

                    # Draw horizontal line (precision range)
                    plt.plot([mean_recall_expert, mean_recall_expert], [ci_precision_lower, ci_precision_upper],
                             color=expert_color, alpha=0.6, linewidth=1)

                    # Draw vertical line (recall range)
                    plt.plot([ci_recall_lower, ci_recall_upper], [mean_precision_expert, mean_precision_expert],
                             color=expert_color, alpha=0.6, linewidth=1)

            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)

            # Add legend
            plt.legend(legend_handles, legend_labels, loc='lower left',
                       fontsize=18, handlelength=0, handletextpad=0,
                       frameon=False, title=f'{label_name}', title_fontsize=20)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.show()

    # Print the formatted output
    print(formatted_output)

    return results, formatted_output


def plot_auc_euc_with_bootstrap_MoEBS(fig_path, results_dir='/Users/chenxisun/Downloads/MorgothV3Results/BS',
                                      n_bootstrap=1000):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    from sklearn.preprocessing import label_binarize

    # String to collect formatted output
    formatted_output = ""

    labels = [
        'BS',
    ]
    binary_label = [
        'BS'
    ]
    label_maps = [
        {0: 'Others', 1: 'BS'},
    ]

    # Set colors for the two models
    model_colors = ['steelblue', 'orange']
    model_names = ['Morgoth', 'Baseline']
    expert_color = 'grey'

    # Dictionary to store results
    results = {}

    for label, label_map in zip(labels, label_maps):
        results[label] = {}
        class_names = list(label_map.values())
        n_classes = len(class_names)

        result_df = pd.read_csv(os.path.join(results_dir, f'{label}_MoE_models_and_experts_results.csv'))

        expert_columns = [c for c in result_df.columns if c.startswith('expert')]

        plt.figure(figsize=(4 * (n_classes - 1), 8) if label in binary_label else (4 * n_classes, 8))

        # Get prediction probabilities for both models
        if label in binary_label:
            y_pred_prob_M = result_df[['M_pred']].values
            y_pred_prob_B = result_df[['B_pred']].values
        else:
            y_pred_prob_M = result_df[[f'class_{i}_prob' for i in range(n_classes)]].values
            y_pred_prob_B = result_df[[f'B_class_{i}_prob' for i in range(n_classes)]].values

        for index, class_id in enumerate(label_map.keys()):
            if class_id == 0 and label in binary_label:
                continue

            class_name = label_map[class_id]
            results[label][class_name] = {}

            # Initialize structures for storing bootstrap results
            bootstrap_fpr_models = [[] for _ in range(2)]
            bootstrap_tpr_models = [[] for _ in range(2)]
            bootstrap_precision_models = [[] for _ in range(2)]
            bootstrap_recall_models = [[] for _ in range(2)]
            bootstrap_auc_roc_models = [[] for _ in range(2)]
            bootstrap_auc_pr_models = [[] for _ in range(2)]

            # Store interpolated bootstrap curves for percentile computation
            bootstrap_interp_tprs = [[] for _ in range(2)]
            bootstrap_interp_precisions = [[] for _ in range(2)]
            mean_fpr = np.linspace(0, 1, 100)
            mean_recall = np.linspace(0, 1, 100)

            # Store expert bootstrap results
            bootstrap_expert_fpr = []
            bootstrap_expert_tpr = []
            bootstrap_expert_recall = []
            bootstrap_expert_precision = []

            # For computing mean expert points
            expert_mean_points_roc = {expert: {'fpr': [], 'tpr': []} for expert in expert_columns}
            expert_mean_points_pr = {expert: {'recall': [], 'precision': []} for expert in expert_columns}

            # Store EUC computation results
            bootstrap_euc_roc = [[] for _ in range(2)]
            bootstrap_euc_pr = [[] for _ in range(2)]

            # Get original curves for reference
            original_fpr_models = [[] for _ in range(2)]
            original_tpr_models = [[] for _ in range(2)]
            original_precision_models = [[] for _ in range(2)]
            original_recall_models = [[] for _ in range(2)]
            original_auc_roc_models = [[] for _ in range(2)]
            original_auc_pr_models = [[] for _ in range(2)]

            # Compute original curves for each model
            for model_idx, y_pred_prob in enumerate([y_pred_prob_M, y_pred_prob_B]):
                y_true = result_df['majority'].values
                y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])

                if label in binary_label:
                    fpr_model, tpr_model, _ = roc_curve(y_true_bin[:, class_id - 1], y_pred_prob[:, class_id - 1])
                    precision_model, recall_model, _ = precision_recall_curve(y_true_bin[:, class_id - 1],
                                                                              y_pred_prob[:, class_id - 1])
                    roc_auc = auc(fpr_model, tpr_model)
                    pr_auc = average_precision_score(y_true_bin[:, class_id - 1], y_pred_prob[:, class_id - 1])
                else:
                    fpr_model, tpr_model, _ = roc_curve(y_true_bin[:, class_id], y_pred_prob[:, class_id])
                    precision_model, recall_model, _ = precision_recall_curve(y_true_bin[:, class_id],
                                                                              y_pred_prob[:, class_id])
                    roc_auc = auc(fpr_model, tpr_model)
                    pr_auc = average_precision_score(y_true_bin[:, class_id], y_pred_prob[:, class_id])

                original_fpr_models[model_idx] = fpr_model
                original_tpr_models[model_idx] = tpr_model
                original_precision_models[model_idx] = precision_model
                original_recall_models[model_idx] = recall_model
                original_auc_roc_models[model_idx] = roc_auc
                original_auc_pr_models[model_idx] = pr_auc

            # Run bootstrap
            n_samples = len(result_df)

            for bootstrap_idx in range(n_bootstrap):
                # Generate bootstrap sample (sampling with replacement)
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_df = result_df.iloc[bootstrap_indices].copy()

                # Get model predictions for the bootstrap sample
                if label in binary_label:
                    y_pred_prob_M_bootstrap = bootstrap_df[['M_pred']].values
                    y_pred_prob_B_bootstrap = bootstrap_df[['B_pred']].values
                else:
                    y_pred_prob_M_bootstrap = bootstrap_df[[f'class_{i}_prob' for i in range(n_classes)]].values
                    y_pred_prob_B_bootstrap = bootstrap_df[[f'B_class_{i}_prob' for i in range(n_classes)]].values

                # Get ground truth for the bootstrap sample
                y_true_bootstrap = bootstrap_df['majority'].values
                y_true_bin_bootstrap = label_binarize(y_true_bootstrap, classes=[i for i in range(n_classes)])

                # Bootstrap for models
                for model_idx, y_pred_prob in enumerate([y_pred_prob_M_bootstrap, y_pred_prob_B_bootstrap]):
                    try:
                        # Compute ROC curve
                        if label in binary_label:
                            fpr_model, tpr_model, _ = roc_curve(y_true_bin_bootstrap[:, class_id - 1],
                                                                y_pred_prob[:, class_id - 1])
                            precision_model, recall_model, _ = precision_recall_curve(
                                y_true_bin_bootstrap[:, class_id - 1], y_pred_prob[:, class_id - 1])
                            roc_auc = auc(fpr_model, tpr_model)
                            pr_auc = average_precision_score(y_true_bin_bootstrap[:, class_id - 1],
                                                             y_pred_prob[:, class_id - 1])
                        else:
                            fpr_model, tpr_model, _ = roc_curve(y_true_bin_bootstrap[:, class_id],
                                                                y_pred_prob[:, class_id])
                            precision_model, recall_model, _ = precision_recall_curve(y_true_bin_bootstrap[:, class_id],
                                                                                      y_pred_prob[:, class_id])
                            roc_auc = auc(fpr_model, tpr_model)
                            pr_auc = average_precision_score(y_true_bin_bootstrap[:, class_id],
                                                             y_pred_prob[:, class_id])

                        bootstrap_fpr_models[model_idx].append(fpr_model)
                        bootstrap_tpr_models[model_idx].append(tpr_model)
                        bootstrap_precision_models[model_idx].append(precision_model)
                        bootstrap_recall_models[model_idx].append(recall_model)
                        bootstrap_auc_roc_models[model_idx].append(roc_auc)
                        bootstrap_auc_pr_models[model_idx].append(pr_auc)

                        # Interpolate curves for this bootstrap sample
                        if len(fpr_model) > 1:
                            # Find unique FPR values and corresponding TPR values for ROC
                            unique_indices = np.unique(fpr_model, return_index=True)[1]
                            unique_fpr = fpr_model[np.sort(unique_indices)]
                            unique_tpr = tpr_model[np.sort(unique_indices)]

                            if len(unique_fpr) > 1:
                                interp_tpr = np.interp(mean_fpr, unique_fpr, unique_tpr)
                                interp_tpr[0] = 0.0  # Ensure curve starts at (0,0)
                                bootstrap_interp_tprs[model_idx].append(interp_tpr)

                        if len(recall_model) > 1:
                            # Reverse PR curve arrays
                            recall_rev = recall_model[::-1]
                            precision_rev = precision_model[::-1]

                            # Find unique recall values and corresponding precision values
                            unique_indices = np.unique(recall_rev, return_index=True)[1]
                            unique_recall = recall_rev[np.sort(unique_indices)]
                            unique_precision = precision_rev[np.sort(unique_indices)]

                            if len(unique_recall) > 1:
                                interp_precision = np.interp(mean_recall, unique_recall, unique_precision)
                                bootstrap_interp_precisions[model_idx].append(interp_precision)

                    except Exception as e:
                        print(f"Error in bootstrap iteration {bootstrap_idx} for model {model_idx}: {e}")

                # Expert bootstrap - resample rows independently for each expert
                iter_expert_fpr = []
                iter_expert_tpr = []
                iter_expert_recall = []
                iter_expert_precision = []

                for expert_idx, expert in enumerate(expert_columns):
                    try:
                        # Create bootstrap sample for this expert
                        valid_rows = bootstrap_df[[expert, f'exclude_{expert}']].dropna()
                        valid_rows = valid_rows[valid_rows[f'exclude_{expert}'] != -1]

                        # Bootstrap rows for this expert
                        if len(valid_rows) > 0:
                            bootstrap_row_indices = np.random.choice(len(valid_rows), len(valid_rows), replace=True)
                            bootstrap_rows = valid_rows.iloc[bootstrap_row_indices]

                            tp = ((bootstrap_rows[expert] == class_id) & (
                                        bootstrap_rows[f'exclude_{expert}'] == class_id)).sum()
                            fp = ((bootstrap_rows[expert] == class_id) & (
                                        bootstrap_rows[f'exclude_{expert}'] != class_id)).sum()
                            fn = ((bootstrap_rows[expert] != class_id) & (
                                        bootstrap_rows[f'exclude_{expert}'] == class_id)).sum()
                            tn = ((bootstrap_rows[expert] != class_id) & (
                                        bootstrap_rows[f'exclude_{expert}'] != class_id)).sum()

                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tpr

                            iter_expert_fpr.append(fpr)
                            iter_expert_tpr.append(tpr)
                            iter_expert_recall.append(recall)
                            iter_expert_precision.append(precision)

                            # Store individual expert results
                            expert_mean_points_roc[expert]['fpr'].append(fpr)
                            expert_mean_points_roc[expert]['tpr'].append(tpr)
                            expert_mean_points_pr[expert]['recall'].append(recall)
                            expert_mean_points_pr[expert]['precision'].append(precision)
                        else:
                            # Add placeholder values
                            iter_expert_fpr.append(np.nan)
                            iter_expert_tpr.append(np.nan)
                            iter_expert_recall.append(np.nan)
                            iter_expert_precision.append(np.nan)
                    except Exception as e:
                        print(f"Error in expert bootstrap calculation for {expert}: {e}")
                        # Add placeholder values to maintain alignment with expert columns
                        iter_expert_fpr.append(np.nan)
                        iter_expert_tpr.append(np.nan)
                        iter_expert_recall.append(np.nan)
                        iter_expert_precision.append(np.nan)

                # Store expert statistics for this iteration
                if len(iter_expert_fpr) > 0:
                    bootstrap_expert_fpr.append(iter_expert_fpr)
                    bootstrap_expert_tpr.append(iter_expert_tpr)
                    bootstrap_expert_recall.append(iter_expert_recall)
                    bootstrap_expert_precision.append(iter_expert_precision)

                # Compute EUC for this bootstrap iteration
                # EUC: Experts Under Curve - percentage of experts below (or on) the curve

                # For each model in this bootstrap iteration
                for model_idx in range(2):
                    experts_above_curve_roc = 0
                    experts_above_curve_pr = 0
                    valid_experts_count = 0

                    # For each expert in this bootstrap iteration
                    for expert_idx, expert in enumerate(expert_columns):
                        if expert_idx < len(iter_expert_fpr) and not np.isnan(
                                iter_expert_fpr[expert_idx]) and not np.isnan(iter_expert_tpr[expert_idx]):
                            valid_experts_count += 1

                            # Get expert point
                            expert_fpr = iter_expert_fpr[expert_idx]
                            expert_tpr = iter_expert_tpr[expert_idx]

                            # Get corresponding model curve
                            if model_idx < len(bootstrap_fpr_models) and bootstrap_idx < len(
                                    bootstrap_fpr_models[model_idx]):
                                model_fpr = bootstrap_fpr_models[model_idx][bootstrap_idx]
                                model_tpr = bootstrap_tpr_models[model_idx][bootstrap_idx]

                                # Interpolate expected TPR at expert's FPR
                                if len(model_fpr) > 1:
                                    try:
                                        expected_tpr = np.interp(expert_fpr, model_fpr, model_tpr)

                                        # Check whether the expert is above the curve
                                        if expert_tpr > expected_tpr:
                                            experts_above_curve_roc += 1
                                    except Exception as e:
                                        print(f"Error in ROC EUC interpolation: {e}")

                        # Same for PR curve
                        if expert_idx < len(iter_expert_recall) and not np.isnan(
                                iter_expert_recall[expert_idx]) and not np.isnan(iter_expert_precision[expert_idx]):
                            # Get expert point
                            expert_recall = iter_expert_recall[expert_idx]
                            expert_precision = iter_expert_precision[expert_idx]

                            # Get corresponding model curve
                            if model_idx < len(bootstrap_recall_models) and bootstrap_idx < len(
                                    bootstrap_recall_models[model_idx]):
                                model_recall = bootstrap_recall_models[model_idx][bootstrap_idx]
                                model_precision = bootstrap_precision_models[model_idx][bootstrap_idx]

                                # Interpolate expected precision at expert's recall
                                if len(model_recall) > 1:
                                    try:
                                        # Reverse arrays for PR curve interpolation
                                        recall_rev = model_recall[::-1]
                                        precision_rev = model_precision[::-1]

                                        expected_precision = np.interp(expert_recall, recall_rev, precision_rev)

                                        # Check whether the expert is above the curve
                                        if expert_precision > expected_precision:
                                            experts_above_curve_pr += 1
                                    except Exception as e:
                                        print(f"Error in PR EUC interpolation: {e}")

                    # Compute EUC percentage for this bootstrap iteration
                    if valid_experts_count > 0:
                        euc_roc = (valid_experts_count - experts_above_curve_roc) / valid_experts_count * 100
                        euc_pr = (valid_experts_count - experts_above_curve_pr) / valid_experts_count * 100

                        bootstrap_euc_roc[model_idx].append(euc_roc)
                        bootstrap_euc_pr[model_idx].append(euc_pr)

            # Compute mean expert points from bootstrap
            expert_mean_roc = {}
            expert_mean_pr = {}

            for expert in expert_columns:
                fpr_values = expert_mean_points_roc[expert]['fpr']
                tpr_values = expert_mean_points_roc[expert]['tpr']
                recall_values = expert_mean_points_pr[expert]['recall']
                precision_values = expert_mean_points_pr[expert]['precision']

                if len(fpr_values) > 0 and len(tpr_values) > 0:
                    mean_fpr_expert = np.mean(fpr_values)
                    mean_tpr_expert = np.mean(tpr_values)
                    expert_mean_roc[expert] = (mean_fpr_expert, mean_tpr_expert)
                else:
                    expert_mean_roc[expert] = (0, 0)

                if len(recall_values) > 0 and len(precision_values) > 0:
                    mean_recall_expert = np.mean(recall_values)
                    mean_precision_expert = np.mean(precision_values)
                    expert_mean_pr[expert] = (mean_recall_expert, mean_precision_expert)
                else:
                    expert_mean_pr[expert] = (0, 0)

            # Compute bootstrap mean EUC and confidence intervals
            mean_euc_roc = [np.mean(bootstrap_euc_roc[i]) if len(bootstrap_euc_roc[i]) > 0 else 0 for i in range(2)]
            ci_lower_euc_roc = [np.percentile(bootstrap_euc_roc[i], 2.5) if len(bootstrap_euc_roc[i]) > 0 else 0 for i
                                in range(2)]
            ci_upper_euc_roc = [np.percentile(bootstrap_euc_roc[i], 97.5) if len(bootstrap_euc_roc[i]) > 0 else 0 for i
                                in range(2)]

            mean_euc_pr = [np.mean(bootstrap_euc_pr[i]) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in range(2)]
            ci_lower_euc_pr = [np.percentile(bootstrap_euc_pr[i], 2.5) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in
                               range(2)]
            ci_upper_euc_pr = [np.percentile(bootstrap_euc_pr[i], 97.5) if len(bootstrap_euc_pr[i]) > 0 else 0 for i in
                               range(2)]

            # Plot ROC curve
            plt.subplot(2, 1, 1)

            # Compute and plot expert confidence intervals as crosses
            if len(bootstrap_expert_fpr) > 0 and len(bootstrap_expert_tpr) > 0:
                # Reorganize bootstrap data by expert
                expert_bootstrap_data_roc = {}

                # For each bootstrap iteration
                for iter_idx in range(len(bootstrap_expert_fpr)):
                    # For each expert in this iteration
                    for expert_idx, expert_name in enumerate(expert_columns):
                        if expert_idx >= len(bootstrap_expert_fpr[iter_idx]):
                            continue

                        if expert_name not in expert_bootstrap_data_roc:
                            expert_bootstrap_data_roc[expert_name] = {"fpr": [], "tpr": []}

                        if not np.isnan(bootstrap_expert_fpr[iter_idx][expert_idx]) and not np.isnan(
                                bootstrap_expert_tpr[iter_idx][expert_idx]):
                            expert_bootstrap_data_roc[expert_name]["fpr"].append(
                                bootstrap_expert_fpr[iter_idx][expert_idx])
                            expert_bootstrap_data_roc[expert_name]["tpr"].append(
                                bootstrap_expert_tpr[iter_idx][expert_idx])

                # Compute CIs and plot crosses for each expert
                for expert_name, data in expert_bootstrap_data_roc.items():
                    if len(data["fpr"]) > 5 and len(data["tpr"]) > 5:  # Ensure enough bootstrap samples
                        # Compute median point
                        median_fpr = np.median(data["fpr"])
                        median_tpr = np.median(data["tpr"])

                        # Compute FPR and TPR ranges
                        fpr_min = np.percentile(data["fpr"], 2.5)
                        fpr_max = np.percentile(data["fpr"], 97.5)
                        tpr_min = np.percentile(data["tpr"], 2.5)
                        tpr_max = np.percentile(data["tpr"], 97.5)

                        # Draw vertical line (TPR range)
                        plt.plot([median_fpr, median_fpr], [tpr_min, tpr_max],
                                 color=expert_color, alpha=0.5, linewidth=1)

                        # Draw horizontal line (FPR range)
                        plt.plot([fpr_min, fpr_max], [median_tpr, median_tpr],
                                 color=expert_color, alpha=0.5, linewidth=1)

            # Store legend handles and labels
            legend_handles = []
            legend_labels = []

            # Plot model curves with bootstrap confidence intervals
            for model_idx in range(2):
                # Compute mean AUC and confidence intervals
                if len(bootstrap_auc_roc_models[model_idx]) > 0:
                    mean_auc_roc = np.mean(bootstrap_auc_roc_models[model_idx])
                    ci_lower_auc_roc = np.percentile(bootstrap_auc_roc_models[model_idx], 2.5)
                    ci_upper_auc_roc = np.percentile(bootstrap_auc_roc_models[model_idx], 97.5)
                else:
                    mean_auc_roc = original_auc_roc_models[model_idx]
                    ci_lower_auc_roc = mean_auc_roc
                    ci_upper_auc_roc = mean_auc_roc

                # Store results for this model
                results[label][class_name][model_names[model_idx]] = {
                    'roc_auc': {
                        'mean': mean_auc_roc,
                        'min': ci_lower_auc_roc,
                        'max': ci_upper_auc_roc
                    },
                    'euc_roc': {
                        'mean': mean_euc_roc[model_idx],
                        'min': ci_lower_euc_roc[model_idx],
                        'max': ci_upper_euc_roc[model_idx]
                    }
                }

                # Add to formatted output
                formatted_output += f"\n{label} - {class_name} - {model_names[model_idx]}:\n"
                formatted_output += f"ROC AUC: {mean_auc_roc:.3f}({ci_lower_auc_roc:.3f}, {ci_upper_auc_roc:.3f})\n"
                formatted_output += f"ROC EUC: {mean_euc_roc[model_idx]:.1f}%({ci_lower_euc_roc[model_idx]:.1f}%, {ci_upper_euc_roc[model_idx]:.1f}%)\n"

                # Compute true 95% confidence band and mean curve from bootstrap samples
                if len(bootstrap_interp_tprs[model_idx]) > 0:
                    bootstrap_interp_tprs_array = np.array(bootstrap_interp_tprs[model_idx])
                    # Compute mean curve
                    mean_tpr = np.mean(bootstrap_interp_tprs_array, axis=0)
                    # Compute true 95% confidence bounds directly from percentiles
                    tpr_lower = np.percentile(bootstrap_interp_tprs_array, 2.5, axis=0)
                    tpr_upper = np.percentile(bootstrap_interp_tprs_array, 97.5, axis=0)

                    # Plot mean bootstrap curve (instead of original curve)
                    line, = plt.plot(mean_fpr, mean_tpr, color=model_colors[model_idx], lw=2)
                else:
                    # If no valid bootstrap samples, use original curve
                    mean_tpr = np.interp(mean_fpr, original_fpr_models[model_idx], original_tpr_models[model_idx])
                    tpr_upper = mean_tpr
                    tpr_lower = mean_tpr

                    # Plot original curve as fallback
                    line, = plt.plot(original_fpr_models[model_idx], original_tpr_models[model_idx],
                                     color=model_colors[model_idx], lw=2)

                # Create simplified legend label with EUC confidence interval
                legend_handles.append(line)
                legend_labels.append(
                    f"  {model_names[model_idx]}\nAUC={mean_auc_roc:.3f}\nEUC={mean_euc_roc[model_idx]:.1f}%")

                # Plot confidence interval
                plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=model_colors[model_idx], alpha=0.3)

            # Plot mean expert points (instead of original points)
            for expert in expert_columns:
                mean_fpr_expert, mean_tpr_expert = expert_mean_roc[expert]
                plt.scatter(mean_fpr_expert, mean_tpr_expert, marker='o', color=expert_color, alpha=0.6, s=20)

            # Hide axis labels
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)

            plt.legend(legend_handles, legend_labels, loc='lower right', handlelength=0, handletextpad=0,
                       fontsize=16, frameon=False, title=f'Burst Suppression', title_fontsize=18)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

            # Plot PR curve
            plt.subplot(2, 1, 2)

            # Compute and plot PR curve expert confidence intervals as crosses
            if len(bootstrap_expert_recall) > 0 and len(bootstrap_expert_precision) > 0:
                # Reorganize bootstrap data by expert
                expert_bootstrap_data_pr = {}

                # For each bootstrap iteration
                for iter_idx in range(len(bootstrap_expert_recall)):
                    # For each expert in this iteration
                    for expert_idx, expert_name in enumerate(expert_columns):
                        if expert_idx >= len(bootstrap_expert_recall[iter_idx]):
                            continue

                        if expert_name not in expert_bootstrap_data_pr:
                            expert_bootstrap_data_pr[expert_name] = {"recall": [], "precision": []}

                        if not np.isnan(bootstrap_expert_recall[iter_idx][expert_idx]) and not np.isnan(
                                bootstrap_expert_precision[iter_idx][expert_idx]):
                            expert_bootstrap_data_pr[expert_name]["recall"].append(
                                bootstrap_expert_recall[iter_idx][expert_idx])
                            expert_bootstrap_data_pr[expert_name]["precision"].append(
                                bootstrap_expert_precision[iter_idx][expert_idx])

                # Compute CIs and plot crosses for each expert
                for expert_name, data in expert_bootstrap_data_pr.items():
                    if len(data["recall"]) > 5 and len(data["precision"]) > 5:  # Ensure enough bootstrap samples
                        # Compute median point
                        median_recall = np.median(data["recall"])
                        median_precision = np.median(data["precision"])

                        # Compute recall and precision ranges
                        recall_min = np.percentile(data["recall"], 2.5)
                        recall_max = np.percentile(data["recall"], 97.5)
                        precision_min = np.percentile(data["precision"], 2.5)
                        precision_max = np.percentile(data["precision"], 97.5)

                        # Draw vertical line (precision range)
                        plt.plot([median_recall, median_recall], [precision_min, precision_max],
                                 color=expert_color, alpha=0.5, linewidth=1)

                        # Draw horizontal line (recall range)
                        plt.plot([recall_min, recall_max], [median_precision, median_precision],
                                 color=expert_color, alpha=0.5, linewidth=1)

            # Reset legend handles and labels for PR curve
            legend_handles = []
            legend_labels = []

            # Plot model curves for PR curve with bootstrap confidence intervals
            for model_idx in range(2):
                # Compute mean PR AUC and confidence intervals
                if len(bootstrap_auc_pr_models[model_idx]) > 0:
                    mean_auc_pr = np.mean(bootstrap_auc_pr_models[model_idx])
                    ci_lower_auc_pr = np.percentile(bootstrap_auc_pr_models[model_idx], 2.5)
                    ci_upper_auc_pr = np.percentile(bootstrap_auc_pr_models[model_idx], 97.5)
                else:
                    mean_auc_pr = original_auc_pr_models[model_idx]
                    ci_lower_auc_pr = mean_auc_pr
                    ci_upper_auc_pr = mean_auc_pr

                # Store PR results for this model
                results[label][class_name][model_names[model_idx]].update({
                    'pr_auc': {
                        'mean': mean_auc_pr,
                        'min': ci_lower_auc_pr,
                        'max': ci_upper_auc_pr
                    },
                    'euc_pr': {
                        'mean': mean_euc_pr[model_idx],
                        'min': ci_lower_euc_pr[model_idx],
                        'max': ci_upper_euc_pr[model_idx]
                    }
                })

                # Add to formatted output
                formatted_output += f"PR AUC: {mean_auc_pr:.3f}({ci_lower_auc_pr:.3f}, {ci_upper_auc_pr:.3f})\n"
                formatted_output += f"PR EUC: {mean_euc_pr[model_idx]:.1f}%({ci_lower_euc_pr[model_idx]:.1f}%, {ci_upper_euc_pr[model_idx]:.1f}%)\n"

                # Compute true 95% confidence band and mean curve from bootstrap samples
                if len(bootstrap_interp_precisions[model_idx]) > 0:
                    bootstrap_interp_precisions_array = np.array(bootstrap_interp_precisions[model_idx])
                    # Compute mean curve
                    mean_precision = np.mean(bootstrap_interp_precisions_array, axis=0)
                    # Compute true 95% confidence bounds directly from percentiles
                    precision_lower = np.percentile(bootstrap_interp_precisions_array, 2.5, axis=0)
                    precision_upper = np.percentile(bootstrap_interp_precisions_array, 97.5, axis=0)

                    # Plot mean bootstrap curve (instead of original curve)
                    line, = plt.plot(mean_recall, mean_precision, color=model_colors[model_idx], lw=2)
                else:
                    # If no valid bootstrap samples, use original curve
                    mean_precision = np.interp(mean_recall, np.flip(original_recall_models[model_idx]),
                                               np.flip(original_precision_models[model_idx]))
                    precision_upper = mean_precision
                    precision_lower = mean_precision

                    # Plot original curve as fallback
                    line, = plt.plot(original_recall_models[model_idx], original_precision_models[model_idx],
                                     color=model_colors[model_idx], lw=2)

                # Create simplified legend label with EUC confidence interval
                legend_handles.append(line)
                legend_labels.append(
                    f"  {model_names[model_idx]}\nAUC={mean_auc_pr:.3f}\nEUC={mean_euc_pr[model_idx]:.1f}%")

                # Plot confidence interval
                plt.fill_between(mean_recall, precision_lower, precision_upper, color=model_colors[model_idx],
                                 alpha=0.2)

            # Plot mean expert points (instead of original points)
            for expert in expert_columns:
                mean_recall_expert, mean_precision_expert = expert_mean_pr[expert]
                plt.scatter(mean_recall_expert, mean_precision_expert, marker='o', color=expert_color, alpha=0.6, s=20)

            # Set axis labels
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)

            plt.legend(legend_handles, legend_labels, loc='lower left', handlelength=0, handletextpad=0,
                       fontsize=16, frameon=False, title=f'Burst Suppression', title_fontsize=18)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.show()

    # Print metric table for all models
    print("\n\n===== Complete performance metrics for all models =====\n")
    print(f"{'Label':<10} {'Class':<10} {'Model':<15} {'AUC-ROC':<25} {'AUC-PR':<25} {'EUC-ROC':<25} {'EUC-PR':<25}")
    print("=" * 120)

    # Iterate over each label and class
    for label_name, label_results in results.items():
        for class_name, class_results in label_results.items():
            # Iterate over each model
            for model_name, model_metrics in class_results.items():
                # Get and format AUC-ROC
                if 'roc_auc' in model_metrics:
                    auc_roc = model_metrics['roc_auc']
                    auc_roc_str = f"{auc_roc['mean']:.3f} ({auc_roc['min']:.3f}, {auc_roc['max']:.3f})"
                else:
                    auc_roc_str = "N/A"

                # Get and format AUC-PR
                if 'pr_auc' in model_metrics:
                    auc_pr = model_metrics['pr_auc']
                    auc_pr_str = f"{auc_pr['mean']:.3f} ({auc_pr['min']:.3f}, {auc_pr['max']:.3f})"
                else:
                    auc_pr_str = "N/A"

                # Get and format EUC-ROC
                if 'euc_roc' in model_metrics:
                    if isinstance(model_metrics['euc_roc'], dict):
                        euc_roc = model_metrics['euc_roc']
                        euc_roc_str = f"{euc_roc['mean']:.1f}% ({euc_roc['min']:.1f}%, {euc_roc['max']:.1f}%)"
                    else:
                        # Handle old format where EUC is a plain number
                        euc_roc_str = f"{model_metrics['euc_roc']:.1f}%"
                else:
                    euc_roc_str = "N/A"

                # Get and format EUC-PR
                if 'euc_pr' in model_metrics:
                    if isinstance(model_metrics['euc_pr'], dict):
                        euc_pr = model_metrics['euc_pr']
                        euc_pr_str = f"{euc_pr['mean']:.1f}% ({euc_pr['min']:.1f}%, {euc_pr['max']:.1f}%)"
                    else:
                        # Handle old format where EUC is a plain number
                        euc_pr_str = f"{model_metrics['euc_pr']:.1f}%"
                else:
                    euc_pr_str = "N/A"

                # Print all metrics for the current model
                print(
                    f"{label_name:<10} {class_name:<10} {model_name:<15} {auc_roc_str:<25} {auc_pr_str:<25} {euc_roc_str:<25} {euc_pr_str:<25}")

            # Add separator between classes
            print("-" * 120)

    # Print formatted output
    print("\n\n")
    print(formatted_output)

    return results, formatted_output