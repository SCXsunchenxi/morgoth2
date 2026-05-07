import os
import sys
import subprocess
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
from pathlib import Path

def run_mac_event(
        task_model,
        dataset,
        data_format,
        sampling_rate,
        already_format_channel_order,
        already_average_montage,
        allow_missing_channels,
        max_length_hour,
        eval_sub_dir,
        eval_results_dir,
        prediction_slipping_step_second,
        polarity,
        rewrite_results,
):
    # use MPS fallback (allow CPU back)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Use: {device}")

    python_bin = sys.executable

    cmd = [
        python_bin,
        "finetune_classification.py",
        "--abs_pos_emb",
        "--model", "base_patch200_200",
        "--predict",
        "--task_model", task_model,
        "--dataset", dataset,
        "--data_format", data_format,
        "--sampling_rate", sampling_rate,
        "--already_format_channel_order", already_format_channel_order,
        "--already_average_montage", already_average_montage,
        "--allow_missing_channels", allow_missing_channels,
        "--max_length_hour", max_length_hour,
        "--eval_sub_dir", eval_sub_dir,
        "--eval_results_dir", eval_results_dir,
        "--prediction_slipping_step_second", prediction_slipping_step_second,
        "--polarity", polarity,
        "--rewrite_results", rewrite_results,
        "--num_workers", "0",
        "--device", device,
    ]

    subprocess.run(cmd, check=True)


def run_mac_event_pt(
        task_model,
        dataset,
        data_format,
        sampling_rate,
        already_format_channel_order,
        already_average_montage,
        allow_missing_channels,
        max_length_hour,
        eval_sub_dir,
        eval_results_dir,
        prediction_slipping_step,
        polarity,
        rewrite_results,
        need_spikes_1s_result,
        smooth_result
):
    # use MPS fallback (allow CPU back)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Use: {device}")

    python_bin = sys.executable

    cmd = [
        python_bin,
        "finetune_classification.py",
        "--abs_pos_emb",
        "--model", "base_patch200_200",
        "--predict",
        "--task_model", task_model,
        "--dataset", dataset,
        "--data_format", data_format,
        "--sampling_rate", sampling_rate,
        "--already_format_channel_order", already_format_channel_order,
        "--already_average_montage", already_average_montage,
        "--allow_missing_channels", allow_missing_channels,
        "--max_length_hour", max_length_hour,
        "--eval_sub_dir", eval_sub_dir,
        "--eval_results_dir", eval_results_dir,
        "--prediction_slipping_step", prediction_slipping_step,
        "--polarity", polarity,
        "--rewrite_results", rewrite_results,
        "--num_workers", "0",
        "--device", device,
        "--need_spikes_1s_result", need_spikes_1s_result,
        "--smooth_result", smooth_result,
    ]

    subprocess.run(cmd, check=True)


def run_mac_eeg(
        task_model,
        dataset,
        test_csv_dir,
        result_dir,
):
    python_bin = sys.executable  # equivalent to $(which python)

    cmd = (
        f"OMP_NUM_THREADS=1 {python_bin} "
        f"EEG_level_head.py "
        f"--mode predict "
        f"--task_model {task_model} "
        f"--dataset {dataset} "
        f"--test_csv_dir {test_csv_dir} "
        f"--result_dir {result_dir} "
    )

    subprocess.run(cmd, shell=True, check=True)







if __name__ == "__main__":

    eval_sub_dir = "/Users/chenxisun/Documents/sshcode/EEG_report/Morgoth2/test_data/IIIC/segments_raw/"
    eval_results_dir = "/Users/chenxisun/Documents/sshcode/EEG_report/Morgoth2/test_data/IIIC/pred"

    # #########################################
    # Normal:
    # event-level
    run_mac_event(
        task_model="checkpoints/NORMAL.pth",
        dataset="NORMAL",
        data_format="mat",
        sampling_rate="0",
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        max_length_hour="no",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=f"{eval_results_dir}/pred_NORMAL_1sStep",
        prediction_slipping_step_second="1",
        polarity="1",
        rewrite_results="no"
    )

    # EEG-level
    run_mac_eeg(
        task_model="checkpoints/NORMAL_EEGlevel.pth",
        dataset="NORMAL",
        test_csv_dir=f"{eval_results_dir}/pred_NORMAL_1sStep",
        result_dir=eval_results_dir
    )

    # #########################################
    # Slowing:
    # event-level
    run_mac_event(
        task_model="checkpoints/SLOWING.pth",
        dataset="SLOWING",
        data_format="mat",
        sampling_rate="0",
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        max_length_hour="no",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=f"{eval_results_dir}/pred_SLOWING_1sStep",
        prediction_slipping_step_second="1",
        polarity="1",
        rewrite_results="no"
    )

    # EEG-level
    FOC_GEN_SLOWING_datasets = ["FOC_SLOWING", "GEN_SLOWING"]
    for FOC_GEN_SLOWING_dataset in FOC_GEN_SLOWING_datasets:
        run_mac_eeg(
            task_model=f"checkpoints/{FOC_GEN_SLOWING_dataset}_EEGlevel.pth",
            dataset=FOC_GEN_SLOWING_dataset,
            test_csv_dir=f"{eval_results_dir}/pred_SLOWING_1sStep",
            result_dir=eval_results_dir
        )

    # #########################################
    # BS:
    # event-level
    run_mac_event(
        task_model="checkpoints/BS.pth",
        dataset="BS",
        data_format="mat",
        sampling_rate="0",
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        max_length_hour="no",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=f"{eval_results_dir}/pred_BS_1sStep",
        prediction_slipping_step_second="1",
        polarity="1",
        rewrite_results="no"
    )

    # EEG-level
    run_mac_eeg(
        task_model="checkpoints/BS_EEGlevel.pth",
        dataset="BS",
        test_csv_dir=f"{eval_results_dir}/pred_BS_1sStep",
        result_dir=eval_results_dir
    )

    # #########################################
    # FOC GEN SPIKES:
    # event-level
    run_mac_event(
        task_model="checkpoints/FOCGENSPIKES.pth",
        dataset="FOC_GEN_SPIKES",
        data_format="mat",
        sampling_rate="0",
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        max_length_hour="no",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=f"{eval_results_dir}/pred_FOCGENSPIKES_1sStep",
        prediction_slipping_step_second="1",
        polarity="1",
        rewrite_results="no",
    )

    # EEG-level
    FOC_GEN_SPIKES_datasets = ["FOC_SPIKES", "GEN_SPIKES"]
    for FOC_GEN_SPIKES_dataset in FOC_GEN_SPIKES_datasets:
        run_mac_eeg(
            task_model=f"checkpoints/{FOC_GEN_SPIKES_dataset}_EEGlevel.pth",
            dataset=FOC_GEN_SPIKES_dataset,
            test_csv_dir=f"{eval_results_dir}/pred_FOCGENSPIKES_1sStep",
            result_dir=eval_results_dir
        )

    # #########################################
    # Spike 8pt:
    # event-level
    run_mac_event_pt(
        task_model="checkpoints/SPIKES.pth",
        dataset="SPIKES",
        data_format="mat",
        sampling_rate="0",
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        max_length_hour="no",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=f"{eval_results_dir}/pred_SPIKES_8pStep",
        prediction_slipping_step="8",
        polarity="1",
        rewrite_results="no",
        need_spikes_1s_result="yes",  # generate 1s results in pred_SPIKES_1sStep
        smooth_result='ema'
    )

    run_mac_eeg(
        task_model="checkpoints/SPIKES_EEGlevel.pth",
        dataset="SPIKES",
        test_csv_dir=f"{eval_results_dir}/pred_SPIKES_1sStep",
        result_dir=eval_results_dir
    )

    # #########################################
    # IIIC:
    # event-level
    run_mac_event(
        task_model="checkpoints/sz_hm6.pth",
        dataset="IIIC",
        data_format="mat",
        sampling_rate="0",
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        max_length_hour="no",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=f"{eval_results_dir}/pred_IIIC_1sStep",
        prediction_slipping_step_second="1",
        polarity="1",
        rewrite_results="no",
    )

    # EEG-level
    IIIC_datasets = ["SEIZURE", "LPD", "GPD", "LRDA", "GRDA"]
    for IIIC_dataset in IIIC_datasets:
        run_mac_eeg(
            task_model=f"checkpoints/{IIIC_dataset}_EEGlevel.pth",
            dataset=IIIC_dataset,
            test_csv_dir=f"{eval_results_dir}/pred_IIIC_1sStep",
            result_dir=eval_results_dir
        )

    # #########################################
    # SLEEP 5:
    # event-level
    run_mac_event(
        task_model="checkpoints/ss_hm_1.pth",
        dataset="SLEEPPSG",
        data_format="mat",
        sampling_rate="0",
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        max_length_hour="no",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=f"{eval_results_dir}/pred_SLEEPPSG_1sStep",
        prediction_slipping_step_second="1",
        polarity="1",
        rewrite_results="no",
    )

    # EEG-level
    run_mac_eeg(
        task_model="checkpoints/SLEEPPSG_EEGlevel.pth",
        dataset="SLEEPPSG",
        test_csv_dir=f"{eval_results_dir}/pred_SLEEPPSG_1sStep",
        result_dir=eval_results_dir
    )

