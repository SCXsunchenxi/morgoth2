import os
import sys
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def _base_env():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    return env

def run_windows_event(
    task_model,
    dataset,
    data_format,
    sampling_rate,
    already_format_channel_order,
    already_average_montage,
    allow_missing_channels,
    eval_sub_dir,
    eval_results_dir,
    prediction_slipping_step_second,
    rewrite_results,
    polarity,
):
    python_bin = sys.executable
    script = (Path(__file__).resolve().parent / "../morgoth/finetune_classification.py").resolve()

    args = [
        python_bin,
        str(script),
        "--abs_pos_emb",
        "--model", "base_patch200_200",
        "--predict",
        "--task_model", str(task_model),
        "--dataset", str(dataset),
        "--data_format", str(data_format),
        "--sampling_rate", str(sampling_rate),
        "--already_format_channel_order", str(already_format_channel_order),
        "--already_average_montage", str(already_average_montage),
        "--allow_missing_channels", str(allow_missing_channels),
        "--eval_sub_dir", str(eval_sub_dir),
        "--eval_results_dir", str(eval_results_dir),
        "--prediction_slipping_step_second", str(prediction_slipping_step_second),
        "--rewrite_results", str(rewrite_results),
        "--polarity", str(polarity),
    ]
    subprocess.run(args, check=True, env=_base_env())



def run_windows_event_pt(
    task_model,
    dataset,
    data_format,
    sampling_rate,
    already_format_channel_order,
    already_average_montage,
    allow_missing_channels,
    eval_sub_dir,
    eval_results_dir,
    prediction_slipping_step,
    rewrite_results,
    polarity,
    need_spikes_1s_result,
    smooth_result
):
    python_bin = sys.executable
    script = (Path(__file__).resolve().parent / "../morgoth/finetune_classification.py").resolve()

    args = [
        python_bin,
        str(script),
        "--abs_pos_emb",
        "--model", "base_patch200_200",
        "--predict",
        "--task_model", str(task_model),
        "--dataset", str(dataset),
        "--data_format", str(data_format),
        "--sampling_rate", str(sampling_rate),
        "--already_format_channel_order", str(already_format_channel_order),
        "--already_average_montage", str(already_average_montage),
        "--allow_missing_channels", str(allow_missing_channels),
        "--eval_sub_dir", str(eval_sub_dir),
        "--eval_results_dir", str(eval_results_dir),
        "--prediction_slipping_step", str(prediction_slipping_step),
        "--rewrite_results", str(rewrite_results),
        "--polarity", str(polarity),
        "--need_spikes_1s_result", str(need_spikes_1s_result),
        "--smooth_result", str(smooth_result),
    ]
    subprocess.run(args, check=True, env=_base_env())



def run_windows_eeg(
    task_model,
    dataset,
    test_csv_dir,
    result_dir,
):
    python_bin = sys.executable
    script = (Path(__file__).resolve().parent / "../morgoth/EEG_level_head.py").resolve()

    args = [
        python_bin,
        str(script),
        "--mode", "predict",
        "--task_model", str(task_model),
        "--dataset", str(dataset),
        "--test_csv_dir", str(test_csv_dir),
        "--result_dir", str(result_dir),
    ]
    subprocess.run(args, check=True, env=_base_env())

if __name__ == "__main__":

    eval_sub_dir = Path(sys.argv[2]).resolve()
    eval_results_root = Path(sys.argv[3]).resolve()
    polarity = int(sys.argv[4])

    #######################################################################
    # Normal:
    # event-level
    run_windows_event(
        task_model=Path("../morgoth/checkpoints/NORMAL.pth").resolve(),
        dataset="NORMAL",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=eval_results_root / "pred_NORMAL_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=polarity,
    )

    # EEG-level
    run_windows_eeg(
        task_model=Path("../morgoth/checkpoints/NORMAL_EEGlevel.pth").resolve(),
        dataset="NORMAL",
        test_csv_dir=eval_results_root / "pred_NORMAL_1sStep",
        result_dir=eval_results_root,
    )

    #######################################################################
    # Slowing:
    # event-level
    run_windows_event(
        task_model=Path("../morgoth/checkpoints/SLOWING.pth").resolve(),
        dataset="SLOWING",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=eval_results_root / "pred_SLOWING_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=polarity,
    )

    # EEG-level
    for ds in ["FOC_SLOWING", "GEN_SLOWING"]:
        run_windows_eeg(
            task_model=Path(f"../morgoth/checkpoints/{ds}_EEGlevel.pth").resolve(),
            dataset=ds,
            test_csv_dir=eval_results_root / "pred_SLOWING_1sStep",
            result_dir=eval_results_root,
        )

    #######################################################################
    # BS:
    # event-level
    run_windows_event(
        task_model=Path("../morgoth/checkpoints/BS.pth").resolve(),
        dataset="BS",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=eval_results_root / "pred_BS_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=polarity,
    )

    # EEG-level
    run_windows_eeg(
        task_model=Path("../morgoth/checkpoints/BS_EEGlevel.pth").resolve(),
        dataset="BS",
        test_csv_dir=eval_results_root / "pred_BS_1sStep",
        result_dir=eval_results_root,
    )

    #######################################################################
    # FOC GEN SPIKES:
    # event-level
    run_windows_event(
        task_model=Path("../morgoth/checkpoints/FOCGENSPIKES.pth").resolve(),
        dataset="FOC_GEN_SPIKES",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=eval_results_root / "pred_FOCGENSPIKES_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=polarity,
    )

    # EEG-level
    for ds in ["FOC_SPIKES", "GEN_SPIKES"]:
        run_windows_eeg(
            task_model=Path(f"../morgoth/checkpoints/{ds}_EEGlevel.pth").resolve(),
            dataset=ds,
            test_csv_dir=eval_results_root / "pred_FOCGENSPIKES_1sStep",
            result_dir=eval_results_root,
        )

    #######################################################################
    # Spikes: 8pt version (generates 1s results into pred_SPIKES_1sStep)
    run_windows_event_pt(
        task_model=Path("../morgoth/checkpoints/SPIKES.pth").resolve(),
        dataset="SPIKES",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=eval_results_root / "pred_SPIKES_8pStep",
        prediction_slipping_step=8,
        rewrite_results="no",
        polarity=polarity,
        need_spikes_1s_result="yes",
        smooth_result="ema",
    )

    # EEG-level
    run_windows_eeg(
        task_model=Path("../morgoth/checkpoints/SPIKES_EEGlevel.pth").resolve(),
        dataset="SPIKES",
        test_csv_dir=eval_results_root / "pred_SPIKES_1sStep",
        result_dir=eval_results_root,
    )

    #######################################################################
    # IIIC:
    # event-level
    run_windows_event(
        task_model=Path("../morgoth/checkpoints/sz_hm6.pth").resolve(),
        dataset="IIIC",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=eval_results_root / "pred_IIIC_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=polarity,
    )

    # EEG-level
    for ds in ["SEIZURE", "LPD", "GPD", "LRDA", "GRDA"]:
        run_windows_eeg(
            task_model=Path(f"../morgoth/checkpoints/{ds}_EEGlevel.pth").resolve(),
            dataset=ds,
            test_csv_dir=eval_results_root / "pred_IIIC_1sStep",
            result_dir=eval_results_root,
        )

    #######################################################################
    # SLEEP 5:
    # event-level
    run_windows_event(
        task_model=Path("../morgoth/checkpoints/SLEEPPSG.pth").resolve(),
        dataset="SLEEPPSG",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=eval_sub_dir,
        eval_results_dir=eval_results_root / "pred_SLEEPPSG_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=polarity,
    )

    # EEG-level
    run_windows_eeg(
        task_model=Path("../morgoth/checkpoints/SLEEPPSG_EEGlevel.pth").resolve(),
        dataset="SLEEPPSG",
        test_csv_dir=eval_results_root / "pred_SLEEPPSG_1sStep",
        result_dir=eval_results_root,
    )