import os
import sys
import subprocess
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"


def _run_with_sudo(password: str, argv: list[str], omp_num_threads: int | None = 1) -> None:
    env = os.environ.copy()
    if omp_num_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_num_threads)

    subprocess.run(
        ["sudo", "-S", "-E", *argv],   # add -E flag
        input=password + "\n",
        text=True,
        check=True,
        env=env,
    )

def run_distributed_finetune_predict(
    *,
    sudo_password: str,
    nnodes: int,
    nproc_per_node: int,
    master_port: int,
    finetune_script: str,
    task_model: str,
    dataset: str,
    data_format: str,
    sampling_rate: int,
    already_format_channel_order: str,
    already_average_montage: str,
    allow_missing_channels: str,
    max_length_hour: str,
    polarity: int,
    eval_sub_dir: str,
    eval_results_dir: str,
    rewrite_results: str,
    prediction_slipping_step_second: int | None = None,
    prediction_slipping_step: int | None = None,
    smooth_result: str | None = None,
    need_spikes_1s_result: str | None = None,
    need_vw_1s_result: str | None = None,
    need_spike_localization_1s_result: str | None = None,
    # spike localization / vw / special heads
    task_model_2: str | None = None,
    model_selection_for_spike_localization: str | None = None,
    refer_spike_result_dir: str | None = None,
    refer_hm_model: str | None = None,
) -> None:
    """
    Mirrors the finetune_classification.py --predict commands in run_all_task_M2.sh.
    Supports both 1-second step (prediction_slipping_step_second) and pt-step (prediction_slipping_step).
    """
    python_bin = sys.executable

    argv = [
        python_bin,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        finetune_script,
        "--predict",
        "--model",
        "base_patch200_200",
        "--task_model",
        task_model,
        "--abs_pos_emb",
        "--dataset",
        dataset,
        "--data_format",
        data_format,
        "--sampling_rate",
        str(sampling_rate),
        "--already_format_channel_order",
        already_format_channel_order,
        "--already_average_montage",
        already_average_montage,
        "--allow_missing_channels",
        allow_missing_channels,
        "--max_length_hour",
        max_length_hour,
        "--polarity",
        str(polarity),
        "--eval_sub_dir",
        eval_sub_dir,
        "--eval_results_dir",
        eval_results_dir,
        "--rewrite_results",
        rewrite_results,
    ]

    # Optional args (match the shell script)
    if prediction_slipping_step_second is not None:
        argv += ["--prediction_slipping_step_second", str(prediction_slipping_step_second)]
    if prediction_slipping_step is not None:
        argv += ["--prediction_slipping_step", str(prediction_slipping_step)]
    if smooth_result is not None:
        argv += ["--smooth_result", smooth_result]
    if need_spikes_1s_result is not None:
        argv += ["--need_spikes_1s_result", need_spikes_1s_result]
    if need_vw_1s_result is not None:
        argv += ["--need_vw_1s_result", need_vw_1s_result]
    if need_spike_localization_1s_result is not None:
        argv += ["--need_spike_localization_1s_result", need_spike_localization_1s_result]

    if task_model_2 is not None:
        argv += ["--task_model_2", task_model_2]
    if model_selection_for_spike_localization is not None:
        argv += ["--model_selection_for_spike_localization", model_selection_for_spike_localization]
    if refer_spike_result_dir is not None:
        argv += ["--refer_spike_result_dir", refer_spike_result_dir]
    if refer_hm_model is not None:
        argv += ["--refer_hm_model", refer_hm_model]

    _run_with_sudo(sudo_password, argv, omp_num_threads=1)


def run_eeg_level_head(
    *,
    sudo_password: str,
    eeg_level_script: str,
    dataset: str,
    test_csv_dir: str,
    result_dir: str,
    task_model: str | None = None,
    align_spike_detection_and_location: bool = False,
) -> None:
    """
    Mirrors EEG_level_head.py calls in run_all_task_M2.sh.
    Some calls in the shell script omit --task_model (Sleep 5-stage section), so we allow task_model=None.
    """
    python_bin = sys.executable

    argv = [
        python_bin,
        eeg_level_script,
        "--mode",
        "predict",
        "--dataset",
        dataset,
        "--test_csv_dir",
        test_csv_dir,
        "--result_dir",
        result_dir,
    ]
    if task_model is not None:
        argv += ["--task_model", task_model]
    if align_spike_detection_and_location:
        argv += ["--align_spike_detection_and_location"]

    _run_with_sudo(sudo_password, argv, omp_num_threads=None)


if __name__ == "__main__":

    # Use the same positional-argument style as the original run_predict.py:
    #   sys.argv[1] = sudo_password
    #   sys.argv[2] = dataset_dir
    #   sys.argv[3] = result_dir
    #   sys.argv[4] = polarity
    if len(sys.argv) < 5:
        print("Usage: python run_predict_updated.py <sudo_password> <dataset_dir> <result_dir> <polarity>")
        sys.exit(2)

    from types import SimpleNamespace

    args = SimpleNamespace(
        sudo_password=str(sys.argv[1]),
        dataset_dir=str(sys.argv[2]),
        data_format="mat",
        sampling_rate=0,
        result_dir=str(sys.argv[3]),
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="no",
        max_length_hour="no",
        polarity=int(sys.argv[4]),
        need_spikes_1s_result="yes",
        need_vw_1s_result="yes",
        need_spike_localization_1s_result="yes",
        rewrite_results="no",
        master_port=12345,
        nproc_per_node=2,
    )

    # Assume this script lives next to finetune_classification.py and EEG_level_head.py (same as bash script)
    finetune_script = str((Path(__file__).resolve().parent / "finetune_classification.py").resolve())
    eeg_level_script = str((Path(__file__).resolve().parent / "EEG_level_head.py").resolve())

    dataset_dir = args.dataset_dir
    result_dir = args.result_dir

    # ---------------- Normal ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/NORMAL.pth",
        dataset="NORMAL",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_NORMAL_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results=args.rewrite_results,
    )
    run_eeg_level_head(
        sudo_password=args.sudo_password,
        eeg_level_script=eeg_level_script,
        dataset="NORMAL",
        task_model="checkpoints/morgoth/NORMAL_EEGlevel.pth",
        test_csv_dir=f"{result_dir}/pred_NORMAL_1sStep",
        result_dir=result_dir,
    )

    # ---------------- Slowing ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/SLOWING.pth",
        dataset="SLOWING",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_SLOWING_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results=args.rewrite_results,
    )
    for slowing_dataset in ["FOC_SLOWING", "GEN_SLOWING"]:
        run_eeg_level_head(
            sudo_password=args.sudo_password,
            eeg_level_script=eeg_level_script,
            dataset=slowing_dataset,
            task_model=f"checkpoints/morgoth/{slowing_dataset}_EEGlevel.pth",
            test_csv_dir=f"{result_dir}/pred_SLOWING_1sStep",
            result_dir=result_dir,
        )

    # ---------------- BS ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/BS.pth",
        dataset="BS",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_BS_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results=args.rewrite_results,
    )
    run_eeg_level_head(
        sudo_password=args.sudo_password,
        eeg_level_script=eeg_level_script,
        dataset="BS",
        task_model="checkpoints/morgoth/BS_EEGlevel.pth",
        test_csv_dir=f"{result_dir}/pred_BS_1sStep",
        result_dir=result_dir,
    )

    # ---------------- FOC/GEN SPIKES ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/FOCGENSPIKES.pth",
        dataset="FOC_GEN_SPIKES",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_FOCGENSPIKES_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results=args.rewrite_results,
    )
    for spikes_dataset in ["FOC_SPIKES", "GEN_SPIKES"]:
        run_eeg_level_head(
            sudo_password=args.sudo_password,
            eeg_level_script=eeg_level_script,
            dataset=spikes_dataset,
            task_model=f"checkpoints/morgoth/{spikes_dataset}_EEGlevel.pth",
            test_csv_dir=f"{result_dir}/pred_FOCGENSPIKES_1sStep",
            result_dir=result_dir,
        )

    # ---------------- SPIKES (8pt + optional 1s) ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/SPIKES.pth",
        dataset="SPIKES",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_SPIKES_8pStep",
        prediction_slipping_step=8,
        smooth_result="ema",
        need_spikes_1s_result=args.need_spikes_1s_result,
        rewrite_results=args.rewrite_results,
    )
    run_eeg_level_head(
        sudo_password=args.sudo_password,
        eeg_level_script=eeg_level_script,
        dataset="SPIKES",
        task_model="checkpoints/morgoth/SPIKES_EEGlevel.pth",
        test_csv_dir=f"{result_dir}/pred_SPIKES_1sStep",
        result_dir=result_dir,
        align_spike_detection_and_location=True,
    )

    # ---------------- Spike localization ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/SPIKE_localization_4.pth",
        task_model_2="checkpoints/morgoth/SPIKE_1channel_1.pth",
        model_selection_for_spike_localization="single_channel", # multi_channel
        refer_spike_result_dir=f"{result_dir}/pred_SPIKES_8pStep",
        dataset="SPIKE_localization",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_SPIKESLOC_8pStep",
        prediction_slipping_step=8,
        smooth_result="ema",
        need_spike_localization_1s_result=args.need_spike_localization_1s_result,
        rewrite_results=args.rewrite_results,
    )

    # ---------------- VW ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/VW_resampled_4.pth",
        refer_spike_result_dir=f"{result_dir}/pred_SPIKES_8pStep",
        dataset="VW",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_VW_8pStep",
        prediction_slipping_step=8,
        smooth_result="ema",
        need_vw_1s_result=args.need_vw_1s_result,
        rewrite_results=args.rewrite_results,
    )

    # ---------------- IIIC (HM) ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=1234,  # as in the bash script
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/sz_hm12_1.pth",
        refer_hm_model="checkpoints/morgoth/sz_hm13_3_fs.pth",
        dataset="IIIC_hm",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_IIIC_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results=args.rewrite_results,
    )
    for iiic_dataset in ["SEIZURE", "LPD", "GPD", "LRDA", "GRDA"]:
        run_eeg_level_head(
            sudo_password=args.sudo_password,
            eeg_level_script=eeg_level_script,
            dataset=iiic_dataset,
            task_model=f"checkpoints/morgoth/{iiic_dataset}_EEGlevel.pth",
            test_csv_dir=f"{result_dir}/pred_IIIC_1sStep/original_results",
            result_dir=result_dir,
        )

    # ---------------- Sleep 5 stage ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/SLEEPPSG.pth",
        dataset="SLEEPPSG",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_SLEEPSTAGING5class_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results=args.rewrite_results,
    )
    # Shell script omits --task_model here; keep optional to match.
    run_eeg_level_head(
        sudo_password=args.sudo_password,
        eeg_level_script=eeg_level_script,
        dataset="SLEEPPSG",
        task_model=None,
        test_csv_dir=f"{result_dir}/pred_SLEEPSTAGING5class_1sStep",
        result_dir=result_dir,
    )

    # ---------------- Sleep 6 stage ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/SLEEPPSG_6class.pth",
        dataset="SLEEPPSG_6class",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_SLEEPSTAGING6class_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results=args.rewrite_results,
    )

    # ---------------- Sleep arousal ----------------
    run_distributed_finetune_predict(
        sudo_password=args.sudo_password,
        nnodes=1,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        finetune_script=finetune_script,
        task_model="checkpoints/morgoth/SLEEPPSG_arousal.pth",
        dataset="SLEEP_AROUSAL",
        data_format=args.data_format,
        sampling_rate=args.sampling_rate,
        already_format_channel_order=args.already_format_channel_order,
        already_average_montage=args.already_average_montage,
        allow_missing_channels=args.allow_missing_channels,
        max_length_hour=args.max_length_hour,
        polarity=args.polarity,
        eval_sub_dir=dataset_dir,
        eval_results_dir=f"{result_dir}/pred_SLEEPAROUSAL_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results=args.rewrite_results,
    )


