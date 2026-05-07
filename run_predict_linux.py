import os
import sys
import subprocess
import warnings
os.environ["PYTHONWARNINGS"]="ignore"
warnings.filterwarnings("ignore")
from pathlib import Path

def run_distributed_event(
    sudo_password,
    nnodes,
    nproc_per_node,
    master_port,
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
    python_bin = sys.executable  # equivalent to $(which python)

    cmd = (
        f"echo '{sudo_password}' | sudo -S OMP_NUM_THREADS=1 {python_bin} "
        f"-m torch.distributed.run "
        f"--nnodes={nnodes} "
        f"--nproc_per_node={nproc_per_node} "
        f"--master_port={master_port} "
        f"../morgoth/finetune_classification.py "
        f"--abs_pos_emb "
        f"--model base_patch200_200 "
        f"--predict "
        f"--task_model {task_model} "
        f"--dataset {dataset} "
        f"--data_format {data_format} "
        f"--sampling_rate {sampling_rate} "
        f"--already_format_channel_order {already_format_channel_order} "
        f"--already_average_montage {already_average_montage} "
        f"--allow_missing_channels {allow_missing_channels} "
        f"--eval_sub_dir {eval_sub_dir} "
        f"--eval_results_dir {eval_results_dir} "
        f"--prediction_slipping_step_second {prediction_slipping_step_second} "
        f"--rewrite_results {rewrite_results} "
        f"--polarity {polarity} "
    )

    subprocess.run(cmd, shell=True, check=True)

def run_distributed_event_pt(
    sudo_password,
    nnodes,
    nproc_per_node,
    master_port,
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
    python_bin = sys.executable  # equivalent to $(which python)
    cmd = (
        f"echo '{sudo_password}' | sudo -S OMP_NUM_THREADS=1 {python_bin} "
        f"-m torch.distributed.run "
        f"--nnodes={nnodes} "
        f"--nproc_per_node={nproc_per_node} "
        f"--master_port={master_port} "
        f"../morgoth/finetune_classification.py "
        f"--abs_pos_emb "
        f"--model base_patch200_200 "
        f"--predict "
        f"--task_model {task_model} "
        f"--dataset {dataset} "
        f"--data_format {data_format} "
        f"--sampling_rate {sampling_rate} "
        f"--already_format_channel_order {already_format_channel_order} "
        f"--already_average_montage {already_average_montage} "
        f"--allow_missing_channels {allow_missing_channels} "
        f"--eval_sub_dir {eval_sub_dir} "
        f"--eval_results_dir {eval_results_dir} "
        f"--prediction_slipping_step {prediction_slipping_step} "
        f"--rewrite_results {rewrite_results} "
        f"--polarity {polarity} "
        f"--need_spikes_1s_result {need_spikes_1s_result} "
        f"--smooth_result {smooth_result}"
    )

    subprocess.run(cmd, shell=True, check=True)

def run_distributed_eeg(
    sudo_password,
    task_model,
    dataset,
    test_csv_dir,
    result_dir,
):
    python_bin = sys.executable  # equivalent to $(which python)

    cmd = (
        f"echo '{sudo_password}' | sudo -S OMP_NUM_THREADS=1 {python_bin} "
        f"../morgoth/EEG_level_head.py "
        f"--mode predict "
        f"--task_model {task_model} "
        f"--dataset {dataset} "
        f"--test_csv_dir {test_csv_dir} "
        f"--result_dir {result_dir} "
    )

    subprocess.run(cmd, shell=True, check=True)




if __name__ == "__main__":
    # current_directory = os.getcwd()
    # print(f"Current working directory: {current_directory}")
    #
    #######################################################################
    # Normal:
    # event-level
    run_distributed_event(
        sudo_password=sys.argv[1],
        nnodes=1,
        nproc_per_node=2,
        master_port=12345,
        task_model="../morgoth/checkpoints/NORMAL.pth",
        dataset="NORMAL",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=str(sys.argv[2]),
        eval_results_dir=str(sys.argv[3])+"/pred_NORMAL_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=int(sys.argv[4]),
    )

    # EEG-level
    run_distributed_eeg(
        sudo_password=sys.argv[1],
        task_model="../morgoth/checkpoints/NORMAL_EEGlevel.pth",
        dataset="NORMAL",
        test_csv_dir=str(sys.argv[3])+"/pred_NORMAL_1sStep",
        result_dir=str(sys.argv[3])
    )


    #######################################################################
    # Slowing:
    # event-level
    run_distributed_event(
        sudo_password=sys.argv[1],
        nnodes=1,
        nproc_per_node=2,
        master_port=12345,
        task_model="../morgoth/checkpoints/SLOWING.pth",
        dataset="SLOWING",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=str(sys.argv[2]),
        eval_results_dir=str(sys.argv[3])+"/pred_SLOWING_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=int(sys.argv[4]),
    )

    # EEG-level
    FOC_GEN_SLOWING_datasets=["FOC_SLOWING","GEN_SLOWING"]
    for FOC_GEN_SLOWING_dataset in FOC_GEN_SLOWING_datasets:
        run_distributed_eeg(
            sudo_password=sys.argv[1],
            task_model="../morgoth/checkpoints/"+FOC_GEN_SLOWING_dataset+"_EEGlevel.pth",
            dataset=FOC_GEN_SLOWING_dataset,
            test_csv_dir=str(sys.argv[3])+"/pred_SLOWING_1sStep",
            result_dir=str(sys.argv[3])
        )

    #######################################################################
    # BS:
    # event-level
    run_distributed_event(
        sudo_password=sys.argv[1],
        nnodes=1,
        nproc_per_node=2,
        master_port=12345,
        task_model="../morgoth/checkpoints/BS.pth",
        dataset="BS",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=str(sys.argv[2]),
        eval_results_dir=str(sys.argv[3])+"/pred_BS_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=int(sys.argv[4]),
    )

    # EEG-level
    run_distributed_eeg(
        sudo_password=sys.argv[1],
        task_model="../morgoth/checkpoints/BS_EEGlevel.pth",
        dataset="BS",
        test_csv_dir=str(sys.argv[3])+"/pred_BS_1sStep",
        result_dir=str(sys.argv[3])
    )

    #######################################################################
    # FOC GEN SPIKES:
    # event-level
    run_distributed_event(
        sudo_password=sys.argv[1],
        nnodes=1,
        nproc_per_node=2,
        master_port=12345,
        task_model="../morgoth/checkpoints/FOCGENSPIKES.pth",
        dataset="FOC_GEN_SPIKES",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=str(sys.argv[2]),
        eval_results_dir=str(sys.argv[3])+"/pred_FOCGENSPIKES_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=int(sys.argv[4]),
    )

    # EEG-level
    FOC_GEN_SPIKES_datasets=["FOC_SPIKES","GEN_SPIKES"]
    for FOC_GEN_SPIKES_dataset in FOC_GEN_SPIKES_datasets:
        run_distributed_eeg(
            sudo_password=sys.argv[1],
            task_model="../morgoth/checkpoints/"+FOC_GEN_SPIKES_dataset+"_EEGlevel.pth",
            dataset=FOC_GEN_SPIKES_dataset,
            test_csv_dir=str(sys.argv[3])+"/pred_FOCGENSPIKES_1sStep",
            result_dir=str(sys.argv[3])
        )

    #######################################################################
    # Spikes:
    # no need if run Spike 8pt
    # run_distributed_event(
    #     sudo_password=sys.argv[1],
    #     nnodes=1,
    #     nproc_per_node=2,
    #     master_port=12345,
    #     task_model="../morgoth/checkpoints/SPIKES.pth",
    #     dataset="SPIKES",
    #     data_format="mat",
    #     sampling_rate=0,
    #     already_format_channel_order="no",
    #     already_average_montage="no",
    #     allow_missing_channels="yes",
    #     eval_sub_dir=str(sys.argv[2]),
    #     eval_results_dir=str(sys.argv[3])+"/pred_SPIKES_1sStep",
    #     prediction_slipping_step_second=1,
    #     rewrite_results="no",
    #     polarity=int(sys.argv[4]),
    # )

    # Spike 8pt:
    run_distributed_event_pt(
        sudo_password=sys.argv[1],
        nnodes=1,
        nproc_per_node=2,
        master_port=12345,
        task_model="../morgoth/checkpoints/SPIKES.pth",
        dataset="SPIKES",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=str(sys.argv[2]),
        eval_results_dir=str(sys.argv[3]) + "/pred_SPIKES_8pStep",
        prediction_slipping_step=8,
        rewrite_results="no",
        polarity=int(sys.argv[4]),
        need_spikes_1s_result="yes", # generate 1s results in pred_SPIKES_1sStep
        smooth_result="ema"
    )

    # EEG-level
    run_distributed_eeg(
        sudo_password=sys.argv[1],
        task_model="../morgoth/checkpoints/SPIKES_EEGlevel.pth",
        dataset="SPIKES",
        test_csv_dir=str(sys.argv[3])+"/pred_SPIKES_1sStep",
        result_dir=str(sys.argv[3])
    )


    #######################################################################
    # IIIC:
    # event-level
    run_distributed_event(
        sudo_password=sys.argv[1],
        nnodes=1,
        nproc_per_node=2,
        master_port=12345,
        task_model="../morgoth/checkpoints/sz_hm6.pth",
        dataset="IIIC",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=str(sys.argv[2]),
        eval_results_dir=str(sys.argv[3])+"/pred_IIIC_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=int(sys.argv[4]),
    )

    # EEG-level
    IIIC_datasets=["SEIZURE","LPD","GPD","LRDA","GRDA"]
    for IIIC_dataset in IIIC_datasets:
        run_distributed_eeg(
            sudo_password=sys.argv[1],
            task_model="../morgoth/checkpoints/"+IIIC_dataset+"_EEGlevel.pth",
            dataset=IIIC_dataset,
            test_csv_dir=str(sys.argv[3])+"/pred_IIIC_1sStep",
            result_dir=str(sys.argv[3])
        )

    #######################################################################
    # SLEEP 5:
    # event-level
    run_distributed_event(
        sudo_password=sys.argv[1],
        nnodes=1,
        nproc_per_node=2,
        master_port=12345,
        task_model="../morgoth/checkpoints/SLEEPPSG.pth",
        dataset="SLEEPPSG",
        data_format="mat",
        sampling_rate=0,
        already_format_channel_order="no",
        already_average_montage="no",
        allow_missing_channels="yes",
        eval_sub_dir=str(sys.argv[2]),
        eval_results_dir=str(sys.argv[3])+"/pred_SLEEPPSG_1sStep",
        prediction_slipping_step_second=1,
        rewrite_results="no",
        polarity=int(sys.argv[4]),
    )

    # EEG-level
    run_distributed_eeg(
        sudo_password=sys.argv[1],
        task_model="../morgoth/checkpoints/SLEEPPSG_EEGlevel.pth",
        dataset="SLEEPPSG",
        test_csv_dir=str(sys.argv[3])+"/pred_SLEEPPSG_1sStep",
        result_dir=str(sys.argv[3])
    )




