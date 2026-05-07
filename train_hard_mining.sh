#!/usr/bin/env bash
# =============================================================================
# train_hard_mining.sh
# Continue training a fine-tuned EEG classification model with hard example mining.
#
# Edit the variables in the CONFIG section, then run:
#   bash train_hard_mining.sh  [mode]
#
# Modes (pass as first argument, default = A_last_n_full):
#   A_last_n_full          Mode A, last-N blocks, all data
#   A_all_full             Mode A, all layers,    all data
#   A_last_n_hard_kd       Mode A, last-N blocks, hard-only + Knowledge Distillation
#   A_all_hard_ewc         Mode A, all layers,    hard-only + EWC
#   B_adapter_full         Mode B, adapter,       all data
#   B_adapter_hard_kd      Mode B, adapter,       hard-only + KD
#   B_adapter_hard_ewc     Mode B, adapter,       hard-only + EWC
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIG — edit these paths and hyperparameters
# =============================================================================

FINETUNE_CKPT="/path/to/finetuned_checkpoint.pth"   # fine-tuned model from finetune_classification.py
TRAIN_DATA_DIR="/path/to/training_data"              # same training data used in finetune_classification.py
VAL_DATA_DIR="/path/to/val_data"                     # optional; leave empty "" to skip validation
OUTPUT_BASE_DIR="./hard_mining_output"               # parent directory for all run outputs
LOG_BASE_DIR="./hard_mining_logs"                    # tensorboard log directory (set "" to disable)

DATASET="IIIC"          # IIIC | IIIC_hm | TUAB | TUEP | TUEV | SLEEP
NB_CLASSES=6            # number of output classes
IS_BINARY=""            # set to "--is_binary" for binary classification, otherwise ""
MONTAGE="average"       # average | bipolar | combine

MODEL="base_patch200_200"
LAYER_SCALE_INIT=0.1
LAST_N_BLOCKS=4         # Mode A last_n: unfreeze last N transformer blocks (4 for base-12, 6 for large-24)
ADAPTER_REDUCTION=4     # Mode B: bottleneck reduction factor  (embed_dim // 4 = 50 for base)
ADAPTER_DROPOUT=0.0

EPOCHS=10
BATCH_SIZE=64
LR=1e-4
MIN_LR=1e-6
WARMUP_LR=1e-6
WARMUP_EPOCHS=2
WEIGHT_DECAY=0.05
CLIP_GRAD=3.0
SMOOTHING=0.0           # label smoothing; 0 to disable

HARD_RATIO=0.3          # top 30 % by loss
HARD_MIN_LOSS=-1        # alternative threshold; -1 to disable
REMINE_EVERY=5          # re-mine every N epochs; 0 = mine once only

EWC_LAMBDA=1000.0       # EWC regularisation strength
EWC_SAMPLES=2000        # samples used to estimate Fisher Information Matrix
KD_WEIGHT=1.0           # Knowledge Distillation loss weight
KD_TEMPERATURE=2.0      # KD softmax temperature

SAVE_CKPT_FREQ=1
NUM_WORKERS=8

# =============================================================================
# Helper: build common flags shared by all runs
# =============================================================================
common_flags() {
    echo \
        --finetune      "${FINETUNE_CKPT}" \
        --train_data_dir "${TRAIN_DATA_DIR}" \
        --dataset        "${DATASET}" \
        --nb_classes     "${NB_CLASSES}" \
        ${IS_BINARY} \
        --train_eeg_montage "${MONTAGE}" \
        --model          "${MODEL}" \
        --layer_scale_init_value "${LAYER_SCALE_INIT}" \
        --epochs         "${EPOCHS}" \
        --batch_size     "${BATCH_SIZE}" \
        --lr             "${LR}" \
        --min_lr         "${MIN_LR}" \
        --warmup_lr      "${WARMUP_LR}" \
        --warmup_epochs  "${WARMUP_EPOCHS}" \
        --weight_decay   "${WEIGHT_DECAY}" \
        --clip_grad      "${CLIP_GRAD}" \
        --smoothing      "${SMOOTHING}" \
        --hard_ratio     "${HARD_RATIO}" \
        --hard_min_loss  "${HARD_MIN_LOSS}" \
        --remine_every   "${REMINE_EVERY}" \
        --save_ckpt_freq "${SAVE_CKPT_FREQ}" \
        --num_workers    "${NUM_WORKERS}"
}

val_flag() {
    [ -n "${VAL_DATA_DIR}" ] && echo --val_data_dir "${VAL_DATA_DIR}" || echo ""
}

log_flag() {
    local tag="$1"
    [ -n "${LOG_BASE_DIR}" ] && echo --log_dir "${LOG_BASE_DIR}/${tag}" || echo ""
}

# =============================================================================
# Run functions — one per mode
# =============================================================================

run_A_last_n_full() {
    local out="${OUTPUT_BASE_DIR}/A_last_n_full"
    echo "[Mode] A — last ${LAST_N_BLOCKS} blocks — full data"
    python hard_mining.py \
        $(common_flags) \
        $(val_flag) \
        $(log_flag "A_last_n_full") \
        --mode            A \
        --finetune_layers last_n \
        --last_n_blocks   "${LAST_N_BLOCKS}" \
        --data_strategy   full \
        --output_dir      "${out}"
}

run_A_all_full() {
    local out="${OUTPUT_BASE_DIR}/A_all_full"
    echo "[Mode] A — all layers — full data"
    python hard_mining.py \
        $(common_flags) \
        $(val_flag) \
        $(log_flag "A_all_full") \
        --mode            A \
        --finetune_layers all \
        --data_strategy   full \
        --output_dir      "${out}"
}

run_A_last_n_hard_kd() {
    local out="${OUTPUT_BASE_DIR}/A_last_n_hard_kd"
    echo "[Mode] A — last ${LAST_N_BLOCKS} blocks — hard-only + Knowledge Distillation"
    python hard_mining.py \
        $(common_flags) \
        $(val_flag) \
        $(log_flag "A_last_n_hard_kd") \
        --mode            A \
        --finetune_layers last_n \
        --last_n_blocks   "${LAST_N_BLOCKS}" \
        --data_strategy   hard_only \
        --cl_method       kd \
        --kd_weight       "${KD_WEIGHT}" \
        --kd_temperature  "${KD_TEMPERATURE}" \
        --output_dir      "${out}"
}

run_A_all_hard_ewc() {
    local out="${OUTPUT_BASE_DIR}/A_all_hard_ewc"
    echo "[Mode] A — all layers — hard-only + EWC"
    python hard_mining.py \
        $(common_flags) \
        $(val_flag) \
        $(log_flag "A_all_hard_ewc") \
        --mode            A \
        --finetune_layers all \
        --data_strategy   hard_only \
        --cl_method       ewc \
        --ewc_lambda      "${EWC_LAMBDA}" \
        --ewc_samples     "${EWC_SAMPLES}" \
        --output_dir      "${out}"
}

run_B_adapter_full() {
    local out="${OUTPUT_BASE_DIR}/B_adapter_full"
    echo "[Mode] B — adapter (reduction=${ADAPTER_REDUCTION}) — full data"
    python hard_mining.py \
        $(common_flags) \
        $(val_flag) \
        $(log_flag "B_adapter_full") \
        --mode              B \
        --adapter_reduction "${ADAPTER_REDUCTION}" \
        --adapter_dropout   "${ADAPTER_DROPOUT}" \
        --data_strategy     full \
        --output_dir        "${out}"
}

run_B_adapter_hard_kd() {
    local out="${OUTPUT_BASE_DIR}/B_adapter_hard_kd"
    echo "[Mode] B — adapter — hard-only + Knowledge Distillation"
    python hard_mining.py \
        $(common_flags) \
        $(val_flag) \
        $(log_flag "B_adapter_hard_kd") \
        --mode              B \
        --adapter_reduction "${ADAPTER_REDUCTION}" \
        --adapter_dropout   "${ADAPTER_DROPOUT}" \
        --data_strategy     hard_only \
        --cl_method         kd \
        --kd_weight         "${KD_WEIGHT}" \
        --kd_temperature    "${KD_TEMPERATURE}" \
        --output_dir        "${out}"
}

run_B_adapter_hard_ewc() {
    local out="${OUTPUT_BASE_DIR}/B_adapter_hard_ewc"
    echo "[Mode] B — adapter — hard-only + EWC"
    python hard_mining.py \
        $(common_flags) \
        $(val_flag) \
        $(log_flag "B_adapter_hard_ewc") \
        --mode              B \
        --adapter_reduction "${ADAPTER_REDUCTION}" \
        --adapter_dropout   "${ADAPTER_DROPOUT}" \
        --also_finetune_backbone \
        --last_n_blocks     "${LAST_N_BLOCKS}" \
        --data_strategy     hard_only \
        --cl_method         ewc \
        --ewc_lambda        "${EWC_LAMBDA}" \
        --ewc_samples       "${EWC_SAMPLES}" \
        --output_dir        "${out}"
}

# =============================================================================
# Dispatch
# =============================================================================

MODE="${1:-A_last_n_full}"

echo "============================================================"
echo " Hard-mining training"
echo " Run mode : ${MODE}"
echo " Checkpoint: ${FINETUNE_CKPT}"
echo " Data      : ${TRAIN_DATA_DIR}"
echo " Output    : ${OUTPUT_BASE_DIR}/${MODE}"
echo "============================================================"

case "${MODE}" in
    A_last_n_full)     run_A_last_n_full     ;;
    A_all_full)        run_A_all_full        ;;
    A_last_n_hard_kd)  run_A_last_n_hard_kd  ;;
    A_all_hard_ewc)    run_A_all_hard_ewc    ;;
    B_adapter_full)    run_B_adapter_full    ;;
    B_adapter_hard_kd) run_B_adapter_hard_kd ;;
    B_adapter_hard_ewc)run_B_adapter_hard_ewc;;
    all)
        # run every mode sequentially
        run_A_last_n_full
        run_A_all_full
        run_A_last_n_hard_kd
        run_A_all_hard_ewc
        run_B_adapter_full
        run_B_adapter_hard_kd
        run_B_adapter_hard_ewc
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Valid modes: A_last_n_full | A_all_full | A_last_n_hard_kd | A_all_hard_ewc"
        echo "             B_adapter_full | B_adapter_hard_kd | B_adapter_hard_ewc | all"
        exit 1
        ;;
esac

echo "============================================================"
echo " Finished: ${MODE}"
echo "============================================================"
