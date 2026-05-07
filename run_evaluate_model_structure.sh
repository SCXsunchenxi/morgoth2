#!/bin/bash
# run_evaluate_model_structure.sh
# ================================
# Internal-structure evaluation for three model types:
#   tokenizer   — VQNSP encoder (representation geometry, attention, CKA, attribution)
#   transformer — Pre-trained backbone (same as tokenizer, no calibration)
#   classifier  — Fine-tuned classification model (all five analyses including calibration)
#
# Edit the CONFIG section for each model type, then run one of:
#   bash run_evaluate_model_structure.sh tokenizer
#   bash run_evaluate_model_structure.sh transformer
#   bash run_evaluate_model_structure.sh classifier
#   bash run_evaluate_model_structure.sh all        # run all three sequentially
#
# Outputs per model (saved to OUTPUT_DIR_*):
#   geom_tsne.png                — t-SNE of CLS embeddings
#   geom_isotropy.png            — singular value concentration
#   attn_head_analysis.png       — per-head entropy + head diversity
#   attn_maps_per_layer.png      — mean attention map per layer
#   cka_layers.png               — pairwise layer CKA matrix
#   calibration.png              — reliability diagram + ECE  [classifier only]
#   attr_grad_input.png          — Gradient × Input saliency
#   attr_attention_rollout.png   — attention rollout by channel and time
#   structure_metrics.json       — all numeric metrics

# ─────────────────────────────────────────────────────────────────────────────
# COMMON CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEVICE="cpu"            # "cuda" or "cpu"
EEG_SIZE=1600           # input EEG length in samples (must match training)
N_SAMPLES=1000          # EEG segments to load for analysis
N_SALIENCY=32           # segments used for gradient/rollout computation

# Dataset — used to infer channel names automatically.
# Choices: IIIC | TUAB | TUEV | SLEEP
# Leave empty and set CH_NAMES manually if your dataset is not listed.
DATASET="IIIC"
CH_NAMES=""             # e.g. "FP1 FP2 F3 F4 ..."  (overrides DATASET)

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TOK_CHECKPOINT="EEGfounder/checkpoints/tokenizer/checkpoint.pth"
TOK_MODEL="vqnsp_encoder_base_decoder_3x200x12"
TOK_N_EMBED=8192
TOK_EMBED_DIM=32
TOK_DATA_PATH=""        # HDF5 file for analysis; leave empty to skip data-dependent analyses
TOK_OUTPUT_DIR="eval_structure_out/tokenizer"

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER (pretrained backbone) CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRF_CHECKPOINT="EEGfounder/checkpoints/eegfounder/checkpoint.pth"
TRF_MODEL="base_patch200_1600_8k_vocab"
TRF_VOCAB_SIZE=8192
TRF_DATA_PATH=""        # HDF5 file for analysis
TRF_OUTPUT_DIR="eval_structure_out/transformer"

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CLS_CHECKPOINT="EEGfounder/checkpoints/finetune/checkpoint_best.pth"
CLS_MODEL="base_patch200_200"
CLS_NB_CLASSES=6        # number of output classes
CLS_IS_BINARY=""        # set to "--is_binary" for binary classification, else leave empty
CLS_DROP=0.0
CLS_DROP_PATH=0.0
CLS_DATA_PATH=""        # HDF5 val file with labels (required for calibration)
CLS_OUTPUT_DIR="eval_structure_out/classifier"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_ch_args() {
    # Emit --dataset or --ch_names argument based on config
    if [ -n "${CH_NAMES}" ]; then
        echo "--ch_names ${CH_NAMES}"
    elif [ -n "${DATASET}" ]; then
        echo "--dataset ${DATASET}"
    fi
}

_run_banner() {
    local model_type="$1"
    local checkpoint="$2"
    local output_dir="$3"
    echo ""
    echo "=========================================================="
    echo "  evaluate_model_structure.py  —  ${model_type}"
    echo "=========================================================="
    echo "  checkpoint : ${checkpoint}"
    echo "  output dir : ${output_dir}"
    echo "  device     : ${DEVICE}"
    echo "----------------------------------------------------------"
}

# ─────────────────────────────────────────────────────────────────────────────
# Run functions
# ─────────────────────────────────────────────────────────────────────────────

run_tokenizer() {
    _run_banner "tokenizer" "${TOK_CHECKPOINT}" "${TOK_OUTPUT_DIR}"

    CMD="python evaluate_model_structure.py \
        --model_type  tokenizer \
        --checkpoint  ${TOK_CHECKPOINT} \
        --model       ${TOK_MODEL} \
        --n_embed     ${TOK_N_EMBED} \
        --embed_dim   ${TOK_EMBED_DIM} \
        --eeg_size    ${EEG_SIZE} \
        --n_samples   ${N_SAMPLES} \
        --n_saliency  ${N_SALIENCY} \
        --output_dir  ${TOK_OUTPUT_DIR} \
        --device      ${DEVICE} \
        $(_ch_args) \
        --skip_calib"

    # calibration is not applicable to tokenizers
    if [ -n "${TOK_DATA_PATH}" ]; then
        CMD="${CMD} --data_path ${TOK_DATA_PATH}"
    fi

    echo "${CMD}"
    echo "=========================================================="
    eval ${CMD}
}

run_transformer() {
    _run_banner "transformer" "${TRF_CHECKPOINT}" "${TRF_OUTPUT_DIR}"

    CMD="python evaluate_model_structure.py \
        --model_type  transformer \
        --checkpoint  ${TRF_CHECKPOINT} \
        --model       ${TRF_MODEL} \
        --vocab_size  ${TRF_VOCAB_SIZE} \
        --eeg_size    ${EEG_SIZE} \
        --n_samples   ${N_SAMPLES} \
        --n_saliency  ${N_SALIENCY} \
        --output_dir  ${TRF_OUTPUT_DIR} \
        --device      ${DEVICE} \
        $(_ch_args) \
        --skip_calib"

    # calibration is not applicable to pretrained transformers
    if [ -n "${TRF_DATA_PATH}" ]; then
        CMD="${CMD} --data_path ${TRF_DATA_PATH}"
    fi

    echo "${CMD}"
    echo "=========================================================="
    eval ${CMD}
}

run_classifier() {
    _run_banner "classifier" "${CLS_CHECKPOINT}" "${CLS_OUTPUT_DIR}"

    CMD="python evaluate_model_structure.py \
        --model_type  classifier \
        --checkpoint  ${CLS_CHECKPOINT} \
        --model       ${CLS_MODEL} \
        --nb_classes  ${CLS_NB_CLASSES} \
        --drop        ${CLS_DROP} \
        --drop_path   ${CLS_DROP_PATH} \
        --eeg_size    ${EEG_SIZE} \
        --n_samples   ${N_SAMPLES} \
        --n_saliency  ${N_SALIENCY} \
        --output_dir  ${CLS_OUTPUT_DIR} \
        --device      ${DEVICE} \
        $(_ch_args) \
        ${CLS_IS_BINARY}"

    if [ -n "${CLS_DATA_PATH}" ]; then
        CMD="${CMD} --data_path ${CLS_DATA_PATH}"
    fi

    echo "${CMD}"
    echo "=========================================================="
    eval ${CMD}
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

TARGET="${1:-all}"

case "${TARGET}" in
    tokenizer)
        run_tokenizer
        ;;
    transformer)
        run_transformer
        ;;
    classifier)
        run_classifier
        ;;
    all)
        run_tokenizer
        run_transformer
        run_classifier
        ;;
    *)
        echo "Usage: bash run_evaluate_model_structure.sh [tokenizer|transformer|classifier|all]"
        echo "  Default: all"
        exit 1
        ;;
esac
