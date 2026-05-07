#!/bin/bash
# run_model_scaling.sh
# ====================
# Expand a trained EEG model to a larger scale.
#
# Three methods, three model types.  Edit the CONFIG sections below, then run:
#
#   bash run_model_scaling.sh tokenizer  A   # depth expansion
#   bash run_model_scaling.sh tokenizer  B   # width expansion
#   bash run_model_scaling.sh tokenizer  C   # weight inheritance
#
#   bash run_model_scaling.sh transformer A
#   bash run_model_scaling.sh transformer B
#   bash run_model_scaling.sh transformer C
#
#   bash run_model_scaling.sh classifier  A
#   bash run_model_scaling.sh classifier  B
#   bash run_model_scaling.sh classifier  C
#
# Method summary
# ──────────────
#   A  depth expansion  — insert near-identity blocks (gamma ≈ 0) between
#                         existing blocks; function-preserving at t=0
#   B  width expansion  — double embed_dim by duplicating attention heads;
#                         function-preserving (Net2WiderNet style)
#   C  weight inherit   — create larger architecture, copy matching weights,
#                         leave new parameters randomly initialised

# ─────────────────────────────────────────────────────────────────────────────
# COMMON
# ─────────────────────────────────────────────────────────────────────────────
DEVICE="cpu"          # "cuda" or "cpu"
EEG_SIZE=1600

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
TOK_CHECKPOINT="EEGfounder/checkpoints/tokenizer/checkpoint.pth"
TOK_MODEL="vqnsp_encoder_base_decoder_3x200x12"
TOK_N_EMBED=8192
TOK_EMBED_DIM=32

# Method A: how deep should the encoder become?
TOK_A_NEW_DEPTH=24
TOK_A_OUTPUT="scaled_models/tokenizer_depthA.pth"

# Method B: embed_dim is doubled automatically (200 → 400)
TOK_B_OUTPUT="scaled_models/tokenizer_widthB.pth"

# Method C: target architecture name + output path
TOK_C_TARGET_MODEL="vqnsp_encoder_large_decoder_3x200x24"
TOK_C_OUTPUT="scaled_models/tokenizer_inheritC.pth"

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER (pretrained backbone)
# ─────────────────────────────────────────────────────────────────────────────
TRF_CHECKPOINT="EEGfounder/checkpoints/eegfounder/checkpoint.pth"
TRF_MODEL="base_patch200_1600_8k_vocab"
TRF_VOCAB_SIZE=8192

TRF_A_NEW_DEPTH=24
TRF_A_OUTPUT="scaled_models/transformer_depthA.pth"

TRF_B_OUTPUT="scaled_models/transformer_widthB.pth"

TRF_C_TARGET_MODEL="large_patch200_1600_8k_vocab"
TRF_C_OUTPUT="scaled_models/transformer_inheritC.pth"

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER (fine-tuned)
# ─────────────────────────────────────────────────────────────────────────────
CLS_CHECKPOINT="EEGfounder/checkpoints/finetune/checkpoint_best.pth"
CLS_MODEL="base_patch200_200"
CLS_NB_CLASSES=6
CLS_DROP=0.0
CLS_DROP_PATH=0.0

CLS_A_NEW_DEPTH=24
CLS_A_OUTPUT="scaled_models/classifier_depthA.pth"

CLS_B_OUTPUT="scaled_models/classifier_widthB.pth"

CLS_C_TARGET_MODEL="large_patch200_200"
CLS_C_OUTPUT="scaled_models/classifier_inheritC.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_banner() {
    echo ""
    echo "=========================================================="
    echo "  model_scaling.py  —  ${1}  /  Method ${2}"
    echo "=========================================================="
}

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

run_tokenizer_A() {
    _banner "tokenizer" "A (depth expansion)"
    CMD="python model_scaling.py \
        --model_type  tokenizer \
        --method      A \
        --checkpoint  ${TOK_CHECKPOINT} \
        --model       ${TOK_MODEL} \
        --n_embed     ${TOK_N_EMBED} \
        --embed_dim   ${TOK_EMBED_DIM} \
        --eeg_size    ${EEG_SIZE} \
        --new_depth   ${TOK_A_NEW_DEPTH} \
        --output      ${TOK_A_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

run_tokenizer_B() {
    _banner "tokenizer" "B (width expansion)"
    CMD="python model_scaling.py \
        --model_type  tokenizer \
        --method      B \
        --checkpoint  ${TOK_CHECKPOINT} \
        --model       ${TOK_MODEL} \
        --n_embed     ${TOK_N_EMBED} \
        --embed_dim   ${TOK_EMBED_DIM} \
        --eeg_size    ${EEG_SIZE} \
        --output      ${TOK_B_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

run_tokenizer_C() {
    _banner "tokenizer" "C (weight inheritance)"
    CMD="python model_scaling.py \
        --model_type  tokenizer \
        --method      C \
        --checkpoint  ${TOK_CHECKPOINT} \
        --model       ${TOK_MODEL} \
        --target_model ${TOK_C_TARGET_MODEL} \
        --n_embed     ${TOK_N_EMBED} \
        --embed_dim   ${TOK_EMBED_DIM} \
        --eeg_size    ${EEG_SIZE} \
        --output      ${TOK_C_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

# ─────────────────────────────────────────────────────────────────────────────
# Transformer
# ─────────────────────────────────────────────────────────────────────────────

run_transformer_A() {
    _banner "transformer" "A (depth expansion)"
    CMD="python model_scaling.py \
        --model_type  transformer \
        --method      A \
        --checkpoint  ${TRF_CHECKPOINT} \
        --model       ${TRF_MODEL} \
        --vocab_size  ${TRF_VOCAB_SIZE} \
        --new_depth   ${TRF_A_NEW_DEPTH} \
        --output      ${TRF_A_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

run_transformer_B() {
    _banner "transformer" "B (width expansion)"
    CMD="python model_scaling.py \
        --model_type  transformer \
        --method      B \
        --checkpoint  ${TRF_CHECKPOINT} \
        --model       ${TRF_MODEL} \
        --vocab_size  ${TRF_VOCAB_SIZE} \
        --output      ${TRF_B_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

run_transformer_C() {
    _banner "transformer" "C (weight inheritance)"
    CMD="python model_scaling.py \
        --model_type  transformer \
        --method      C \
        --checkpoint  ${TRF_CHECKPOINT} \
        --model       ${TRF_MODEL} \
        --target_model ${TRF_C_TARGET_MODEL} \
        --vocab_size  ${TRF_VOCAB_SIZE} \
        --output      ${TRF_C_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

run_classifier_A() {
    _banner "classifier" "A (depth expansion)"
    CMD="python model_scaling.py \
        --model_type  classifier \
        --method      A \
        --checkpoint  ${CLS_CHECKPOINT} \
        --model       ${CLS_MODEL} \
        --nb_classes  ${CLS_NB_CLASSES} \
        --drop        ${CLS_DROP} \
        --drop_path   ${CLS_DROP_PATH} \
        --new_depth   ${CLS_A_NEW_DEPTH} \
        --output      ${CLS_A_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

run_classifier_B() {
    _banner "classifier" "B (width expansion)"
    CMD="python model_scaling.py \
        --model_type  classifier \
        --method      B \
        --checkpoint  ${CLS_CHECKPOINT} \
        --model       ${CLS_MODEL} \
        --nb_classes  ${CLS_NB_CLASSES} \
        --drop        ${CLS_DROP} \
        --drop_path   ${CLS_DROP_PATH} \
        --output      ${CLS_B_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

run_classifier_C() {
    _banner "classifier" "C (weight inheritance)"
    CMD="python model_scaling.py \
        --model_type  classifier \
        --method      C \
        --checkpoint  ${CLS_CHECKPOINT} \
        --model       ${CLS_MODEL} \
        --target_model ${CLS_C_TARGET_MODEL} \
        --nb_classes  ${CLS_NB_CLASSES} \
        --drop        ${CLS_DROP} \
        --drop_path   ${CLS_DROP_PATH} \
        --output      ${CLS_C_OUTPUT} \
        --device      ${DEVICE}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

MODEL_TYPE="${1:-}"
METHOD="${2:-}"

case "${MODEL_TYPE}_${METHOD}" in
    tokenizer_A)    run_tokenizer_A    ;;
    tokenizer_B)    run_tokenizer_B    ;;
    tokenizer_C)    run_tokenizer_C    ;;
    transformer_A)  run_transformer_A  ;;
    transformer_B)  run_transformer_B  ;;
    transformer_C)  run_transformer_C  ;;
    classifier_A)   run_classifier_A   ;;
    classifier_B)   run_classifier_B   ;;
    classifier_C)   run_classifier_C   ;;
    *)
        echo ""
        echo "Usage: bash run_model_scaling.sh <model_type> <method>"
        echo ""
        echo "  model_type : tokenizer | transformer | classifier"
        echo "  method     : A (depth) | B (width) | C (weight-inherit)"
        echo ""
        echo "Examples:"
        echo "  bash run_model_scaling.sh tokenizer  A"
        echo "  bash run_model_scaling.sh transformer C"
        echo "  bash run_model_scaling.sh classifier  B"
        exit 1
        ;;
esac
