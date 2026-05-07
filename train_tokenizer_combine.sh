#!/bin/bash
# train_tokenizer_combine.sh
# ==========================
# Train / evaluate the multi-tokenizer fusion classifier.
#
# Four fusion strategies available (set FUSION_TYPE below):
#   concat       — full self-attention over all modality tokens (baseline)
#   hierarchical — local per-modality transformer → global CLS fusion
#   cross_attn   — anchor modality (EEG) as Q; other modalities as KV
#   cls_cross    — independent per-modality transformers → fuse K CLS tokens
#
# Run modes:
#   bash train_tokenizer_combine.sh make_cfg          — write tok_cfg.json only
#   bash train_tokenizer_combine.sh train             — train from scratch
#   bash train_tokenizer_combine.sh resume            — resume from last ckpt
#   bash train_tokenizer_combine.sh eval              — evaluate best ckpt
#   bash train_tokenizer_combine.sh train_all         — train all 6 strategies

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# ── Tokenizer checkpoints ──────────────────────────────────────────────────
TOK_CFG="tok_cfg.json"
TOK_EEG_CKPT="EEGfounder/checkpoints/tokenizer_eeg/checkpoint.pth"
TOK_ECG_CKPT="EEGfounder/checkpoints/tokenizer_ecg/checkpoint.pth"
TOK_EMG_CKPT="EEGfounder/checkpoints/tokenizer_emg/checkpoint.pth"
TOK_EOG_CKPT="EEGfounder/checkpoints/tokenizer_eog/checkpoint.pth"

TOK_MODEL="vqnsp_encoder_base_decoder_3x200x12"
TOK_N_EMBED=8192
TOK_CODE_DIM=32
TOK_EEG_SIZE=1600
TOK_PATCH_SIZE=200

# ── Data ──────────────────────────────────────────────────────────────────
# Option A — single file containing all channels
# DATA_TRAIN="data/train.hdf5"
# DATA_VAL="data/val.hdf5"
#
# Option B — one file per modality (concatenated along channel axis)
DATA_TRAIN="data/eeg_train.hdf5 data/ecg_train.hdf5 data/emg_train.hdf5 data/eog_train.hdf5"
DATA_VAL="data/eeg_val.hdf5   data/ecg_val.hdf5   data/emg_val.hdf5   data/eog_val.hdf5"
DATA_KEYS=""        # e.g. "eeg ecg emg eog" — leave empty to auto-detect
LABEL_KEY="label"

# ── Fusion strategy ────────────────────────────────────────────────────────
# concat | hierarchical | cross_attn | cls_cross | gated | perceiver
FUSION_TYPE="concat"

# ── Shared transformer hyper-params ───────────────────────────────────────
FUSION_DIM=256
NUM_HEADS=8
MLP_RATIO=4.0
DROP=0.0
ATTN_DROP=0.0
DROP_PATH=0.1
POOL="cls"              # cls | mean  (used by concat and cross_attn)

# Depth params
DEPTH=6                 # total depth (concat, cross_attn)
DEPTH_LOCAL=3           # per-modality depth (hierarchical, cls_cross, gated)
DEPTH_GLOBAL=3          # global fusion depth (hierarchical, cls_cross, gated)

# Perceiver-specific
N_LATENTS=64            # number of learnable latent query vectors
N_LAYERS=6              # number of (cross-attn + self-attn) rounds

# ── Classification head ────────────────────────────────────────────────────
# linear | mlp | mlp_norm
HEAD_TYPE="mlp"
HEAD_HIDDEN_DIM=""      # leave empty to default to fusion_dim
HEAD_DROPOUT=0.0

# ── Classification ─────────────────────────────────────────────────────────
NB_CLASSES=6
IS_BINARY=""            # set to "--is_binary" for binary classification

# ── Tokenizer fine-tuning ──────────────────────────────────────────────────
UNFREEZE=""             # set to "--unfreeze_tokenizers" to fine-tune tokenizers
TOK_LR_SCALE=0.1

# ── Training ──────────────────────────────────────────────────────────────
EPOCHS=50
BATCH_SIZE=32
LR=1e-3
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=5
LABEL_SMOOTHING=0.1
NUM_WORKERS=4

# ── Output ────────────────────────────────────────────────────────────────
# When running train_all, each strategy saves to its own subdirectory.
OUTPUT_BASE="multi_tok_out"
DEVICE="cuda"
SEED=42

# ─────────────────────────────────────────────────────────────────────────────
# Generate JSON config
# ─────────────────────────────────────────────────────────────────────────────
# Edit channel_indices to match your data layout.
# If you load multiple HDF5 files, column indices follow the file order:
#   file 0 (EEG, 19 ch) → 0-18
#   file 1 (ECG,  1 ch) → 19
#   file 2 (EMG,  2 ch) → 20-21
#   file 3 (EOG,  2 ch) → 22-23

make_cfg() {
    cat > "${TOK_CFG}" <<EOF
{
  "tokenizers": [
    {
      "name": "EEG",
      "checkpoint": "${TOK_EEG_CKPT}",
      "model": "${TOK_MODEL}",
      "n_embed": ${TOK_N_EMBED},
      "code_dim": ${TOK_CODE_DIM},
      "eeg_size": ${TOK_EEG_SIZE},
      "patch_size": ${TOK_PATCH_SIZE},
      "channel_indices": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
      "freeze": true
    },
    {
      "name": "ECG",
      "checkpoint": "${TOK_ECG_CKPT}",
      "model": "${TOK_MODEL}",
      "n_embed": ${TOK_N_EMBED},
      "code_dim": ${TOK_CODE_DIM},
      "eeg_size": ${TOK_EEG_SIZE},
      "patch_size": ${TOK_PATCH_SIZE},
      "channel_indices": [19],
      "freeze": true
    },
    {
      "name": "EMG",
      "checkpoint": "${TOK_EMG_CKPT}",
      "model": "${TOK_MODEL}",
      "n_embed": ${TOK_N_EMBED},
      "code_dim": ${TOK_CODE_DIM},
      "eeg_size": ${TOK_EEG_SIZE},
      "patch_size": ${TOK_PATCH_SIZE},
      "channel_indices": [20,21],
      "freeze": true
    },
    {
      "name": "EOG",
      "checkpoint": "${TOK_EOG_CKPT}",
      "model": "${TOK_MODEL}",
      "n_embed": ${TOK_N_EMBED},
      "code_dim": ${TOK_CODE_DIM},
      "eeg_size": ${TOK_EEG_SIZE},
      "patch_size": ${TOK_PATCH_SIZE},
      "channel_indices": [22,23],
      "freeze": true
    }
  ]
}
EOF
    echo "[train_tokenizer_combine] Wrote ${TOK_CFG}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_data_keys_arg() {
    [ -n "${DATA_KEYS}" ] && echo "--data_keys ${DATA_KEYS}" || echo ""
}

_banner() {
    echo ""
    echo "=========================================================="
    echo "  tokenizer_combine.py  —  ${1}"
    echo "  fusion_type : ${2}"
    echo "  output_dir  : ${3}"
    echo "=========================================================="
}

_head_hidden_arg() {
    [ -n "${HEAD_HIDDEN_DIM}" ] && echo "--head_hidden_dim ${HEAD_HIDDEN_DIM}" || echo ""
}

_common_args() {
    local output_dir="${1}"
    local fusion_type="${2}"
    echo "--tokenizer_cfg   ${TOK_CFG} \
          --eeg_size        ${TOK_EEG_SIZE} \
          --patch_size      ${TOK_PATCH_SIZE} \
          --fusion_type     ${fusion_type} \
          --fusion_dim      ${FUSION_DIM} \
          --num_heads       ${NUM_HEADS} \
          --mlp_ratio       ${MLP_RATIO} \
          --drop            ${DROP} \
          --attn_drop       ${ATTN_DROP} \
          --drop_path       ${DROP_PATH} \
          --pool            ${POOL} \
          --depth           ${DEPTH} \
          --depth_local     ${DEPTH_LOCAL} \
          --depth_global    ${DEPTH_GLOBAL} \
          --n_latents       ${N_LATENTS} \
          --n_layers        ${N_LAYERS} \
          --head_type       ${HEAD_TYPE} \
          --head_dropout    ${HEAD_DROPOUT} \
          --nb_classes      ${NB_CLASSES} \
          --epochs          ${EPOCHS} \
          --batch_size      ${BATCH_SIZE} \
          --lr              ${LR} \
          --weight_decay    ${WEIGHT_DECAY} \
          --warmup_epochs   ${WARMUP_EPOCHS} \
          --label_smoothing ${LABEL_SMOOTHING} \
          --num_workers     ${NUM_WORKERS} \
          --output_dir      ${output_dir} \
          --device          ${DEVICE} \
          --seed            ${SEED} \
          --tok_lr_scale    ${TOK_LR_SCALE} \
          ${IS_BINARY} \
          ${UNFREEZE} \
          $(_data_keys_arg) \
          $(_head_hidden_arg)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Train one fusion type
# ─────────────────────────────────────────────────────────────────────────────

_run_train() {
    local fusion_type="${1}"
    local output_dir="${OUTPUT_BASE}/${fusion_type}"
    _banner "TRAIN" "${fusion_type}" "${output_dir}"

    CMD="python tokenizer_combine.py \
        $(_common_args ${output_dir} ${fusion_type}) \
        --data_train ${DATA_TRAIN} \
        --data_val   ${DATA_VAL} \
        --label_key  ${LABEL_KEY}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

_run_resume() {
    local fusion_type="${1}"
    local output_dir="${OUTPUT_BASE}/${fusion_type}"
    local last="${output_dir}/checkpoint_last.pth"
    if [ ! -f "${last}" ]; then
        echo "[error] No checkpoint at ${last}"; exit 1
    fi
    _banner "RESUME" "${fusion_type}" "${output_dir}"

    CMD="python tokenizer_combine.py \
        $(_common_args ${output_dir} ${fusion_type}) \
        --data_train ${DATA_TRAIN} \
        --data_val   ${DATA_VAL} \
        --label_key  ${LABEL_KEY} \
        --checkpoint ${last}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

_run_eval() {
    local fusion_type="${1}"
    local output_dir="${OUTPUT_BASE}/${fusion_type}"
    local best="${output_dir}/checkpoint_best.pth"
    if [ ! -f "${best}" ]; then
        echo "[error] No best checkpoint at ${best}"; exit 1
    fi
    _banner "EVAL" "${fusion_type}" "${output_dir}"

    CMD="python tokenizer_combine.py \
        --eval \
        $(_common_args ${output_dir} ${fusion_type}) \
        --data_val   ${DATA_VAL} \
        --label_key  ${LABEL_KEY} \
        --checkpoint ${best}"
    echo "${CMD}"; echo "=========================================================="
    eval ${CMD}
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

TARGET="${1:-train}"

case "${TARGET}" in
    make_cfg)
        make_cfg
        ;;
    train)
        make_cfg
        _run_train "${FUSION_TYPE}"
        ;;
    resume)
        _run_resume "${FUSION_TYPE}"
        ;;
    eval)
        _run_eval "${FUSION_TYPE}"
        ;;
    train_all)
        # Train all six strategies sequentially for ablation comparison.
        make_cfg
        for ft in concat hierarchical cross_attn cls_cross gated perceiver; do
            _run_train "${ft}"
        done
        echo ""
        echo "=========================================================="
        echo "  All strategies trained. Evaluating best checkpoints ..."
        echo "=========================================================="
        for ft in concat hierarchical cross_attn cls_cross gated perceiver; do
            _run_eval "${ft}"
        done
        ;;
    eval_all)
        # Evaluate all six strategies.
        for ft in concat hierarchical cross_attn cls_cross gated perceiver; do
            _run_eval "${ft}"
        done
        ;;
    *)
        echo ""
        echo "Usage: bash train_tokenizer_combine.sh [mode]"
        echo ""
        echo "  make_cfg   — write tok_cfg.json from CONFIG section"
        echo "  train      — train with FUSION_TYPE (default: ${FUSION_TYPE})"
        echo "  resume     — resume from checkpoint_last.pth"
        echo "  eval       — evaluate checkpoint_best.pth"
        echo "  train_all  — train all 6 strategies then evaluate each"
        echo "  eval_all   — evaluate all 6 strategies"
        echo ""
        echo "Set FUSION_TYPE in the CONFIG section to choose a strategy."
        exit 1
        ;;
esac
