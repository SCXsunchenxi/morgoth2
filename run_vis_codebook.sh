#!/bin/bash
# run_vis_codebook.sh
# ===================
# Visualise the learned VQ-NSP codebook from a trained tokenizer checkpoint.
#
# Edit the variables in the CONFIG section, then run:
#   bash run_vis_codebook.sh
#
# Outputs (saved to OUTPUT_DIR):
#   codebook_embedding.png       — t-SNE / UMAP projection coloured by usage
#   codebook_usage.png           — usage distribution + cumulative coverage
#   codebook_similarity.png      — pairwise cosine-similarity heatmap
#   codebook_spectra.png         — decoded spectral signatures of top-k codes
#   codebook_topo.png            — dominant code per channel  (needs DATA_PATH)
#   codebook_channel_entropy.png — token entropy per channel  (needs DATA_PATH)
#   codebook_channel_vocab_sim.png — channel vocabulary similarity (needs DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths before running
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT="EEGfounder/checkpoints/tokenizer/checkpoint.pth"

# Model architecture — must match the checkpoint
MODEL="vqnsp_encoder_base_decoder_3x200x12"
N_EMBED=8192        # codebook size K
EMBED_DIM=32        # code dimensionality D
EEG_SIZE=1600       # input EEG length in samples

# Output directory
OUTPUT_DIR="vis_codebook_out"

# Device: "cuda" or "cpu"
DEVICE="cpu"

# ── Panel 1 ──────────────────────────────────────────────────────────────────
# Set to "--umap" to use UMAP instead of t-SNE (requires: pip install umap-learn)
EMBEDDING_METHOD=""   # or "--umap"

# ── Panel 3 ──────────────────────────────────────────────────────────────────
SIM_MAX_SHOW=256      # max number of codes shown in similarity heatmap

# ── Panel 4 ──────────────────────────────────────────────────────────────────
SPECTRA_TOP_K=32      # how many top-used codes to decode

# ── Panel 5 (token spatial assignment) ───────────────────────────────────────
# Leave DATA_PATH empty to skip panel 5
DATA_PATH=""          # e.g. "/data/eeg/dataset.hdf5"
N_DATA_SAMPLES=2048   # number of EEG segments to encode

# Optional: space-separated list of channel names for axis labels
# Leave empty to use numeric indices
CH_NAMES=""           # e.g. "Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 ..."

# ─────────────────────────────────────────────────────────────────────────────
# Build the command
# ─────────────────────────────────────────────────────────────────────────────

CMD="python vis_codebook.py \
    --checkpoint  ${CHECKPOINT} \
    --model       ${MODEL} \
    --n_embed     ${N_EMBED} \
    --embed_dim   ${EMBED_DIM} \
    --eeg_size    ${EEG_SIZE} \
    --output_dir  ${OUTPUT_DIR} \
    --device      ${DEVICE} \
    --sim_max_show    ${SIM_MAX_SHOW} \
    --spectra_top_k   ${SPECTRA_TOP_K} \
    --n_data_samples  ${N_DATA_SAMPLES} \
    ${EMBEDDING_METHOD}"

# Add data path for panel 5 if provided
if [ -n "${DATA_PATH}" ]; then
    CMD="${CMD} --data_path ${DATA_PATH}"
fi

# Add channel names if provided
if [ -n "${CH_NAMES}" ]; then
    CMD="${CMD} --ch_names ${CH_NAMES}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo " vis_codebook.py"
echo "=========================================="
echo "  checkpoint : ${CHECKPOINT}"
echo "  model      : ${MODEL}  (K=${N_EMBED}, D=${EMBED_DIM})"
echo "  output dir : ${OUTPUT_DIR}"
echo "  device     : ${DEVICE}"
[ -n "${DATA_PATH}" ] && echo "  data path  : ${DATA_PATH} (${N_DATA_SAMPLES} samples)"
echo "------------------------------------------"
echo "${CMD}"
echo "=========================================="

eval ${CMD}
