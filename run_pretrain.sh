#!/bin/bash
# sudo ~/miniconda3/envs/torchenv/bin/python EEGfounder/pretrain_data_provider.py -- make_hdf5

#OMP_NUM_THREADS=1 torchrun --nnodes=1  --nproc_per_node=2 EEGfounder/train_tokenizer.py \
#    --output_dir EEGfounder/checkpoints/tokenizer/ \
#    --log_dir EEGfounder/log/tokenizer/ \
#    --model vqnsp_encoder_base_decoder_3x200x12 \
#    --quantize_kmeans_init \
#    --opt_betas 0.9 0.99 \
#    --batch_size 128 \
#    --epochs 2 \
#    --save_ckpt_freq 1 \
#    --warmup_epochs 1



OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 EEGfounder/pretrain_with_mask.py \
        --output_dir EEGfounder/checkpoints/eegfounder \
        --log_dir EEGfounder/log/eegfounder \
        --model base_patch200_1600_8k_vocab \
        --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
        --tokenizer_weight EEGfounder/checkpoints/tokenizer/checkpoint.pth \
        --batch_size 128 \
        --lr 5e-4 \
        --warmup_epochs 1 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 0.1 \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 2 \
        --save_ckpt_freq 1 \
        --codebook_dim 64 \
        --gradient_accumulation_steps 1