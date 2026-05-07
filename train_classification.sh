#!/bin/bash

# TUAB: abnormal classification -------------------------------------------------
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune_tuab_base2 \
#        --log_dir log/finetune_tuab_base2 \
#        --model base_patch200_200 \
#        --finetune pretrained_model/labram-base.pth \
#        --training_data_dir data/tuh_eeg/TUAB/edf/processed/test \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --epochs 30 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --dataset TUAB \
#        --disable_qkv_bias \
#        --seed 0
# TUAB: abnormal classification -------------------------------------------------


# TUEV: event classification -------------------------------------------------
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune_tuev_base \
#        --log_dir log/finetune_tuev_base2 \
#        --model base_patch200_200 \
#        --finetune pretrained_model/labram-base.pth \
#        --training_data_dir /data/tuh_eeg/TUEV/edf/processed/test \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --epochs 30 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --dataset TUEV \
#        --disable_qkv_bias \
#        --seed 0
# TUEV: event classification -------------------------------------------------


#TUEP: epilepsy classification -------------------------------------------------
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune_tuep_base \
#        --log_dir log/finetune_tuep_base2 \
#        --model base_patch200_200 \
#        --finetune pretrained_model/labram-base.pth \
#        --training_data_dir /data/tuh_eeg/TUEP/processed/test \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --epochs 30 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --dataset TUEP \
#        --disable_qkv_bias \
#        --seed 0
#TUEP: epilepsy classification -------------------------------------------------





# NORMAL ----------------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#          --output_dir checkpoints/finetune6_NORMAL2 \
#          --log_dir log/finetune6_NORMAL2 \
#          --dataset NORMAL \
#          --training_data_dir /data/NORMAL/processed_10second \
#          --model base_patch200_200 \
#          --epochs ${EPOCH} \
#          --finetune pretrained_model/base6.pth \
#          --focalloss \
#          --focal_gamma 2 \
#          --focal_alpha "0.5"\
#          --weight_decay 0.05 \
#          --batch_size 64 \
#          --lr 5e-4 \
#          --update_freq 1 \
#          --warmup_epochs 3 \
#          --layer_decay 0.65 \
#          --drop_path 0.1 \
#          --dist_eval \
#          --save_ckpt_freq 1 \
#          --disable_rel_pos_bias \
#          --abs_pos_emb \
#          --disable_qkv_bias \
#          --seed 0
# NORMAL ----------------------------------------------------------------------


# SLOWING ----------------------------------------------------------------------
#EPOCH=5
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#          --output_dir checkpoints/finetune6_SLOWING2 \
#          --log_dir log/finetune6_SLOWING2 \
#          --dataset SLOWING \
#          --training_data_dir /data/SLOWING/processed_10second \
#          --focalloss \
#          --focal_gamma 2 \
#          --focal_alpha "0.25 0.35 0.3" \
#          --model base_patch200_200 \
#          --epochs ${EPOCH} \
#          --finetune pretrained_model/base6.pth \
#          --weight_decay 0.05 \
#          --batch_size 64 \
#          --lr 5e-4 \
#          --update_freq 1 \
#          --warmup_epochs 3 \
#          --layer_decay 0.65 \
#          --drop_path 0.1 \
#          --dist_eval \
#          --save_ckpt_freq 1 \
#          --disable_rel_pos_bias \
#          --abs_pos_emb \
#          --disable_qkv_bias \
#          --seed 0
# SLOWING ----------------------------------------------------------------------

# BS ----------------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune6_BS2 \
#        --log_dir log/finetune6_BS2 \
#        --dataset BS \
#        --training_data_dir /data/BS/processed_10second \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.5" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
# BS ----------------------------------------------------------------------


# IIIC  ----------------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune6_IIIC2 \
#        --log_dir log/finetune6_IIIC2 \
#        --dataset IIIC \
#        --training_data_dir /data/IIIC/processed_10second \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.2 0.25 0.25 0.25 0.25 0.25" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# IIIC ----------------------------------------------------------------------




# FOC GEN SPIKES 3 class--------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune6_FOCGENSPIKES2 \
#        --log_dir log/finetune6_FOCGENSPIKES2 \
#        --dataset FOC_GEN_SPIKES \
#        --training_data_dir /data/FOC_GEN_SPIKES/processed_10second \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.25 0.25 0.25" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# FOC GEN SPIKES 3 class--------------------------------------------------------------


# SPIKES -----------------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune6_SPIKE \
#        --log_dir log/finetune6_SPIKE \
#        --dataset SPIKES \
#        --model base_patch200_200 \
#        --training_data_dir /data/SPIKES/processed_1second \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.5" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# SPIKES -----------------------------------------------------------------------





 # MGB 3 stages SLEEP----------------------------------------------------------------------
# EPOCH=10
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune6_MGBSLEEP3stages \
#         --log_dir log/finetune6_MGBSLEEP3stages \
#         --dataset MGBSLEEP3stages \
#         --training_data_dir /data/MGB_SLEEP/processed_10sec \
#         --model base_patch200_200 \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/base6.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.25 0.5 0.25" \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0
 # MGB 3 stages SLEEP ----------------------------------------------------------------------



# SLEEPPSG  ----------------------------------------------------------------------
# EPOCH=10
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune6_SLEEPPSG \
#         --log_dir log/finetune6_SLEEPPSG \
#         --dataset SLEEPPSG \
#         --training_data_dir /data/SLEEP_PSG \
#         --model base_patch200_200 \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/base6.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.5 0.5 0.5 0.5 0.5" \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0


# EPOCH=30
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune_BCHPSG_scaling_focal \
#         --log_dir log/finetune_BCHPSG_scaling_focal \
#         --dataset SLEEPPSG \
#         --training_data_dir /data/SLEEP_PSG/BCH_processed_10sec \
#         --model base_patch200_200 \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/labram-base.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.5 0.5 0.5 0.5 0.5" \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0
#--training_data_dir /data/SLEEP_PSG/processed_30sec \

 # SLEEPPSG ----------------------------------------------------------------------





 # MASS SLEEP----------------------------------------------------------------------
# EPOCH=30
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune3_MASS \
#         --log_dir log/finetune3_MASS \
#         --dataset SLEEPMASS \
#         --training_data_dir /data/MASS/processed_10sec \
#         --model base_patch200_200 \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/base3.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.5 0.5 0.5 0.5 0.5" \
#         --weight_decay 0.05 \
#         --batch_size 128 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0
 # MASS SLEEP ----------------------------------------------------------------------


 # PENN SLEEP----------------------------------------------------------------------
# EPOCH=10
# echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#         --output_dir checkpoints/finetune3_PENN_nofinetune \
#         --log_dir log/finetune3_PENN_nofinetune \
#         --dataset SLEEPPENN \
#         --training_data_dir /data/PENN/processed_10second \
#         --model base_patch200_200 \
#         --epochs ${EPOCH} \
#         --finetune pretrained_model/base3.pth \
#         --focalloss \
#         --focal_gamma 2 \
#         --focal_alpha "0.25 0.5 0.25 0.25 0.4" \
#         --weight_decay 0.05 \
#         --batch_size 64 \
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 3 \
#         --layer_decay 0.65 \
#         --drop_path 0.1 \
#         --dist_eval \
#         --save_ckpt_freq 1 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --disable_qkv_bias \
#         --seed 0
 # PENN SLEEP----------------------------------------------------------------------







# FOC SPIKES --------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune3_FOCSPIKES \
#        --log_dir log/finetune3_FOCSPIKES \
#        --dataset FOC_SPIKES \
#        --training_data_dir /data/FOC_GEN_SPIKES/FOC_NO/processed_10second \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base3.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.6" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
#
#echo "exxact@1" | sudo mkdir checkpoints/finetune3_FOCSPIKES_Occasion
#echo "exxact@1" | sudo cp checkpoints/finetune3_FOCSPIKES/checkpoint-best.pth checkpoints/finetune3_FOCSPIKES_Occasion/checkpoint-0.pth
#EPOCH=3
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune3_FOCSPIKES_Occasion \
#        --log_dir log/finetune3_FOCSPIKES_Occasion \
#        --dataset GEN_SPIKES \
#        --training_data_dir /data/OccasionNoise/proccessed_10second/fs_train \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base3.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.6" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# FOC SPIKES--------------------------------------------------------------


# GEN SPIKES --------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune3_GENSPIKES \
#        --log_dir log/finetune3_GENSPIKES \
#        --dataset GEN_SPIKES \
#        --training_data_dir /data/FOC_GEN_SPIKES/GEN_NO/processed_10second \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base3.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.6" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

#echo "exxact@1" | sudo mkdir /home/exx/Documents/EEG_report/EEGfounder/checkpoints/finetune3_GENSPIKES_Occasion
#echo "exxact@1" | sudo cp /home/exx/Documents/EEG_report/EEGfounder/checkpoints/finetune3_GENSPIKES/checkpoint-best.pth /home/exx/Documents/EEG_report/EEGfounder/checkpoints/finetune3_GENSPIKES_Occasion/checkpoint-0.pth
#EPOCH=3
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune3_GENSPIKES_Occasion \
#        --log_dir log/finetune3_GENSPIKES_Occasion \
#        --dataset GEN_SPIKES \
#        --training_data_dir /data/OccasionNoise/proccessed_10second/gs_train \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base3.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.6" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
# GEN SPIKES--------------------------------------------------------------



# VW--------------------------------------------------------------
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_VW_resampled_1 \
#        --log_dir log/finetune_VW_resampled_1 \
#        --dataset VW \
#        --model base_patch200_200 \
#        --training_data_dir /data/VW/processed_1second \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.5" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0


#EPOCH=20
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_VW_onlyResampled_2 \
#        --log_dir log/finetune_VW_onlyResampled_2 \
#        --dataset VW \
#        --model base_patch200_200 \
#        --training_data_dir /data/VW/processed_1second \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.5" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0


#EPOCH=20
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12345 finetune_classification.py \
#        --output_dir checkpoints/finetune_VW_SPIKES \
#        --log_dir log/finetune_VW_SPIKES \
#        --dataset VW_SPIKES \
#        --training_data_dir /data/VW_SPIKES/processed_1second \
#        --model base_patch200_200 \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.25 0.5 0.25" \
#        --weight_decay 0.05 \
#        --batch_size 128 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# VW--------------------------------------------------------------



#
## BIPD--------------------------------------------------------------
#EPOCH=5
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_BIPD \
#        --log_dir log/finetune_BIPD \
#        --dataset BIPD \
#        --model base_patch200_200 \
#        --training_data_dir /data/BIPD/processed_10second \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.75" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
## BIPD--------------------------------------------------------------
#
## BIRD--------------------------------------------------------------
#EPOCH=5
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_BIRD \
#        --log_dir log/finetune_BIRD \
#        --dataset BIRD \
#        --model base_patch200_200 \
#        --training_data_dir /data/BIRD/processed_10second \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.75" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
#
## BIRD--------------------------------------------------------------


# Partial PD--------------------------------------------------------------
#EPOCH=20
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_PD3 \
#        --log_dir log/finetune_PD3 \
#        --dataset PD \
#        --model base_patch200_200 \
#        --training_data_dir /data/partial_PD/processed_10second \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.75" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 3 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# Partial PD--------------------------------------------------------------

# Spike localization --------------------------------------------------------------

#
#EPOCH=10
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_SPIKE_localization3 \
#        --log_dir log/finetune_SPIKE_localization3 \
#        --dataset SPIKE_localization \
#        --model base_patch200_200 \
#        --training_data_dir /data/SPIKE_localization/processed_1second \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --multilabel_focalloss \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 1 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

#--exchange_positive_channel \
#--exchange_channel \

#  Spike localization--------------------------------------------------------------



# Spike 1 channel --------------------------------------------------------------
# finetune_SPIKE_1channel's checkpoint-0 is copied from morgoth1's spike results

#
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_SPIKE_1channel_2 \
#        --log_dir log/finetune_SPIKE_1channel_2 \
#        --dataset SPIKE_1channel \
#        --train_spike_1channel_idx -1 \
#        --model base_patch200_200 \
#        --training_data_dir /data/SPIKE_localization/processed_1second \
#        --epochs 20 \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.75" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 1 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0
#
#
#for ch_idx in {0..18}; do
#  echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#          --output_dir checkpoints/finetune_SPIKE_1channel_2 \
#          --log_dir log/finetune_SPIKE_1channel_2 \
#          --dataset SPIKE_1channel \
#          --train_spike_1channel_idx ${ch_idx} \
#          --model base_patch200_200 \
#          --training_data_dir /data/SPIKE_localization/processed_1second \
#          --epochs 3 \
#          --finetune pretrained_model/base6.pth \
#          --focalloss \
#          --focal_gamma 2 \
#          --focal_alpha "0.75" \
#          --weight_decay 0.05 \
#          --batch_size 64 \
#          --lr 5e-4 \
#          --update_freq 1 \
#          --warmup_epochs 1 \
#          --layer_decay 0.65 \
#          --drop_path 0.1 \
#          --dist_eval \
#          --save_ckpt_freq 1 \
#          --disable_rel_pos_bias \
#          --abs_pos_emb \
#          --disable_qkv_bias \
#          --seed 0
#done


#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_SPIKE_1channel_2 \
#        --log_dir log/finetune_SPIKE_1channel_2 \
#        --dataset SPIKE_1channel \
#        --train_spike_1channel_idx -1 \
#        --model base_patch200_200 \
#        --training_data_dir /data/SPIKE_localization/processed_1second \
#        --epochs 20 \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.75" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 1 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

#  Spike 1 channel--------------------------------------------------------------



#  Sleep staging --------------------------------------------------------------

EPOCH=20
echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12346 finetune_classification.py \
       --output_dir checkpoints/hm_SLEEPSTAGING_MGH_6class \
       --log_dir log/hm_SLEEPSTAGING_MGH_6class \
       --dataset SLEEPPSG_6class  \
       --training_data_dir /data/MGH_PSG/processed_10sec \
       --model base_patch200_200 \
       --epochs ${EPOCH} \
       --finetune pretrained_model/base6.pth \
       --focalloss \
       --focal_gamma 2 \
       --focal_alpha "0.25 0.25 0.25 0.25 0.25 0.25" \
       --weight_decay 0.05 \
       --batch_size 64 \
       --lr 5e-4 \
       --update_freq 1 \
       --warmup_epochs 1 \
       --layer_decay 0.65 \
       --drop_path 0.1 \
       --dist_eval \
       --save_ckpt_freq 1 \
       --disable_rel_pos_bias \
       --abs_pos_emb \
       --disable_qkv_bias \
       --seed 0

# Sleep staging --------------------------------------------------------------


#  Arousal --------------------------------------------------------------

#EPOCH=20
#echo "exxact@1" | sudo -S OMP_NUM_THREADS=1 ~/miniconda3/envs/torchenv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=1 finetune_classification.py \
#        --output_dir checkpoints/finetune_AROUSAL \
#        --log_dir log/finetune_AROUSAL \
#        --dataset SLEEP_AROUSAL \
#        --model base_patch200_200 \
#        --training_data_dir /data/MGH_PSG/processed_10sec_arousal \
#        --epochs ${EPOCH} \
#        --finetune pretrained_model/base6.pth \
#        --focalloss \
#        --focal_gamma 2 \
#        --focal_alpha "0.5" \
#        --weight_decay 0.05 \
#        --batch_size 64 \
#        --lr 5e-4 \
#        --update_freq 1 \
#        --warmup_epochs 1 \
#        --layer_decay 0.65 \
#        --drop_path 0.1 \
#        --dist_eval \
#        --save_ckpt_freq 1 \
#        --disable_rel_pos_bias \
#        --abs_pos_emb \
#        --disable_qkv_bias \
#        --seed 0

# Arousal --------------------------------------------------------------