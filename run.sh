#!/usr/bin/env bash

# python main_stereo.py \
# --eval \
# --val_dataset sintel \
# --batch_size 4 \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_type self_swin2d_cross_swin1d \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1

# python main_stereo.py \
# --no_resume_optimizer \
# --stage sceneflow \
# --batch_size 16 \
# --val_dataset sintel \
# --img_height 384 \
# --img_width 768 \
# --padding_factor 16 \
# --upsample_factor 8 \
# --attn_type self_swin2d_cross_1d \
# --summary_freq 1000 \
# --val_freq 10000 \
# --save_ckpt_freq 1000 \
# --save_latest_ckpt_freq 1000 \
# --num_steps 100000 \
# 2>&1 | tee -a ./train.log


#!/usr/bin/env bash

# GMFlow with hierarchical matching refinement (1/8 + 1/4 features)

# number of gpus for training, please set according to your hardware
# trained on 8x 40GB A100 gpus
# NUM_GPUS=8

# sceneflow
# resume flow things model
CHECKPOINT_DIR=./VisionMambaKittiPretrain && \
mkdir -p ${CHECKPOINT_DIR} && \
python main_stereo.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--no_resume_optimizer \
--stage kitti15mix \
--batch_size 16 \
--val_dataset kitti15 \
--img_height 384 \
--img_width 768 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--summary_freq 100 \
--val_freq 100 \
--save_ckpt_freq 100 \
--save_latest_ckpt_freq 100 \
--num_steps 100 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log





