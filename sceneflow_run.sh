# sceneflow
# resume gmstereo scale2 model, which is trained from flow things model
CHECKPOINT_DIR=./VisionMamba_sceneflow&& \
mkdir -p ${CHECKPOINT_DIR} && \
python main_stereo.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume VisionMambaKittiPretrainLoadv2/step_020000.pth \
--no_resume_optimizer \
--stage sceneflow \
--lr 5e-5 \
--batch_size 4 \
--num_workers 8 \
--val_dataset things kitti15 \
--img_height 384 \
--img_width 768 \
--padding_factor 32 \
--upsample_factor 8 \
--attn_type self_swin2d_cross_swin1d \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 1000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log