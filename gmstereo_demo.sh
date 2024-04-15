#!/usr/bin/env bash


# gmstereo-scale2-regrefine3 model
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--inference_dir demo/sceneflow \
--inference_size 1024 1536 \
--output_path output/gmstereo-scale2-regrefine3-sceneflow \
--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3

# optionally predict both left and right disparities
#--pred_bidir_disp




