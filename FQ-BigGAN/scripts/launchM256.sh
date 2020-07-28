#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1,2
# python train.py \
# --dataset Magic --parallel --shuffle --num_workers 1 --batch_size 256 --load_in_mem \
# --num_G_accumulations 4 --num_D_accumulations 4 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --dim_z 120 \
# --G_eval_mode \--num_workers 1
# --G_ch 64 --D_ch 64 \
# --ema --use_ema --ema_start 20000 \
# --test_every 1000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --discrete_layer 0123 --commitment 15.0 --dict_size 10 --dict_decay 0.8 \
# --use_multiepoch_sampler --name_suffix quant

python3 train.py --shuffle --batch_size 16 --num_workers 8 --parallel \
--load_in_mem \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 1 --G_lr 2e-4 \
--D_lr 2e-4 --dataset Magic --G_ortho 0.0 \
--G_attn 16 --D_attn 16 --G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 2000 --save_every 250 \
--num_best_copies 5 --num_save_copies 2 --seed 0 \
--discrete_layer 0123 --commitment 1.0 --dict_size 10 --dict_decay 0.8 \
--name_suffix quant
