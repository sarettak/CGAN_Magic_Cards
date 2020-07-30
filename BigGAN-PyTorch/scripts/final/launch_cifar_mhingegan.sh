#!/bin/bash
python train.py \
--shuffle --batch_size 25 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset Magic \
--G_ortho 0.0 \
--load_in_mem \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 100 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root data \
--weights_root logs \
--logs_root logs \
--samples_root samples \
--test_every -1 \
--experiment_name mhs_p05 \
--historical_save_every 1000 \
--mh_loss \
--mh_loss_weight 0.05 \
--model=BigGANmh
#--resume
#--load_weights 022500 \


# --data_root /scratch0/ilya/locDoc/data/cifar10 \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \