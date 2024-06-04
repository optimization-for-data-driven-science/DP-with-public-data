#!/bin/bash
export PYTHONPATH=$PWD


python train/train_cifar_cvx_iterations.py  --model_type cifar_10_wide_resnet \
--seed 42 \
--public_private_ratio 0.99 \
--pub_batch_size 256 \
--batch_size 256 \
--data_type cifar10 \
--data_n_worker 0 \
--device cuda \
--max_grad_norm 1.0 \
--epsilon 8 \
--delta 1e-5 \
--iterations=4000 \
--optimizer deep_mind \
--weight_decay 0 \
--lr 1



# --semi_dp_beta 0.4 \