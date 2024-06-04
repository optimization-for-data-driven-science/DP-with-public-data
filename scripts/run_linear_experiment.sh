#!/bin/bash
export PYTHONPATH=$PWD


python train/train_linear_reg.py  --model_type linear_reg \
--warm_start \
--seed 42 \
--linear_reg_p 2000 \
--public_private_ratio 0.03 \
--data_type linear_reg_exp_gaussian \
--data_n_worker 0 \
--device cpu \
--max_grad_norm 1.0 \
--epsilon 1 \
--delta 1e-5 \
--private_epochs 59 \
--optimizer semi_dp \
--semi_dp_beta 0.9425804717040164 \
--lr 2.645031905740823

