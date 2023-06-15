#!/bin/bash
export PYTHONPATH=$PWD


python train/train_linear_reg_ldp.py --data_type linear_reg_exp_gaussian \
--warm_start \
--device cpu \
--epsilon 128 \
--linear_reg_p 2000 \
--lr 0.00017793956039292733 \
--model_type linear_reg \
--optimizer semi_ldp \
--priv_unit_g_p 0.987719558951828 \
--public_private_ratio 0.2 \
--seed 42 