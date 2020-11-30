#!/bin/bash

python -u main.py --train --model_type tree_train_baseline --dataset_type $1 --num_edus_bound 35 --lr 0.001 --warmup_steps 4000 --dropout 0.0 --batch_size 2 --ptr_beam_size 32 --num_heads_tf 8 --num_layers_tf 4 --cuda_id 1
