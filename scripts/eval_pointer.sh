#!/bin/bash
# Evaluate the pointer model on short and long documents

python main.py --eval --model_type pointer_model --dataset_type $1 --num_edus_bound 35 --lr 0.001 --dropout 0.15 --warmup_steps 4000 --batch_size 2 --ptr_beam_size 32 --num_heads_tf 8 --num_layers_tf 4 --cuda_id 0

python main.py --eval --model_type pointer_model --eval_long_docs --dataset_type $1 --num_edus_bound 35 --lr 0.001 --warmup_steps 4000 --dropout 0.15 --batch_size 2 --ptr_beam_size 32 --num_heads_tf 8 --num_layers_tf 4 --cuda_id 0
