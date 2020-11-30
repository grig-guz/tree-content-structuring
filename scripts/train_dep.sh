#!/bin/bash

python main.py --train --model_type dep_model --dataset_type $1 --num_edus_bound 35 --lr 0.001 --warmup_steps 4000 --dropout 0.0 --batch_size 2 --dep_beam_size 5 --num_heads_tf 8 --num_layers_tf 4 --cuda_id 1
