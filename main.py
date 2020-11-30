import argparse
import pickle
import os
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = "/scratch/grigorii/data/transfomer_saves/"

from runner import train, evaluate
from dataset.utils.constants import *
from torch.utils.data.dataset import Subset
import torch as th
from dataset.data_helper import DataHelper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')  
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_long_docs', action='store_true')
    parser.add_argument('--model_type', type=str, default=DEP_MODEL)
    parser.add_argument('--dataset_type', type=int, default=2)
    parser.add_argument('--num_edus_bound', type=int, default=35)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--ptr_beam_size', type=int, default=32)
    parser.add_argument('--dep_beam_size', type=int, default=5)
    parser.add_argument('--num_heads_tf', type=int, default=8)
    parser.add_argument('--num_layers_tf', type=int, default=4)
    parser.add_argument('--save_dir', help='name of the folder to store model weights in')
    parser.add_argument('--cuda_id', help='id of the cuda device')
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    
    config = {
        LSTM_HID: 256,
        NUM_LAYERS: args.num_layers_tf,
        NUM_HEADS: args.num_heads_tf,
        DROPOUT: args.dropout,
        LR: args.lr,
        BATCH_SIZE: args.batch_size,
        DEP_BEAM_SIZE: args.dep_beam_size,
        PTR_BEAM_SIZE: args.ptr_beam_size,
        MODEL_TYPE: args.model_type, ###DEP_MODEL,POINTER_MODEL, DEP_BASELINE, LM_BASELINE### 
        DATASET_TYPE: args.dataset_type, # 1 is 100k, 2 is 250k
        NUM_EDUS_BOUND: args.num_edus_bound,
        WARMUP_STEPS: args.warmup_steps,
    }
    config[DEVICE] = th.cuda.current_device()
    #config[DEVICE] = 'cpu'
    config[EVAL] = args.eval
    
    print("WARNING: THEY WILL CHANGE .TRANSPOSE() IN DGL TO HAVE OPPOSITE SEMANTICS SOON")

    if args.prepare:
        data_helper = DataHelper()
        data_helper.create_data_helper(data_dir="../../data")
    elif args.train:
        train(config)
    elif args.eval:
        config[EVAL_LONG_DOCS] = args.eval_long_docs
        evaluate(config, 0)