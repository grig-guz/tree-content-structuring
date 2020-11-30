import dgl
import torch as th
from torch.nn.utils import clip_grad_norm_ 
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset.utils.constants import *
from dataset.data_helper import DataHelper
from dataset.constituency_tree import ConstituencyTree
from dataset.datasets import DependencyDataset, BucketBatchSampler
from models.dependency_tree_model import DependencyTreeModel, build_trees_graph
from models.lm_decoding import LMDecodingModel
from models.pointer_net_model import PointerNetModel
from models.dependency_baseline import DependencyBaselineModel
from models.dependency_treetrain_baseline import DependencyTreeTrainBaseline
from eval_funcs import *
from models.modules.optimizer import Optimizer
import gc
import numpy as np
import os
import nltk
from tqdm import tqdm


def construct_datasets(data_helper_train, data_helper_val, config):
    if config[EVAL]:
        if config[EVAL_LONG_DOCS]:
            data_helper_val.constituency_trees = [tree for tree in data_helper_val.constituency_trees 
                                                  if len(tree.doc.edu_dict) > 35]
        dataset = DependencyDataset(data_helper_val.constituency_trees, config)
        data_loader = DataLoader(dataset, 
                              shuffle=False,
                              collate_fn=lambda x: x[0])
        return data_loader
    else:
        data_helper_train.constituency_trees = [tree for tree in data_helper_train.constituency_trees 
                                                  if len(tree.doc.edu_dict) <= config[NUM_EDUS_BOUND]]
        
        train_dataset = DependencyDataset(data_helper_train.constituency_trees, config)
        val_dataset = DependencyDataset(data_helper_val.constituency_trees, config)
        bucket_batch_sampler = BucketBatchSampler(data_helper_train.constituency_trees, 
                                                  config[BATCH_SIZE])
        
        train_data_loader = DataLoader(train_dataset,
                                       batch_sampler=bucket_batch_sampler,
                                       batch_size=1, 
                                       shuffle=False,
                                       collate_fn=lambda x: collate_graphs(x, config),
                                       drop_last=False
                                      )
        
        test_data_loader = DataLoader(val_dataset, 
                                      shuffle=False,
                                      collate_fn=lambda x: x[0])
        
        return train_data_loader, test_data_loader

def construct_model(config):
    model_name = ""
    if config[MODEL_TYPE] == DEP_MODEL:
        model = DependencyTreeModel(config)
        model_name += "dep_model_directed_"            
    elif config[MODEL_TYPE] == POINTER_MODEL:
        model = PointerNetModel(config)           
        model_name += "pointer_"
    elif config[MODEL_TYPE] == DEP_TREETRAIN_BASELINE:
        model = DependencyTreeTrainBaseline(config)           
        model_name += "treetrain_baseline_"
    elif config[MODEL_TYPE] == DEP_BASELINE:
        model = DependencyBaselineModel(config)           
        model_name += "dep_baseline_"
    elif config[MODEL_TYPE] == LM_BASELINE:
        model = LMDecodingModel(config)   
        model_name += "lm_decoding_"
    model.to(config[DEVICE])
        
    if config[DATASET_TYPE] == 1:
        print("Running with 100k dataset.")
        model_name += "100k"
    elif config[DATASET_TYPE] == 2:
        print("Running with 250k dataset.")
        model_name += "250k"
    else:
        model_name += "testing"
    return model, model_name

def collate_graphs(samples, config):
    """
        Returns:
             all_edus: List of edus with corresponding word ids
             trees: List of DGLGraph for Constituency trees
    """
    
    all_edus = []
    l_trees, r_trees, all_trees, roots = [], [], [], []
    num_nodes = 0
    
    for edus, tree_g in samples:
        l_tree, r_tree, root = tree_g
                
        all_edus.append(edus)
        l_trees.append(l_tree)
        r_trees.append(r_tree)
        trees_graph = build_trees_graph(l_tree, l_tree)
        all_trees.append(trees_graph)
        roots.append(root)
        num_nodes += l_tree.nodes().shape[0]
                        
    batched_ltrees = dgl.batch(l_trees)
    batched_rtrees = dgl.batch(r_trees)
    batched_alltrees = dgl.batch(all_trees)
    
    assert num_nodes == batched_ltrees.nodes().shape[0]
    
    return all_edus, (batched_ltrees, batched_rtrees, batched_alltrees, th.tensor(roots, device=config[DEVICE]))


def train(config):
    
    data_helper_train, data_helper_val = DataHelper.load_data_helper(config)
    train_loader, val_loader = construct_datasets(data_helper_train, data_helper_val, config)
    print("Number of train datapoints: %d, number of val datapoints: %d" 
          % (len(train_loader), len(val_loader)))
    
    model_tree, model_name = construct_model(config)
    del data_helper_train
    del data_helper_val
    optim_tree = Optimizer(model_tree.parameters(), lr=config[LR], warmup_steps=config[WARMUP_STEPS])
    model_path = os.path.join("model_saves/", model_name + ".pt")
    if os.path.exists(model_path):
        checkpoint = th.load(model_path)
        start_epoch = checkpoint["epoch"]
        model_tree.load_state_dict(checkpoint['model_state_dict'])
        optim_tree.load_state_dict(checkpoint['optimizer_state_dict'], checkpoint['step'])
    else:
        start_epoch = 0 

    best_val_loss = 1000000000
    
    for epoch in range(start_epoch + 1, 300):
        model_tree.train()
        total_loss = 0
        print("Training, epoch ", epoch)
        for i, batch in enumerate(tqdm(train_loader)):
            optim_tree.zero_grad()
            loss_tree = model_tree(*batch)   
            if loss_tree != 0:
                total_loss += loss_tree.item()
                loss_tree.backward()
                clip_grad_norm_(model_tree.parameters(), 0.2)
                optim_tree.step()
        print("Completed epoch ", epoch)
        model_tree.eval()                    
        logprob_acc = eval_pass(val_loader, model_tree, config, epoch)
        
        if best_val_loss > logprob_acc:
            best_val_loss = logprob_acc
            th.save({
                'epoch': epoch,
                'model_state_dict': model_tree.state_dict(),
                'optimizer_state_dict': optim_tree.state_dict(),
                'step': optim_tree._step,
                'loss': total_loss,
                'dev_loss': logprob_acc,
            }, os.path.join("model_saves/", model_name + ".pt"))
            
            
def evaluate(config, epoch):
    model, model_name = construct_model(config)
    data_helper = DataHelper.load_data_helper(config)
    data_loader = construct_datasets(None, data_helper, config)
    
    if config[MODEL_TYPE] not in [DEP_BASELINE, LM_BASELINE]:
        checkpoint = th.load(os.path.join("model_saves/", model_name + ".pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.eval()
    print("Number of eval datapoints: ", len(data_loader))
    eval_pass(data_loader, model, config, epoch)

def eval_pass(data_loader, model, config, epoch):
    perfect_match_score, position_match_score, kendalls, block_kendalls = 0, 0, 0, 0
    logprob_acc, uas_acc, las_acc, count = 0, 0, 0, 0
    
    with th.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            edus, tree_g = sample
            pred_edu_order, loss, uas, las = model.decode(edus, tree_g, None)
            if i % 500 == 0 and i > 0:
                print("Evaluated ", i, " datapoints.")            
            kendalls += kendall_tau(pred_edu_order.cpu())
            block_kendalls += blocked_kendall_tau(pred_edu_order.cpu())
            perfect_match_score += perfect_match(pred_edu_order).item()
            position_match_score += position_match(pred_edu_order).item()
            logprob_acc += loss.item()
            
            if uas is not None:
                uas_acc += uas
            if las is not None:
                las_acc += las
            count += 1
            
        display_results(0, 
                        logprob_acc,
                        count, 
                        kendalls, 
                        block_kendalls,
                        position_match_score, 
                        perfect_match_score, 
                        float(uas_acc), 
                        float(las_acc))
        return logprob_acc
