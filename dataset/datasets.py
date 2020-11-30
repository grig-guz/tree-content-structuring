import torch as th
import dgl
import numpy as np
from torch.utils.data import Dataset
from dataset.constituency_tree import ConstituencyTree
import os
from dataset.utils.constants import *
from torch.utils.data import Sampler, Dataset
from collections import OrderedDict
from random import shuffle

class EduDataset(Dataset):

    def __init__(self, constituency_trees, config):
        self.config = config
        self.edus = [self.preprocess_edus(tree) for tree in constituency_trees]
        
    def __len__(self):
        return len(self.edus)

    def __getitem__(self, idx):
        return self.edus[idx]
    
    def preprocess_edus(self, tree):
        """
            in: dictionary edu_id (in the document) -> list of token ids
           out: dictionary edu_id (in the document) -> list of glove ids
        """
        new_edus = {}
        for edu_id, tok_ids in tree.doc.edu_dict.items():
            edu_str = ""
            for tok_id in tok_ids:
                word = tree.doc.token_dict[tok_id].word
                edu_str += word + " "
            new_edus[edu_id] = edu_str
        return new_edus
    
class DependencyDataset(EduDataset):

    def __init__(self, constituency_trees, config):
        super().__init__(constituency_trees, config)        
        self.config = config
        dependency_trees = [ConstituencyTree.hirao_convert_to_dependency(tree.tree, tree.doc) 
                            for tree in constituency_trees]
        self.tree_graphs = [self.build_dependency_tree(tree) for tree in dependency_trees]
                            
    def __len__(self):
        return len(self.tree_graphs)

    def __getitem__(self, idx):
        edus = super().__getitem__(idx)
        return edus, self.tree_graphs[idx]
        
    def build_dependency_tree(self, tree):
        # Root and main nucleus
        left_child_tree = dgl.DGLGraph()
        right_child_tree = dgl.DGLGraph()
        # Root -> main nucleus edges
        main_id, _ = ConstituencyTree.postorder_DFT_dgl_dep_hirao(tree, 
                                                                  left_child_tree, 
                                                                  right_child_tree, 
                                                                  node_id=-1)

        return left_child_tree.reverse(), right_child_tree.reverse(), main_id
    
    
class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, len(p.doc.edu_dict)))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
    