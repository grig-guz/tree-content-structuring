#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-11-28 下午2:47
import gzip
import os
import pickle
from collections import defaultdict

import numpy as np
from dataset.constituency_tree import ConstituencyTree
from dataset.utils.constants import *
import torch


class DataHelper(object):
    
    def __init__(self):
        pass
        
    def create_data_helper(self, data_dir):
        self.constituency_trees, _ = self.read_constituency_trees(data_dir="mega_dt/")
        with open("data/data_helper_100trees.pkl", 'wb') as fout:
            pickle.dump(self, fout)
            
    @staticmethod
    def load_data_helper(config):
        if config[EVAL]:
            fname = "data/helper_test.pkl"
            with open(fname, 'rb') as fin:
                data_helper = pickle.load(fin)
            return data_helper
        else:
            if config[DATASET_TYPE] == 2:
                fname = "data/helper_train_250k.pkl"
            elif config[DATASET_TYPE] == 1:
                fname = "data/helper_train_100k.pkl"
            else:
                raise Exception("Unknown dataset size")
            #fname = "data/data_helper_100trees.pkl"
            with open(fname, 'rb') as fin:
                data_helper_train = pickle.load(fin)
            with open("data/helper_val.pkl", 'rb') as fin:
                data_helper_val = pickle.load(fin)
            
            return data_helper_train, data_helper_val
                        
    def read_constituency_trees(self, data_dir):
        # Read RST tree file
        files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dis')]
        rst_trees = []
        fdis_tree_map = {}
        print("Reading constituency trees")
        j = 0
        for i, fdis in enumerate(files):
            fmerge = fdis.replace('.dis', '.merge')
            if not os.path.isfile(fmerge):
                print("missing tree ", fmerge)
                continue
                raise FileNotFoundError('Corresponding .fmerge file does not exist. You should do preprocessing first.')
            rst_tree = ConstituencyTree(fdis, fmerge)
            rst_tree.build()
            if (rst_tree.get_parse().startswith(" ( EDU")):
                print("tree is corrupt ", fdis)
                continue
            rst_trees.append(rst_tree)
            fdis_tree_map[fdis] = rst_tree
            if i % 100 == 0:
                print("Read ", i, " trees")
            j += 1
        print("Trees actually read: ", j)
        return rst_trees, fdis_tree_map