import torch as th
import torch.nn as nn
import dgl
import numpy as np

from models.edu_embedding import EduEmbeddingModel
from models.lm_decoding import LMDecodingModel
from torch.nn import CosineSimilarity
from models.modules.pointer_net import PointerNet
from models.modules.mst import *
from dataset.utils.span import DepSpanNode
from dataset.utils.constants import *
from collections import defaultdict
import copy
import gc


class DependencyBaselineModel(nn.Module):
    
    def __init__(self, config):
        
        super(DependencyBaselineModel, self).__init__()
        self.device = config[DEVICE]
        self.hid_dim = config[LSTM_HID] * 2
        self.num_lstm_pointer = 1
        self.config = config
        
        self.ptr_criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        
        self.link_predictor = CosineSimilarity(dim=2)
            
        self.root_clf = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim),
                nn.GELU(),
                nn.Linear(self.hid_dim, 1)
        )
        
        self.edu_embed_model = EduEmbeddingModel(config)
        self.lm_decoder = LMDecodingModel(config)
        
        if not self.config[USE_SEP_ENCODER]:
            self.pointer_net = PointerNet(self.hid_dim, \
                                          self.hid_dim, \
                                          self.num_lstm_pointer, \
                                          self.config[DROPOUT])
            
        self.root_embed = nn.Parameter(th.rand(1, self.hid_dim), requires_grad=True)
        self.alpha = 0.5

            
    def forward(self, all_edus, trees_graph, pointer_net=None):
        raise Exception("No forward in baseline!")
                
    def build_target_tensor(self, doc_lengths, seq_len):
        # Batched labels for pointer
        batch_size = len(doc_lengths)
        target_tensor =  th.ones((batch_size, seq_len), device=th.cuda.current_device(),dtype=th.long) * -100
        for i, doc_len in enumerate(doc_lengths):
            target_tensor[i, 0:doc_len] = th.arange(doc_len)
        return target_tensor
            
    def get_compat_matrix(self, h_cat):
        # Scores for N nodes -> NxNx2 compatibility matrix
        # entry i(0, N-1), j(0, N-1), k(0,1) means score for 
        # node i being k'th parent (left or right) of node j
        num_nodes = h_cat.shape[1]
        # [1, 2, 3, ...,N] -> [1,1,1...,2,2,2...,N,N,N] (N times)
        src = h_cat.repeat_interleave(num_nodes, 1)
        # [1, 2, 3, ...,N] -> [1, 2, ..., N, 1, 2, ..., N,...] (N times)
        dst = h_cat.repeat(1, num_nodes, 1)
        raw_scores = self.link_predictor(src, dst).view(-1, num_nodes, num_nodes).unsqueeze(3).repeat(1, 1, 1, 2)
        raw_scores += th.randn(1, num_nodes, num_nodes, 2).cuda() * 0.01
        
        return raw_scores
                    
    def decode(self, all_edus, trees_graph, gold_tree, pointer_net=None):
        self.all_edus = all_edus
        l_trees_graph, r_trees_graph, roots = trees_graph
        trees_graph = dgl.DGLGraph()
        trees_graph.add_nodes(l_trees_graph.nodes().shape[0])
        trees_graph.add_edges(l_trees_graph.edges()[0], l_trees_graph.edges()[1])
        trees_graph.add_edges(r_trees_graph.edges()[0], r_trees_graph.edges()[1])

        left_adj_matrix = l_trees_graph.reverse().adjacency_matrix(transpose=True).cuda().to_dense().unsqueeze(2)
        right_adj_matrix = r_trees_graph.reverse().adjacency_matrix(transpose=True).cuda().to_dense().unsqueeze(2)
        adj_matrix = th.cat([left_adj_matrix,right_adj_matrix], dim=2)
        
        return self.decode_directed(all_edus, gold_tree, adj_matrix, roots, trees_graph)

    def decode_directed(self, all_edus, gold_tree, gold_adj_matrix, gold_root, trees_graph):
        
        h_cat, doc_embed, num_seqs = self.edu_embed_model([all_edus])
        h_cat = h_cat.squeeze(0)
        num_nodes = int(h_cat.shape[0])        
        # Scores for parenthood and rootness
        compat_matrix_full = self.get_compat_matrix(h_cat.unsqueeze(0)).squeeze()
        root_scores = self.root_clf(h_cat).view(-1)
        # Decode the tree structure
        msp_result, etype, root = self.decode_mst(compat_matrix_full, root_scores)
        # Decode the EDU order from the tree
        dep_tree_root, new_adj_matrix = arrange_dep_tree_rootclf(msp_result, etype, int(root))
        uas, las = calc_uas_las(new_adj_matrix, gold_adj_matrix, root, gold_root, num_nodes)
        embeds = (h_cat, doc_embed)
        pred_order = self.node_order(embeds, dep_tree_root)
        assert len(pred_order) == num_nodes
        pred_order = th.cuda.LongTensor(pred_order)      
        # Make 1 to num_nodes instead of 0 to num_nodes - 1
        pred_order += 1
        return pred_order, th.zeros(1), uas, las

    def decode_mst(self, compat_matrix_full, root_scores):
        
        num_nodes = int(root_scores.shape[0])
        beam_size = min(num_nodes, self.config[DEP_BEAM_SIZE])
        arcs = []
        
        if self.config[NUM_CLASSES] == 2:
            compat_matrix, etype = th.max(compat_matrix_full, dim=2)
        else:
            etype = None
        
        _, indices = th.topk(root_scores, beam_size)
        
        for i in range(0, num_nodes):
            for j in range(0, num_nodes):
                if i != j:
                    arcs.append(Arc(j, float(-compat_matrix[i,j]), i))
                    
        # Find best tree                
        candidate_trees = []            
        tree_scores = []
        for root_idx in range(beam_size):
            root_id = int(indices[root_idx])
            msp_result = min_spanning_arborescence(arcs, root_id)
            score = 0
            for node in msp_result.values():
                score += node.weight
            candidate_trees.append(msp_result)
            tree_scores.append(score)
            
        best_idx = np.argmin(np.array(tree_scores))
        msp_result = candidate_trees[best_idx]
        
        return msp_result, etype, indices[best_idx]
    
    def node_order(self, embed, root):
        acc = []
        for lnode in root.lnodes:
            acc.append(self.node_order(embed, lnode))
            
        if len(root.lnodes) > 1:
            acc = self.direct_ch_order(embed, root.lnodes, acc, root.edu_id)
        elif len(root.lnodes) == 1:
            acc = acc[0]
                
        acc.append(root.edu_id)
            
        right_acc = []
                
        for rnode in root.rnodes:
            right_acc.append(self.node_order(embed, rnode))
            
        if len(root.rnodes) > 1:
            right_acc = self.direct_ch_order(embed, root.rnodes, right_acc, root.edu_id)
        elif len(root.rnodes) == 1:
            right_acc = right_acc[0]
            
        acc.extend(right_acc)
        
        return acc

    def direct_ch_order(self, embed, children, acc, root_id):
        
        h_cat, doc_embed = embed
        nodes_indices = th.cuda.LongTensor([node.edu_id for node in children])
        #pred_order = self.pointer_net.decode(h_cat[nodes_indices].unsqueeze(0), doc_embed)
        pred_order, _, _, _ = self.lm_decoder.decode(self.all_edus, nodes_indices, None, None) #th.randperm(len(children))
        ordered_acc = [] 
        for i in pred_order.squeeze(0):
            ordered_acc.extend(acc[i - 1])
            
        return ordered_acc
            
def arrange_dep_tree_rootclf(result_dict, etype, root):
    
    nodes = [0] * etype.shape[0]
    num_nodes = len(result_dict) + 1
    new_adj_matrix = th.zeros((num_nodes, num_nodes, 2), 
                              dtype=th.long, 
                              device=th.cuda.current_device())

    for _, value in result_dict.items():
        new_adj_matrix[value.head, value.tail, etype[value.head, value.tail]] = 1
        if nodes[value.head] == 0:
            nodes[value.head] = DepSpanNode(value.head)
        if nodes[value.tail] == 0:
            nodes[value.tail] = DepSpanNode(value.tail)
        if etype[value.head, value.tail]:
            nodes[value.head].rnodes.append(nodes[value.tail])
        else:
            nodes[value.head].lnodes.append(nodes[value.tail])
    assert nodes[root] != 0
    
    return nodes[root], new_adj_matrix   

def calc_uas_las(new_adj_matrix, gold_adj_matrix,
                root, gold_root, num_nodes):
    
    flat_new_adj = th.sum(new_adj_matrix, dim=2)
    flat_gold_adj = th.sum(gold_adj_matrix, dim=2)
    uas = 1 - (th.sum(flat_new_adj != flat_gold_adj).float()  + int(root != gold_root)) / (num_nodes * 2)
    las = 1 - (th.sum(new_adj_matrix != gold_adj_matrix).float() + int(root != gold_root)) / (num_nodes * 2)
    
    return uas, las