import torch as th
import torch.nn as nn
import dgl
import numpy as np

from models.edu_embedding import EduEmbeddingModel
from models.modules.pointer_net import PointerNet
from models.modules.mst import *
from dataset.utils.span import DepSpanNode
from dataset.utils.constants import *
from collections import defaultdict
import copy
import gc

class DependencyTreeModel(nn.Module):
    
    def __init__(self, config):
        
        super(DependencyTreeModel, self).__init__()
        self.device = config[DEVICE]
        self.hid_dim = config[LSTM_HID] * 2
        self.num_lstm_pointer = 1
        self.config = config
        
        self.ptr_criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        
        self.bilin = nn.Bilinear(self.hid_dim, self.hid_dim, 2)
        self.head_lin = nn.Linear(self.hid_dim, 2, bias=False)
        self.dep_lin = nn.Linear(self.hid_dim, 2, bias=False)
        self.link_predictor = self.bilin_compat        
        
        self.root_clf = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim),
                nn.GELU(),
                nn.Linear(self.hid_dim, 1)
        )
        
        self.edu_embed_model = EduEmbeddingModel(config)
        
        self.pointer_net = PointerNet(self.hid_dim,
                                      self.hid_dim,
                                      self.num_lstm_pointer,
                                      self.config[DROPOUT],
                                      config)
            
        self.root_embed = nn.Parameter(th.rand(1, self.hid_dim), requires_grad=True)
        self.alpha = 0.25

    def message_func(self, edges):
        
        return {'src': edges.src['h'], 
                'dst': edges.dst['h'], 
                'ch_h': edges.src['ch_h'],
                'child_ids': edges.edges()[0], 
                'par_ids': edges.edges()[1]}        
        
    def reduce_func(self, nodes, doc_embed, batch_size, seq_len):
        
        num_seqs, num_children, _ = nodes.mailbox['dst'].shape
        if num_children > 1:
            doc_embed_idx = nodes.mailbox['par_ids'][:, 0] // seq_len
            out_prob = self.pointer_net(nodes.mailbox['src'], doc_embed[doc_embed_idx])
            out_prob = out_prob.reshape(-1, out_prob.shape[-1])
            pointer_targets = th.arange(num_children, device=self.config[DEVICE]).repeat(num_seqs, 1).view(-1)
            ptr_loss = self.ptr_criterion(out_prob, pointer_targets)
            self.total_score += (1 - self.alpha) * ptr_loss / batch_size
            
    def forward(self, all_edus, trees_graph):
        
        self.total_score = 0
        # Add embedding for root node
        l_trees_graph, r_trees_graph, trees_graph, roots = trees_graph
        total_score = self.directed_tree_loss(all_edus, l_trees_graph, r_trees_graph, trees_graph, roots)
        return total_score
    
    def directed_tree_loss(self, all_edus, l_trees_graph, r_trees_graph, trees_graph, roots):

        h_cat, doc_embed, doc_lengths = self.edu_embed_model(all_edus)
        sample_node_embeds = self.split_node_embed(h_cat, doc_lengths)
        batch, seq_len, _ = h_cat.shape
        h_cat_nopadding = th.cat(sample_node_embeds)
        trees_graph.ndata['h'] = h_cat_nopadding
        trees_graph.ndata['ch_h'] = th.zeros_like(h_cat_nopadding)
        trees_graph.register_message_func(self.message_func)
        trees_graph.register_reduce_func(lambda x: self.reduce_func(x, doc_embed, batch, seq_len))    
        trees_graph.pull(trees_graph.nodes())
        del trees_graph.ndata['h']
        del trees_graph.ndata['ch_h']
            
        left_adj, right_adj = [], []
        for i, (l_trees_subg, r_trees_subg) in enumerate(zip(dgl.unbatch(l_trees_graph), 
                                                             dgl.unbatch(r_trees_graph))):
            left_adj.append(l_trees_subg.reverse() \
                            .adjacency_matrix(transpose=True, ctx=th.device(self.config[DEVICE])) \
                            .to_dense() \
                            .unsqueeze(0))
            right_adj.append(r_trees_subg.reverse().adjacency_matrix(transpose=True, ctx=th.device(self.config[DEVICE])) \
                             .to_dense() \
                             .unsqueeze(0))
        
        left_adj = th.cat(left_adj)
        right_adj = th.cat(right_adj)
        compat_matrix = self.get_compat_matrix(h_cat)
        root_scores = self.root_clf(h_cat).view(h_cat.shape[0], -1)
        self.total_score += self.logistic_loss(compat_matrix, 
                                               (left_adj, right_adj), 
                                               (root_scores, roots)) / batch
        return self.total_score
        
    def split_node_embed(self, h_cat, doc_lengths):
        # Extract node sequences (without padding on the right)
        sample_node_embeds = []
        for i, doc_len in enumerate(doc_lengths):
            sample_node_embeds.append(h_cat[i, 0:doc_len])
        return sample_node_embeds
    
    def build_target_tensor(self, doc_lengths, seq_len):
        # Batched labels for pointer
        batch_size = len(doc_lengths)
        target_tensor =  th.ones((batch_size, seq_len), device=self.config[DEVICE],dtype=th.long) * -100
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
        raw_scores = self.link_predictor(src, dst).view(-1, num_nodes, num_nodes, 2)
        return raw_scores
    
    def bilin_compat(self, src, dst):
        # xWy + Wx + Wy + b
        bilin_score = self.bilin(src, dst)
        head_score = self.head_lin(src)
        dep_score = self.dep_lin(dst)
        return bilin_score + head_score + dep_score
        
    def logistic_loss(self, compat_matrix, adj_matrix, root):
        if self.training:
            left_adj_matrix, right_adj_matrix = adj_matrix
        else:
            compat_matrix = compat_matrix.unsqueeze(0)
            adj_matrix = adj_matrix.unsqueeze(0)
            left_adj_matrix, right_adj_matrix = adj_matrix[:,:,:,0], adj_matrix[:,:,:,1]
        root_scores, true_root = root    
        num_nodes = root_scores.shape[1]
        # Convert to double for numerical stability
        root_scores = root_scores.double()
        compat_matrix = compat_matrix.double()
        # Scores for root selection
        index = th.arange(root_scores.shape[0], device=self.config[DEVICE])
        gold_tree_weight = root_scores[index, true_root]
        # Scores for left edges
        left_edge_compat = compat_matrix[:, :, :, 0]
        print(left_edge_compat.shape, left_adj_matrix.shape)
        gold_tree_weight += th.sum(left_edge_compat * left_adj_matrix, dim=(1,2))
        # Scores for right edges
        right_edge_compat = compat_matrix[:, :, :, 1]
        gold_tree_weight += th.sum(right_edge_compat * right_adj_matrix, dim=(1,2))
        # Computing Z
        A = th.exp(compat_matrix)
        root_scores = th.exp(root_scores)
        A = th.sum(A, dim=3)    
        laplacian = th.diag_embed(th.sum(A, dim=1)) - A
        # Replacing top row with root scores (see paper)
        laplacian[:, 0, :] = root_scores
        # Negative log likelihood
        logdet = th.logdet(laplacian)
        # Ignore unstable cases (happens for short documents
        #                        late into training)
        mask = (gold_tree_weight <= logdet * 
                (th.isnan(gold_tree_weight) != 1) * 
                (th.isnan(logdet) != 1)).long()
        # loss = log Z - log e^(score of gold tree)
        loss = (logdet - gold_tree_weight) * mask
        loss[th.isnan(loss) == 1] = 0
        return self.alpha * th.sum(loss)

    def decode(self, all_edus, trees_graph, gold_tree):
        
        l_trees_graph, r_trees_graph, roots = trees_graph
        trees_graph = build_trees_graph(l_trees_graph, r_trees_graph)
        left_adj_matrix = l_trees_graph.reverse() \
                            .adjacency_matrix(transpose=True, ctx=th.device(self.config[DEVICE])) \
                            .to_dense().unsqueeze(2) 
        right_adj_matrix = r_trees_graph.reverse() \
                            .adjacency_matrix(transpose=True, ctx=th.device(self.config[DEVICE])) \
                            .to_dense().unsqueeze(2)
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
        dep_tree_root, new_adj_matrix = self.arrange_dep_tree_rootclf(msp_result, etype, int(root))
        uas, las = calc_uas_las(new_adj_matrix, gold_adj_matrix, root, gold_root, num_nodes)
        embeds = (h_cat, doc_embed)
        pred_order = self.node_order(embeds, dep_tree_root)
        pred_order = th.tensor(pred_order, device=self.config[DEVICE])      
        # Make 1 to num_nodes instead of 0 to num_nodes - 1
        pred_order += 1
        # Computing validation loss
        log_loss = self.logistic_loss(compat_matrix_full, gold_adj_matrix, (root_scores.unsqueeze(0), gold_root))
        self.total_score = 0
        trees_graph.ndata['h'] = h_cat
        trees_graph.ndata['ch_h'] = th.zeros_like(h_cat)
        trees_graph.register_message_func(self.message_func)
        trees_graph.register_reduce_func(lambda x: self.reduce_func(x, doc_embed, 1, num_nodes))  
        # Loss is accumulated in self.total_score
        trees_graph.pull(trees_graph.nodes())
        del trees_graph.ndata['h']
        del trees_graph.ndata['ch_h']
        return pred_order, self.total_score + log_loss, uas, las

    def decode_mst(self, compat_matrix_full, root_scores):
        
        num_nodes = int(root_scores.shape[0])
        beam_size = min(num_nodes, self.config[DEP_BEAM_SIZE])
        arcs = []
        
        compat_matrix, etype = th.max(compat_matrix_full, dim=2)
        
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
        nodes_indices = th.tensor([node.edu_id for node in children], device=self.config[DEVICE])
        pred_order = self.pointer_net.decode(h_cat[nodes_indices].unsqueeze(0), doc_embed)
        ordered_acc = [] 
        for i in pred_order.squeeze(0):
            ordered_acc.extend(acc[i])
            
        return ordered_acc
            
    def arrange_dep_tree_rootclf(self, result_dict, etype, root):
        nodes = [0] * etype.shape[0]
        num_nodes = len(result_dict) + 1
        new_adj_matrix = th.zeros((num_nodes, num_nodes, 2), 
                                  dtype=th.long, 
                                  device=self.config[DEVICE])

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
    uas = 1 - th.sum(flat_new_adj != flat_gold_adj).float() / (num_nodes * 2) - int(root != gold_root) / num_nodes
    las = 1 - th.sum(new_adj_matrix != gold_adj_matrix).float() / (num_nodes * 2) - int(root != gold_root) / num_nodes
    
    return uas, las

def build_trees_graph(l_trees_graph, r_trees_graph):
    trees_graph = dgl.DGLGraph()
    trees_graph.add_nodes(l_trees_graph.nodes().shape[0])
    trees_graph.add_edges(l_trees_graph.edges()[0], l_trees_graph.edges()[1])
    trees_graph.add_edges(r_trees_graph.edges()[0], r_trees_graph.edges()[1])
    return trees_graph