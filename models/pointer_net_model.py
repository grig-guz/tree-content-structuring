import torch as th
import torch.nn as nn
import dgl
from models.edu_embedding import EduEmbeddingModel
from models.modules.pointer_net import PointerNet
from dataset.utils.span import SpanNode
from dataset.utils.constants import *


class PointerNetModel(nn.Module):
    
    def __init__(self, config):
        super(PointerNetModel, self).__init__()
        self.config = config
        self.device = config[DEVICE]
        self.hid_dim = config[LSTM_HID] * 2
        self.num_lstm_pointer = 1
        self.dropout_num = config[DROPOUT]
        self.root_embed = nn.Parameter(th.rand(1, self.hid_dim), requires_grad=True)
        self.ptr_criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        self.edu_encoder = EduEmbeddingModel(config)
        self.pointer_net = PointerNet(self.hid_dim, \
                                      self.hid_dim, \
                                      self.num_lstm_pointer, \
                                      config[DROPOUT], \
                                      config)

    def message_func(self, edges):
        return {'src': edges.src['h'], 
                'dst': edges.dst['h'], 
                'ch_h': edges.src['ch_h'],
                'child_ids': edges.edges()[0], 
                'par_ids': edges.edges()[1]}        
        
    def reduce_func(self, nodes, doc_embed):
        num_seqs, num_children, _ = nodes.mailbox['dst'].shape
        if num_children > 1:
            out_prob, _ = self.pointer_net(nodes.mailbox['src'], doc_embed)
            out_prob = out_prob.reshape(-1, out_prob.shape[-1])
            pointer_targets = th.arange(num_children, device=self.device).repeat(num_seqs, 1).view(-1)
            ptr_loss = self.ptr_criterion(out_prob, pointer_targets)
            self.total_score += ptr_loss
         
    def reduce_func_flat(self, nodes, doc_embed):
        num_seqs, num_children, _ = nodes.mailbox['dst'].shape
        parent_pos_idx = th.sum(nodes.mailbox['par_ids'] > nodes.mailbox['child_ids'], dim=1)
        parent_nodes = nodes.mailbox['dst'][:, 0, :] + self.root_embed
        new_seqs = []
        for i, pos_idx in enumerate(parent_pos_idx):
            new_seq = th.cat([nodes.mailbox['src'][i, :pos_idx],
                                    parent_nodes[i].unsqueeze(0),
                                    nodes.mailbox['src'][i, pos_idx:]])
            new_seqs.append(new_seq.unsqueeze(0))
        new_seqs = th.cat(new_seqs)
        out_prob, _ = self.pointer_net(new_seqs, doc_embed)
        out_prob = out_prob.reshape(-1, out_prob.shape[-1])
        pointer_targets = th.arange(num_children + 1, self.device).repeat(num_seqs, 1).view(-1)
        ptr_loss = self.pointer_loss(out_prob, pointer_targets)
        self.total_score += ptr_loss
    
    def forward(self, all_edus, trees_graph=None, *args):
        h_cat, doc_embed, doc_lengths = self.edu_encoder(all_edus)
        batch = h_cat.shape[0]
        seq_len = h_cat.shape[1]
        self.total_score = 0
        out_prob = self.pointer_net(h_cat, doc_embed)
        out_prob = out_prob.contiguous().view(-1, out_prob.shape[-1])

        pointer_targets = self.build_target_tensor(doc_lengths, seq_len).view(-1)
        self.total_score = self.ptr_criterion(out_prob, pointer_targets) / batch
        return self.total_score, None
    
    def decode(self, all_edus, doc_embed=None, *args):
        h_cat, doc_embed, doc_lengths = self.edu_encoder([all_edus])
        seq_len = h_cat.shape[1]
        out_prob = self.pointer_net(h_cat, doc_embed)
        out_ptr = self.pointer_net.decode(h_cat, doc_embed)
        out_prob = out_prob.contiguous().view(-1, out_prob.shape[-1])
        out_ptr += 1
        pointer_targets = self.build_target_tensor(doc_lengths, seq_len).view(-1)
        logprob = self.ptr_criterion(out_prob, pointer_targets)
        return out_ptr.squeeze(0), logprob, None, None
    
    def build_target_tensor(self, doc_lengths, seq_len):
        batch_size = len(doc_lengths)
        target_tensor =  th.ones((batch_size, seq_len), device=self.config[DEVICE],dtype=th.long) * -100
        for i, doc_len in enumerate(doc_lengths):
            target_tensor[i, 0:doc_len] = th.arange(doc_len)
        return target_tensor