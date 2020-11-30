import torch as th
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import dgl
import dgl.function as fn
from transformers import AlbertModel, AlbertForMaskedLM, AlbertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
from dataset.utils.constants import *
    
class LMDecodingModel(nn.Module):
    
    def __init__(self, config):
        super(LMDecodingModel, self).__init__()
        self.config = config
        self.dep_tree_baseline = config[MODEL_TYPE] == DEP_TREETRAIN_BASELINE
        self.albert = AlbertForMaskedLM.from_pretrained('albert-base-v2')
        self.albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')  
        
        #self.albert = GPT2LMHeadModel.from_pretrained('gpt2')
        #self.albert_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  
        
    def forward(self, edus_dict):
        raise Exception("Not implemented")
        
    def decode(self, edus_dict, node_indices, *args):
        """
            edus_dict - keys are EDU ordering, vals - arrays with 
                        word embedding indices for each word in EDU
        """
        if not self.dep_tree_baseline:
            node_indices = node_indices[0].nodes().cuda()
        best_seq = self.albert_best_seq(edus_dict, node_indices)
        return best_seq
        
    def albert_best_seq(self, edus_dict_list, node_indices):
        
        all_edus, all_masks = [], []
        pred_seq = []
        pred_word_seq = []
        pred_mask_seq = []
        encoded_dict = self.albert_tokenizer.batch_encode_plus(edus_dict_list.values(), 
                                                               add_special_tokens=True,
                                                               is_pretokenized=True) 
        for edu_encoding, mask in zip(encoded_dict['input_ids'], encoded_dict['attention_mask']):
            all_edus.append(th.cuda.LongTensor(edu_encoding))
            all_masks.append(th.cuda.LongTensor(mask))
        edus = pad_sequence(all_edus, batch_first=True)
        masks = pad_sequence(all_masks, batch_first=True)
        new_edu_dict = {int(i+1):edus_dict_list[int(i+1)] for i in node_indices}
        new_node_indices = list(node_indices)
        while new_edu_dict != {}:
            scores = []
            indices = []
            for i in new_edu_dict:
                i -= 1
                i = th.cuda.LongTensor([i])
                if pred_word_seq == []:
                    candidate_wordseq = edus[i]
                    candidate_maskseq = masks[i]
                else:
                    candidate_wordseq = th.cat([pred_word_seq, all_edus[i].unsqueeze(0)], dim=1)
                    candidate_wordseq = candidate_wordseq[:, max(0, candidate_wordseq.shape[1] - 500):]
                    candidate_maskseq = th.cat([pred_mask_seq, all_masks[i].unsqueeze(0)], dim=1)
                    candidate_maskseq = candidate_maskseq[:, max(0, candidate_maskseq.shape[1] - 500):]
                score = self.albert(candidate_wordseq, 
                                      attention_mask=candidate_maskseq,
                                      labels=candidate_wordseq)[0]
                score_norm = score / th.sum(candidate_maskseq).float()
                scores.append(score_norm)
                indices.append(i + 1)
            best_idx = th.argmax(th.cuda.FloatTensor(scores))
            pred_seq.append(new_node_indices.index(indices[best_idx] - 1))
            if pred_word_seq == []:
                pred_word_seq = all_edus[best_idx].unsqueeze(0)
                pred_mask_seq = all_masks[best_idx].unsqueeze(0)
            else:
                pred_word_seq = th.cat([pred_word_seq, all_edus[best_idx].unsqueeze(0)], dim=1)
                pred_mask_seq = th.cat([pred_mask_seq, all_masks[best_idx].unsqueeze(0)],dim=1)
            del new_edu_dict[int(indices[best_idx])]
        return th.cuda.LongTensor(pred_seq), th.zeros(1), None, None
    
        