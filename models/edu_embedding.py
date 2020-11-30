import torch as th
import torch.nn as nn
from torch.nn import LSTM, Embedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import dgl
import dgl.function as fn
from transformers import AlbertModel, AlbertTokenizer

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
from dataset.utils.constants import *
    
class EduEmbeddingModel(nn.Module):
    def __init__(self, config):
        
        super(EduEmbeddingModel, self).__init__()
        self.config = config
        d_model = 768
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        self.albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')            

        self.dropout = nn.Dropout(config[DROPOUT])
        self.word_dropout = nn.Dropout2d(config[DROPOUT])
        layer_norm = nn.LayerNorm(d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, 
                                                 nhead=config[NUM_HEADS], 
                                                 dropout=config[DROPOUT],
                                                 activation='gelu')
        self.sent_encoder = TransformerEncoder(encoder_layers, config[NUM_LAYERS], norm=layer_norm)
        self.lin_project = nn.Linear(d_model, config[LSTM_HID] * 2)
            

        
    def forward(self, edus_dict):
        """
            edus_dict - keys are EDU ordering, vals - arrays with 
                        word embedding indices for each word in EDU
        """
        edu_seqs, doc_lengths = self.albert_embed_edus(edus_dict)
        # Seq len x Batch x Hid dim
        padded_edu_seqs = pad_sequence(edu_seqs)
        padded_edu_seqs = self.word_dropout(padded_edu_seqs.permute(1, 0, 2)).permute(1, 0, 2)
        seq_len, batch_size = max(doc_lengths), len(doc_lengths)
        context_edus = self.sent_encoder(padded_edu_seqs)
        assert padded_edu_seqs.shape == context_edus.shape
        h_cat = self.lin_project(context_edus.permute(1, 0, 2))
        h_cat = self.word_dropout(h_cat)
        doc_embeds = th.sum(h_cat, dim=1) / th.tensor(doc_lengths, device=self.config[DEVICE]).unsqueeze(1)
        doc_embeds = self.dropout(doc_embeds)
        
        return h_cat, doc_embeds, doc_lengths
        
    def albert_embed_edus(self, edus_dict_list):
        
        all_edus, all_masks = [], []
        doc_lengths = []
        for edus in edus_dict_list:
            doc_lengths.append(len(edus))
            # NOTE: The order is only kept correctly by .values() in latest Python versions
            encoded_dict = self.albert_tokenizer.batch_encode_plus(edus.values(), 
                                                                   add_special_tokens=True, 
                                                                   truncation=True,
                                                                   max_length=50, 
                                                                   pad_to_max_length=True, 
                                                                   is_pretokenized=True) 
            for edu_encoding, mask in zip(encoded_dict['input_ids'], encoded_dict['attention_mask']):
                all_edus.append(th.tensor(edu_encoding, device=self.config[DEVICE]))
                all_masks.append(th.tensor(mask, device=self.config[DEVICE]))
        edus = pad_sequence(all_edus, batch_first=True)
        masks = pad_sequence(all_masks, batch_first=True)
        h_cat = self.albert(edus, attention_mask=masks)[0]
        h_cat = self.word_dropout(h_cat)
        # EDU embedding is the mean of last layer hidden states
        h_cat = th.sum(h_cat, dim=1) / th.sum(masks, dim=1).unsqueeze(1)
        assert sum(doc_lengths) == h_cat.shape[0]
        edu_seqs = th.split(h_cat, doc_lengths)
        return edu_seqs, doc_lengths
    
        