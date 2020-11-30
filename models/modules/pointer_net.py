# Modified from
#https://github.com/shirgur/PointerNet
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from queue import PriorityQueue
import operator
from dataset.utils.constants import *



class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, dec_input, logProb, length, mask, word_id):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.dec_input = dec_input
        self.logp = logProb
        self.length = length
        self.mask = mask
        self.word_id = word_id
        
    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim, config):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self.inf = torch.tensor([float('-inf')], device=config[DEVICE])
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))
        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)
        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)
        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        raw_att = att.clone()
        if len(att[mask]) > 0:
            att[mask] = self.inf
            raw_att[mask] = self.inf
        alpha = self.softmax(att)

        return alpha, raw_att

class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim, config):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim, config)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :return: (Output probabilities, Pointers indices), last hidden state
        """
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        runner.data = torch.arange(input_length, device=self.config[DEVICE])
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        # Recurrence loop
        for i in range(input_length):
            h_t, c_t, outs, raw_att = self.step(decoder_input, hidden, mask, context)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            #if self.training:
            indices[:] = i
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()
            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)
            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)
            
            outputs.append(raw_att.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))
            
        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)
        return (outputs, pointers), hidden
    
    def step(self, x, hidden, mask, context):
        """
        Recurrence step function
        :param Tensor x: Input at time t
        :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
        :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
        """

        # Regular LSTM
        h, c = hidden
        gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        input, forget, cell, out = gates.chunk(4, 1)

        input = torch.sigmoid(input)
        forget = torch.sigmoid(forget)
        cell = torch.tanh(cell)
        out = torch.sigmoid(out)

        c_t = (forget * c) + (input * cell)
        h_t = out * torch.tanh(c_t)

        # Attention section
        output, raw_att = self.att(h_t, context, torch.eq(mask, 0))
        return h_t, c_t, output, raw_att
    
    def beam_decode(self, encoder_outputs, decoder_input, decoder_hidden, context):
        # From https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        '''
        :param decoder_hidden: input tensor of shape [B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        beam_width = 64
        decoded_batch = []
        batch_size = 1
        input_length = encoder_outputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        
        # Generating arang(input_length), broadcasted across batch_size
        runner = torch.arange(input_length, device=self.config[DEVICE])
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()
        # decoding goes sentence by sentence
        for idx in range(batch_size):
            # Number of sentence to generate
            node = BeamSearchNode(decoder_hidden, 
                                  None, 
                                  decoder_input, 
                                  torch.zeros(1, device=self.config[DEVICE]), 
                                  0, 
                                  mask.clone(), 
                                  -1)
            nodes = []

            # start the queue
            nodes.append((-node.eval(), node))
            qsize = 1

            # start beam search
            for tstep in range(input_length):
                # give up when decoding takes too long                    
                new_nodes = []
                inputs, hiddens_h, hiddens_c, masks, old_nodes, old_logprobs = [], [], [], [], [], []
                while len(nodes) > 0:
                    # fetch the best nodes
                    score, n = nodes.pop()
                    decoder_input = n.dec_input
                    inputs.append(decoder_input)
                    decoder_hidden = n.h
                    hiddens_h.append(decoder_hidden[0])
                    hiddens_c.append(decoder_hidden[1])
                    mask = n.mask
                    masks.append(mask)
                    old_nodes.append(n)
                    old_logprobs.append(n.logp)
                inputs = torch.cat(inputs, dim=0)
                hiddens_h = torch.cat(hiddens_h, dim=0)
                hiddens_c = torch.cat(hiddens_c, dim=0)
                hiddens = (hiddens_h, hiddens_c)
                masks = torch.cat(masks, dim=0)
                old_logprobs = torch.cat(old_logprobs).unsqueeze(1).expand(-1, input_length)
                # decode for one step using decoder
                h_t, c_t, outs, raw_att = self.step(inputs, hiddens, masks, context.repeat(inputs.shape[0], 1, 1))
                beam_indexes = torch.arange(inputs.shape[0]).repeat_interleave(input_length)
                num_candidates = min(beam_width, input_length * inputs.shape[0])
                att_logprobs = self.log_softmax(raw_att)
                att_logprobs += old_logprobs
                log_prob, indexes = torch.topk(att_logprobs.view(-1), num_candidates)
                beam_indexes = beam_indexes[indexes]
                
                decoded_t = torch.remainder(indexes, input_length)
                one_hot_pointers = (runner == decoded_t.unsqueeze(1).expand(-1, outs.shape[1])).float()
                new_masks  = masks[beam_indexes] * (1 - one_hot_pointers)
                embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()
                decoder_input = encoder_outputs.repeat(num_candidates, 1, 1)[embedding_mask.data].view(num_candidates, self.embedding_dim)
                for new_k in range(num_candidates):
                    if log_prob[new_k] == self.att.inf:
                        break
                    beam_idx = beam_indexes[new_k]
                    node = BeamSearchNode((h_t[beam_idx].unsqueeze(0), c_t[beam_idx].unsqueeze(0)), 
                                          old_nodes[beam_idx], 
                                          decoder_input[beam_idx].unsqueeze(0), 
                                          log_prob[new_k].unsqueeze(0), 
                                          old_nodes[beam_idx].length + 1, 
                                          new_masks[new_k].unsqueeze(0), 
                                          decoded_t[new_k].item())
                    score = -node.eval()
                    new_nodes.append((score, node))
                    qsize += 1
                    
                # Prune the queue if necessary
                if qsize > beam_width:
                    nodes = sorted(new_nodes, key=operator.itemgetter(0))[:beam_width]
                else:
                    nodes = new_nodes
            endnodes = nodes
            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.word_id)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.word_id)
                utterance = utterance[::-1]
                utterances.append(utterance)
            decoded_batch.append(utterances)
        return torch.tensor(decoded_batch[0][0][1:], device=self.config[DEVICE])
    
class PointerNet(nn.Module):
    """
    Pointer-Net
    """
    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 config,
                 bidir=False):
        """
        Initiate Pointer-Net
        :param int embedding_dim: Number of embbeding channels
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.decoder = Decoder(hidden_dim, hidden_dim, config)
        self.config=config
        self.decoder_input0 = Parameter(torch.zeros(hidden_dim), requires_grad=False)

    def forward(self, inputs, decoder_hidden0):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
        decoder_hidden0 = (decoder_hidden0,
                               torch.zeros(inputs[0][-1].shape, device=self.config[DEVICE]))
        (outputs, pointers), _ = self.decoder(inputs,
                                              decoder_input0,
                                              decoder_hidden0,
                                              inputs)
        return outputs
    
    def decode(self, inputs, decoder_hidden0):
        batch_size = inputs.size(0)
        input_length = inputs.size(1)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
        decoder_hidden0 = (decoder_hidden0,
                               torch.zeros(inputs[0][-1].shape, device=self.config[DEVICE]))
        pointers = self.decoder.beam_decode(inputs,decoder_input0,decoder_hidden0,inputs)
        return pointers
    