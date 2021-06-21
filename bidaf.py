# -*- coding: utf-8 -*-
"""
基于原文提取的阅读理解算法：Bidaf
@author:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 保证每次运行生成的随机数相同
torch.manual_seed(123)
torch.cuda.manual_seed(123)

class CharacterEmbedLayer(nn.Module):
    def __init__(self, char_vocab_size, char_embed_size, cnn_out_channels, dropout=0.2):
        super(CharacterEmbedLayer, self).__init__()

        self.char_embed_size = char_embed_size
        self.out_channels = cnn_out_channels

        self.embedding = nn.Embedding(char_vocab_size, char_embed_size, padding_idx=1)
        self.char_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn_out_channels, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, word_len]
        batch_size = x.size(0)
        word_len = x.size(2)
        out = self.embedding(x) # [batch_size, seq_len, word_len, char_embed_size]
        out = self.dropout(out)
        out = out.view(-1, word_len, self.char_embed_size) # [batch_size * seq_len, word_len, char_embed_size]
        out = out.unsqueeze(1) # in_channels = 1
        out = self.char_conv(out) # [batch_size * seq_len, out_channels, conv_out_H, conv_out_W]
        out = F.max_pool2d(out, kernel_size=(out.size(-2), out.size(-1))) # [batch_size * seq_len, out_channels, 1, 1]
        out = out.squeeze()
        out = out.view(batch_size, -1, self.out_channels) # [batch_size, seq_len, out_channels]
        return out
        

class WordEmbedLayer(nn.Module):
    def __init__(self, pretrained=None, word_vocab_size=5000, word_embed_size=100):
        super(WordEmbedLayer, self).__init__()
        if pretrained is not None:
            self.word_embed = nn.Embedding.from_pretrained(pretrained, freeze=True)
        else:
            self.word_embed = nn.Embedding(word_vocab_size, c_word_embed_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        out = self.word_embed(x) # [batch_size, seq_len, word_embed_size]
        return out


class ContextualEmbedLayer(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=0.2):
        super(ContextualEmbedLayer, self).__init__()
        self.bilstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout
                        )

    def forward(self, x, lengths):
        # x: [batch_size, seq_len, embed_size]
        sorted_len, sorted_idx = torch.sort(lengths, descending=True)
        sorted_x = x[sorted_idx.long()]
        _, ori_idx = torch.sort(sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, (h_n, c_n) = self.bilstm(packed_x) 
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = out[ori_idx.long()]
        return out # [batch_size, seq_len, hidden_size*2]


class AttentionFlowLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionFlowLayer, self).__init__()
        self.alpha = nn.Linear(6*hidden_size, 1)
        self.beta = nn.Linear(hidden_size*8, hidden_size*8)

    def forward(self, context, query):
        # context: [batch_size, c_seq_len, hidden_size*2]
        # query: [batch_size, q_seq_len, hidden_size*2]
        batch_size = context.size(0)
        c_seq_len = context.size(1)
        q_seq_len = query.size(1)

        context = context.unsqueeze(2) 
        query = query.unsqueeze(1)
        _context = context.expand(-1, -1, q_seq_len, -1) # [batch_size, c_seq_len, q_seq_len, hidden_size*2]
        _query = query.expand(-1, c_seq_len, -1, -1) # [batch_size, c_seq_len, q_seq_len, hidden_size*2]

        c_q = torch.mul(_context, _query) # [batch_size, c_seq_len, q_seq_len, hidden_size*2],逐元素相乘
        cat1 = torch.cat((_context, _query, c_q), dim=-1)
        S = self.alpha(cat1)
        S = S.squeeze() # [batch_size, c_seq_len, q_seq_len]
        if batch_size == 1: # squeeze会处理掉size为1的维度，因此当batch_size=1时，squeeze操作后需要补充batch_size所在的维度
            S = S.unsqueeze(0)

        query = query.squeeze()
        if batch_size == 1: 
            query = query.unsqueeze(0)

        c_q_atten_w = F.softmax(S, dim=-1)
        query_hat = torch.bmm(c_q_atten_w, query) # [batch_size, c_seq_len, hidden_size*2]

        context = context.squeeze()
        if batch_size == 1: 
            context = context.unsqueeze(0)

        q_c_atten_w = F.softmax(torch.max(S, dim=2)[0], dim=1).unsqueeze(1) # [batch_size, 1, c_seq_len]
        context_hat = torch.bmm(q_c_atten_w, context) # [batch_size, 1, hidden_size*2]
        context_hat = context_hat.expand(-1, c_seq_len, -1) # [batch_size, c_seq_len, hidden_size*2]

        context_query_hat = torch.mul(context, query_hat) # [batch_size, c_seq_len, hidden_size*2]
        context_context_hat = torch.mul(context, context_hat) # [batch_size, c_seq_len, hidden_size*2]
        cat2 = torch.cat((context_hat, query_hat, context_query_hat, context_context_hat), dim=-1)
        out = self.beta(cat2) # [batch_size, c_seq_len, hidden_size*8]

        return out


class ModelingLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super(ModelingLayer, self).__init__()
        self.two_layer_bilstm = nn.LSTM(input_size=hidden_size*8,
                                    hidden_size=hidden_size,
                                    num_layers = 2,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=dropout
                                )
    def forward(self, x, lengths):
        # x: [batch_size, c_seq_len, hidden_size*8]
        sorted_len, sorted_idx = torch.sort(lengths, descending=True)
        sorted_x = x[sorted_idx.long()]
        _, ori_idx = torch.sort(sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, (h_n, c_n) = self.two_layer_bilstm(packed_x) 
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = out[ori_idx.long()]

        return out # [batch_size, c_seq_len, hidden_size*2]


class OutputLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super(OutputLayer, self).__init__()
        self.p1_weight = nn.Linear(hidden_size*10, 1)
        self.p2_weight = nn.Linear(hidden_size*10, 1)
        self.bilstm = nn.LSTM(input_size=hidden_size*2,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout
                        )
        
    def forward(self, G, M, lengths):
        # G: [batch_size, c_seq_len, hidden_size*8]
        # M: [batch_size, c_seq_len, hidden_size*2]
        batch_size = G.size(0)
        cat1 = torch.cat((G,M), dim=-1)
        p1_context = self.p1_weight(cat1).squeeze() # [batch_size, c_seq_len]
        if batch_size == 1: 
            p1_context = p1_context.unsqueeze(0)
        p1 = F.softmax(p1_context, dim=-1)

        sorted_len, sorted_idx = torch.sort(lengths, descending=True)
        sorted_M = M[sorted_idx.long()]
        _, ori_idx = torch.sort(sorted_idx)

        packed_M = nn.utils.rnn.pack_padded_sequence(sorted_M, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_M2, (h_n, c_n) = self.bilstm(packed_M) 
        M2, _ = nn.utils.rnn.pad_packed_sequence(packed_M2, batch_first=True)
        M2 = M2[ori_idx.long()] # M2: [batch_size, c_seq_len, hidden_size*2]

        cat2 = torch.cat((G, M2), dim=-1)
        p2_context = self.p2_weight(cat2).squeeze() # [batch_size, c_seq_len]
        if batch_size == 1: 
            p2_context = p2_context.unsqueeze(0)
        p2 = F.softmax(p2_context, dim=-1)

        return p1, p2
 

class BidafModel(nn.Module):
    def __init__(self, config):
        super(BidafModel, self).__init__()

        self.char_embed = CharacterEmbedLayer(config.char_vocab_size, config.char_embed_size, config.cnn_out_channels)
        self.word_embed = WordEmbedLayer(pretrained=config.pretrained)
        self.contextual_embed = ContextualEmbedLayer(config.embed_size, config.hidden_size)

        self.att_flow = AttentionFlowLayer(config.hidden_size)
        
        self.modeling_layer = ModelingLayer(config.hidden_size)

        self.output_layer = OutputLayer(config.hidden_size)

        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(config.word_embed_size+config.cnn_out_channels, config.word_embed_size+config.cnn_out_channels),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(config.word_embed_size+config.cnn_out_channels, config.word_embed_size+config.cnn_out_channels),
                                  nn.Sigmoid()))

    def highway_network(self, char_embed, word_embed):
        # char_embed: [batch, seq_len, char_channel_size]
        # word_embed: [batch, seq_len, word_embed_size]
        
        embed = torch.cat([char_embed, word_embed], dim=-1) # (batch, seq_len, char_channel_size + word_embed_size)
        for i in range(2):
            h = getattr(self, 'highway_linear{}'.format(i))(embed)
            g = getattr(self, 'highway_gate{}'.format(i))(embed)
            embed = g * h + (1 - g) * embed # (batch, seq_len, char_channel_size + word_embed_size)
        return embed

    def forward(self, context, query):
        '''
            context = (c_word, c_char, c_seq_len)
            query = (q_word, q_char, q_seq_len)
            c_word: [batch_size, c_seq_len]
            c_char: [batch_size, c_seq_len, word_len]
            q_word: [batch_size, q_seq_len]
            q_char: [batch_size, q_seq_len, word_len]
            c_seq_len, q_seq_len: [batch_size]
        '''
        c_word, c_char, c_seq_len = context
        q_word, q_char, q_seq_len = query
        
        c_char_embed = self.char_embed(c_char) # [batch_size, c_seq_len, out_channels]
        c_word_embed = self.word_embed(c_word) # [batch_size, c_seq_len, word_embed_size]
        c_embed = self.highway_network(c_char_embed, c_word_embed) # [batch_size, c_seq_len, embed_size]
        c_contextual_embed = self.contextual_embed(c_embed, c_seq_len) # [batch_size, c_seq_len, hidden_size*2]

        q_char_embed = self.char_embed(q_char) # [batch_size, q_seq_len, out_channels]
        q_word_embed = self.word_embed(q_word) # [batch_size, q_seq_len, word_embed_size]
        q_embed = self.highway_network(q_char_embed, q_word_embed) # [batch_size, q_seq_len, embed_size]
        q_contextual_embed = self.contextual_embed(q_embed, q_seq_len) # [batch_size, q_seq_len, hidden_size*2]

        G = self.att_flow(c_contextual_embed, q_contextual_embed) # [batch_size, c_seq_len, hidden_size*8]
        M = self.modeling_layer(G, c_seq_len) # [batch_size, c_seq_len, hidden_size*2]
        
        p1, p2 = self.output_layer(G, M, c_seq_len) # p1, p2: [batch_size, c_seq_len]
        
        return p1, p2
