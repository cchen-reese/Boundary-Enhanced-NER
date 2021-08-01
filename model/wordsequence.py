# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import torch.autograd as autograd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
from .transformer_encoder import StarEncoderLayer
from .gcn import GCN



class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + autograd.Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.drophead = nn.Dropout(data.HP_dropout)
        self.droptail = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.feature_num = data.feature_num
        self.dropout = data.HP_dropout
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(self.feature_num):
            self.input_size += data.feature_emb_dims[idx]

        print("word+char embedding dim ：", self.input_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

            self.star_layer = data.HP_star_layer
            self.word2star = nn.Linear(self.input_size, data.HP_hidden_dim)
            self.posi = PositionalEncoding(data.HP_hidden_dim, self.dropout)
            self.star_transformer = StarEncoderLayer(
                d_model = data.HP_hidden_dim, 
                n_head = data.HP_star_head,
                d_k = data.HP_hidden_dim, 
                d_v = data.HP_hidden_dim,
                glu_type = data.HP_star_glu,
                dropout = data.HP_star_dropout
                )

        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.HP_hidden_dim
            self.word2cnn = nn.Linear(self.input_size, data.HP_hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = int((kernel-1)/2)
            for idx in range(self.cnn_layer):
                self.cnn_list.append(nn.Conv1d(data.HP_hidden_dim, data.HP_hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.HP_dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.HP_hidden_dim))

        elif self.word_feature_extractor == "STAR":
            self.star_layer = data.HP_star_layer
            self.word2star = nn.Linear(self.input_size, data.HP_hidden_dim)
            self.posi = PositionalEncoding(data.HP_hidden_dim, self.dropout)
            self.star_transformer = StarEncoderLayer(
                d_model=data.HP_hidden_dim,
                n_head=data.HP_star_head,
                d_k=data.HP_hidden_dim,
                d_v=data.HP_hidden_dim,
                glu_type=data.HP_star_glu,
                dropout=data.HP_star_dropout
                )


        # GCN
        self.gcn_layer = data.HP_gcn_layer
        self.gcn_fw_feature = GCN(data)
        self.gcn_bw_feature = GCN(data)
        self.dropgcn = nn.Dropout(data.HP_dropout)

        self.heads_layer = nn.GRU(self.input_size, lstm_hidden, num_layers=data.HP_head_layer, batch_first=True, bidirectional=self.bilstm_flag)
        self.tails_layer = nn.GRU(self.input_size, lstm_hidden, num_layers=data.HP_tail_layer, batch_first=True, bidirectional=self.bilstm_flag)

        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)
        self.hidden2head = nn.Linear(data.HP_hidden_dim, 2)
        self.hidden2tail = nn.Linear(data.HP_hidden_dim, 2)

        self.weight1 = nn.Parameter(torch.ones(1)).cuda()
        self.weight2 = nn.Parameter(torch.ones(1)).cuda()
        self.weight3 = nn.Parameter(torch.ones(1)).cuda()

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.drophead = self.drophead.cuda()
            self.droptail = self.droptail.cuda()
            self.dropgcn = self.dropgcn.cuda()
            self.heads_layer = self.heads_layer.cuda()
            self.tails_layer = self.tails_layer.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.hidden2head = self.hidden2head.cuda()
            self.hidden2tail = self.hidden2tail.cuda()
            self.gcn_fw_feature = self.gcn_fw_feature.cuda()
            self.gcn_bw_feature = self.gcn_bw_feature.cuda()


            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            elif self.word_feature_extractor == "LSTM" or self.word_feature_extractor == "GRU":
                self.lstm = self.lstm.cuda()


                self.word2star = self.word2star.cuda()
                self.posi = self.posi.cuda()
                self.star_transformer = self.star_transformer.cuda()

            elif self.word_feature_extractor == "STAR":
                self.word2star = self.word2star.cuda()
                self.posi = self.posi.cuda()
                self.star_transformer = self.star_transformer.cuda()


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_graph):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        
        word_represent = self.wordrep(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)

        denp_matrix_fw = batch_graph
        denp_matrix_bw = batch_graph.transpose(1, 2)
        gcn_out = self.word2star(word_represent)
        for i in range(self.gcn_layer):
            out_fw = self.gcn_fw_feature(gcn_out, denp_matrix_fw)
            out_bw = self.gcn_bw_feature(gcn_out, denp_matrix_bw)
            gcn_out = self.dropgcn(torch.cat([out_fw, out_bw], dim=-1))

        if self.word_feature_extractor == "CNN":
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2,1).contiguous()
        elif self.word_feature_extractor == "LSTM" or self.word_feature_extractor == "GRU":
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            ## lstm_out (seq_len, seq_len, hidden_size)
            feature_out = self.droplstm(lstm_out.transpose(1,0))
        elif self.word_feature_extractor == "STAR":
            x = gcn_out
            h = self.posi(x)
            s = torch.mean(x, 1) 
            for idx in range(self.star_layer):
                h, s = self.star_transformer(h, x, s)
            feature_out = h

        # extract head feature
        packed_heads = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        heads_out, hidden = self.heads_layer(packed_heads, hidden)
        heads_out, _ = pad_packed_sequence(heads_out)           # heads_out (seq_len, batch_size, hidden_size)
        heads_feature = self.drophead(heads_out.transpose(1, 0))    # heads_out (batch_size, seq_len, hidden_size)
        heads_outputs = self.hidden2head(heads_feature)
        # extract tail feature
        packed_tails = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        tails_out, hidden = self.tails_layer(packed_tails, hidden)
        tails_out, _ = pad_packed_sequence(tails_out)  # tails_out (seq_len, batch_size, hidden_size)
        tails_feature = self.drophead(tails_out.transpose(1, 0))  # tails_out (batch_size, seq_len, hidden_size)
        tails_outputs = self.hidden2tail(tails_feature)

        # feature_out = torch.cat([feature_out, heads_feature, tails_feature], 2)
        outputs = self.weight1 * feature_out + self.weight2 * heads_feature + self.weight3 * tails_feature
        outputs = self.hidden2tag(outputs)

        return heads_outputs, tails_outputs, outputs

    def sentence_representation(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, ), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        batch_size = word_inputs.size(0)
        if self.word_feature_extractor == "CNN":
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = F.max_pool1d(cnn_feature, cnn_feature.size(2)).view(batch_size, -1)
        elif self.word_feature_extractor == "LSTM" or self.word_feature_extractor == "GRU":
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            ## lstm_out (seq_len, seq_len, hidden_size)
            ## feature_out (batch_size, hidden_size)
            feature_out = hidden[0].transpose(1,0).contiguous().view(batch_size,-1)

        feature_list = [feature_out]
        for idx in range(self.feature_num):
            feature_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        final_feature = torch.cat(feature_list, 1)
        outputs = self.hidden2tag(self.droplstm(final_feature))
        return outputs
