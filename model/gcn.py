# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
import math

class GCN(nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        
        self.hid_size = data.HP_hidden_dim
        
        self.W = nn.Parameter(torch.FloatTensor(self.hid_size, self.hid_size//2).cuda())
        self.b = nn.Parameter(torch.FloatTensor(self.hid_size//2, ).cuda())
        
        self.linear_gcn = nn.Linear(data.HP_hidden_dim // 2 *2, data.HP_hidden_dim//2)
        self.init()
    
    def init(self):
        stdv = 1/math.sqrt(self.hid_size//2)
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
    
    def forward(self, inp, adj, is_relu=True):
        inp = torch.matmul(inp, self.W)
        out = torch.matmul(adj, inp) + self.b
        # [batch, seq_len, dim] -> [batch, seq_len, dim//2]
        # [graph_num, batch, seq_len, seq_len] -> [graph_num, batch, seq_len, dim//2]
        batch_size, seq_len, _ = inp.size()

        if len(adj.size()) > 3:
            out = self.linear_gcn(out.transpose(0, 1).contiguous().transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
            # [graph_num, batch, seq_len, dim//2]
            # -> [batch, seq_len, graph_num * dim]
            # -> [batch, seq_len, dim//2]

        if is_relu == True:
            out = nn.functional.relu(out)
        
        return out
