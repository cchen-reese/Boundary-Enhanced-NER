import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class StarEncoderLayer(nn.Module):
    ''' Star-Transformer: https://arxiv.org/pdf/1902.09113v2.pdf '''

    def __init__(self, d_model, n_head, d_k, d_v, glu_type, dropout):
        super(StarEncoderLayer, self).__init__()
        self.slf_attn_satellite = MultiHeadAttention(
            n_head, d_model, d_k, d_v, use_glu=glu_type, dropout=dropout)
        self.slf_attn_relay = MultiHeadAttention(
            n_head, d_model, d_k, d_v, use_glu=glu_type, dropout=dropout)

    def forward(self, h, e, s, non_pad_mask=None, slf_attn_mask=None):
        # satellite node
        batch_size, seq_len, d_model = h.size()
        h_extand = torch.zeros(batch_size, seq_len+2, d_model, dtype=torch.float, device=h.device)
        h_extand[:, 1:seq_len+1, :] = h  # head and tail padding(not cycle)
        s = s.reshape([batch_size, 1, d_model])
        s_expand = s.expand([batch_size, seq_len, d_model])
        context = torch.cat((h_extand[:, 0:seq_len, :],
                             h_extand[:, 1:seq_len+1, :],
                             h_extand[:, 2:seq_len+2, :],
                             e,
                             s_expand),
                            2)
        context = context.reshape([batch_size*seq_len, 5, d_model])
        h = h.reshape([batch_size*seq_len, 1, d_model])

        h, _ = self.slf_attn_satellite(h, context, context, mask=slf_attn_mask)
        h = torch.squeeze(h, 1).reshape([batch_size, seq_len, d_model])
        if non_pad_mask is not None:
            h *= non_pad_mask

        # virtual relay node
        s_h = torch.cat((s, h), 1)
        s, _ = self.slf_attn_relay(
            s, s_h, s_h, mask=slf_attn_mask)
        s = torch.squeeze(s, 1)

        return h, s