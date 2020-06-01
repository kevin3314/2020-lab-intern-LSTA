import torch
import torch.nn as nn
import torch.functional as F


class LSTM_divider(nn.Module):
    def __init__(
            self,
            voc_size,
            w_dims=128,
            h_dims=100,
            num_layers=1,
            bidirectional=True,
            dropout=0.1,
        ):
        super(LSTM_divider, self).__init__()
        self.voc_size = voc_size
        self.w_dims = w_dims
        self.h_dims = h_dims
        self.embdic = nn.Embedding(self.voc_size, self.w_dims)
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        self.num_layers = num_layers

        # Define LSTM
        self.lstm = nn.LSTM(
                self.w_dims,
                self.h_dims,
                num_layers=num_layers,
                bidirectional=bidirectional)

        self.relu = nn.ReLU()
        self.l0 = nn.Linear(self.h_dims * 2, 1)

        # Define Dropout
        p = 0.1
        self.dropout = nn.Dropout(p=p)

    def forward(self, idseq, length_list):
        batchsize, sent_len = idseq.size()
        # (batchsize, sent_len) -> (sent_len, batchsize)
        idseq = torch.transpose(idseq, 0, 1)
        emb = self.embdic(idseq)

        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, length_list, enforce_sorted=False
        )
        h, c = self.init_lstm_state(
                batchsize, self.h_dims, device=idseq.device)
        packed_hidden, (h, c) = self.lstm(packed_emb, (h, c))
        unpacked_hidden, length_list2 = nn.utils.rnn.pad_packed_sequence(
                packed_hidden)
        # (sent_len, batchsize, hdims) -> (batchsize, sent_len, hdims)
        unpacked_hidden = torch.transpose(unpacked_hidden, 0, 1)

        # (batchsize, sent_len, hdims) -> (batchsize, sent_len, 1)
        out = self.l0(self.relu(unpacked_hidden))

        # (batchsize, sent_len) -> (batchsize, sent_len)
        out = out.squeeze(-1)

        out = torch.sigmoid(out)

        return out

    def init_lstm_state(self, batchsize, hdims, device=None):
        h, c = torch.zeros(batchsize, hdims), torch.zeros(batchsize, hdims)
        # (1, batchsize, hdims)
        h = h.expand(self.num_layers * self.num_directions, batchsize, hdims)
        c = c.expand(self.num_layers * self.num_directions, batchsize, hdims)
        if device is not None:
            h = h.to(device)
            c = c.to(device)
        return h, c
