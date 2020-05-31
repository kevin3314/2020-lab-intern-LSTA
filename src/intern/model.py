import torch
import torch.nn as nn


class LSTM_divider(nn.Module):
    def __init__(self, emb_tensor, h_dims=100, dropout=0.1):
        super(LSTM_divider, self).__init__()
        self.embdic = nn.Embedding.from_pretrained(emb_tensor)
        self.w_dims = emb_tensor.shape[1]
        self.h_dims = h_dims

        # Define LSTM
        self.lstm = nn.LSTM(self.w_dims, self.h_dims)

        # Define Dropout
        p = 0.1
        self.dropout = nn.Dropout(p=p)

    def forward(self, idseq, length_list):
        batchsize, sent_len = idseq.size()
        idseq = torch.transpose(idseq, 0, 1)  # (batchsize, sent_len) -> (sent_len, batchsize)
        emb = self.embdic(idseq)

        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, length_list, enforce_sorted=False)
        h, c = self.init_lstm_state(batchsize, self.h_dims, device=idseq.device)
        packed_hidden, (h, c) = self.lstm(packed_emb, (h, c))
        unpacked_hidden, length_list2 = nn.utils.rnn.pad_packed_sequence(packed_hidden)
        unpacked_hidden = torch.transpose(unpacked_hidden, 0, 1)  # (sent_len, batchsize, hdims) -> (batchsize, sent_len, hdims)

        # average on hidden dims
        hidden_avg = torch.sum(unpacked_hidden, axis=-1) / self.h_dims
        out = torch.sigmoid(hidden_avg)

        return out

    def init_lstm_state(self, batchsize, hdims, device=None):
        h, c = torch.zeros(batchsize, hdims), torch.zeros(batchsize, hdims)
        # (1, batchsize, hdims)
        h = h.expand(1, batchsize, hdims)
        c = c.expand(1, batchsize, hdims)
        if device is not None:
            h = h.to(device)
            c = c.to(device)
        return h, c
