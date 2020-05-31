import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from consts import PAD_ID


class Classfier(nn.Module):
    def __init__(self, emb_tensor):
        super(Classfier, self).__init__()
        self.embdic = nn.Embedding.from_pretrained(
            emb_tensor, padding_idx=PAD_ID)
        self.w_dims = emb_tensor.shape[1]

        # Define FCL
        self.l0 = nn.Linear(self.w_dims, 128)
        self.l1 = nn.Linear(self.w_dims, 128)
        self.l2 = nn.Linear(128*2, 3, bias=False)

        # Define Dropout
        p = 0.1
        self.dropout = nn.Dropout(p=p)

    def forward(self, former_idxs, latter_idxs):
        former_emb = self.embdic(former_idxs)
        latter_emb = self.embdic(latter_idxs)
        former_emb = self.dropout(former_emb)
        latter_emb = self.dropout(latter_emb)

        former = torch.sum(former_emb, 1)
        latter = torch.sum(latter_emb, 1)

        num_former_words = (former_idxs != PAD_ID).sum()
        num_latter_words = (latter_idxs != PAD_ID).sum()
        former /= num_former_words
        latter /= num_latter_words

        former = F.relu(self.l0(former))
        latter = F.relu(self.l1(latter))
        former = self.dropout(former)
        latter = self.dropout(latter)

        catted = torch.cat([former, latter], axis=-1)
        logit = self.l2(catted)

        return logit
