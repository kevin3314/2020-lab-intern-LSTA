import numpy as np
import torch

from consts import UNK_ID, PAD_ID


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, word2idx):
        self.data = self.load_data(data_path)
        self.word2idx = word2idx
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        label, former, latter = data

        former_idxs = self.words2idxs(former)
        latter_idxs = self.words2idxs(latter)

        return label, former_idxs, latter_idxs

    def load_data(self, path):
        with open(path) as f:
            res = f.readlines()

        return res

    def words2idxs(self, words):
        return [self.word2idx.get(word, UNK_ID) for word in words.split()]
