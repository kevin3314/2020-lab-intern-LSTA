import numpy as np
import torch

from consts import UNK_ID, PAD_ID


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, word2idx, label2idx, mode='train'):
        self.mode = mode
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

        # strip
        res = [v.strip() for v in res]

        if self.mode == 'train' or self.mode == 'valid':
            res = [v.split('\t') for v in res]
        elif self.mode == 'test':
            # append dummy label
            res = [['neutral', *v.split('\t')] for v in res]
        else:
            raise AttributeError('Dataset.mode should be train/valid/test')

        return res

    def words2idxs(self, words):
        return [self.word2idx.get(word, UNK_ID) for word in words.split()]
