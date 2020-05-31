import pickle

import numpy as np
import torch

from consts import UNK_ID, PAD_ID, char2idx


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, char2idx):
        self.data = self.load_data(data_path)
        self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        ids = self.chars2idxs(data)

        return ids

    def load_data(self, path):
        with open(path, 'rb') as f:
            res = pickle.load(f)

        return res

    def chars2idxs(self, sentence):
        return [self.char2idx.get(char, UNK_ID) for char in sentence]


def make_batch(label_former_latter_list):
    batchsize = len(label_former_latter_list)
    labels = []
    length_list = []

    for label, former_ids, latter_ids in label_former_latter_list:
        labels.append(label)
        length_list.append(len(former_ids))
        length_list.append(len(latter_ids))

    maxlen = max(length_list)

    labels = torch.tensor(labels)
    former_ids = PAD_ID * torch.ones((batchsize, maxlen), dtype=torch.long)
    latter_ids = PAD_ID * torch.ones((batchsize, maxlen), dtype=torch.long)

    for idx, (_, former_id, latter_id) in enumerate(label_former_latter_list):
        former_ids[idx, :len(former_id)] = torch.tensor(former_id)
        latter_ids[idx, :len(latter_id)] = torch.tensor(latter_id)

    return labels, former_ids, latter_ids


class EarlyStopping():
    def __init__(self, max_patient=5):
        self.best = -1
        self.stack = 0
        self.max_patient = max_patient

    def update(self, score):
        # When score is improved.
        if score > self.best:
            self.best = score
            self.stack = 0
            return False

        # When score is not improved.
        self.stack += 1
        if self.stack >= self.max_patient:
            return True

        return False
