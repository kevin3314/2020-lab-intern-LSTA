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
