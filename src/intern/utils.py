import pickle

from pyknp import Juman
import torch

from consts import UNK_ID, PAD_ID


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, char2idx):
        self.data, self.target = self.load_data(data_path)
        self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        ids = self.chars2idxs(data)

        return ids, target

    def load_data(self, path):
        with open(path, 'rb') as f:
            res = pickle.load(f)

        return res

    def chars2idxs(self, sentence):
        return [self.char2idx.get(char, UNK_ID) for char in sentence]


# def make_batch(idseq_list, target):
def make_batch(ids_target_list):
    batchsize = len(ids_target_list)
    length_list = []

    for target, idseq_orig in ids_target_list:
        length_list.append(len(idseq_orig))

    length_list = torch.tensor(length_list, dtype=torch.long)
    maxlen = max(length_list)

    idseq = PAD_ID * torch.ones((batchsize, maxlen), dtype=torch.long)
    targets = 0 * torch.ones((batchsize, maxlen), dtype=torch.float)

    for idx, (idseq_orig, target) in enumerate(ids_target_list):
        idseq[idx, :len(idseq_orig)] = torch.tensor(idseq_orig)
        targets[idx, :len(target)] = torch.tensor(target)

    return idseq, length_list, targets


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
