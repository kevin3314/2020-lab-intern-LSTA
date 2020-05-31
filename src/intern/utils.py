import pickle

from pyknp import Juman
import torch

from consts import UNK_ID, PAD_ID


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, char2idx):
        self.data = self.load_data(data_path)
        self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        ids = self.chars2idxs(data)
        target = self.calc_position(data)

        return ids, target

    def load_data(self, path):
        with open(path, 'rb') as f:
            res = pickle.load(f)

        return res

    def chars2idxs(self, sentence):
        return [self.char2idx.get(char, UNK_ID) for char in sentence]

    def calc_position(self, sentence):
        juman = Juman()
        result = juman.analysis(sentence)
        current = 0
        offset = [0 for _ in range(len(sentence))]

        for mrph in result.mrph_list():
            current = current + len(mrph.midasi)
            try:
                offset[current-1] = 1

            except IndexError as e:
                print(sentence)
                print(current)
                for _mrph in result.mrph_list():
                    print(_mrph.midasi)
                raise e

            except Exception as e:
                raise e

        return offset


def make_batch(idseq_list, target):
    batchsize = len(idseq_list)
    length_list = torch.tensor(list(map(lambda x: len(x), idseq_list)))
    maxlen = max(length_list)

    idseq = PAD_ID * torch.ones((batchsize, maxlen), dtype=torch.long)
    target = torch.tensor(target, dtype=torch.long)

    for idx, idseq_orig in enumerate(idseq_list):
        idseq[idx, :len(idseq_orig)] = torch.tensor(idseq_orig)

    return idseq, length_list, target


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
