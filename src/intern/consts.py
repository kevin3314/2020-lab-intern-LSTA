from collections import Counter
import pickle

import torch


def count_char(sentences):
    sentences = ''.join(sentences)
    co = Counter(sentences)

    sorted_co = sorted(co.items(), key=lambda x: x[1], reverse=True)
    sorted_co = [v for v in sorted_co if v[1] >= 2]

    char2idx = {}
    for i, v in enumerate(sorted_co):
        char2idx[v[0]] = i

    return char2idx


def get_init_embedding(char2idx, w_dim=128):
    voc_size = len(char2idx) + 2  # UNK
    id_max = max(char2idx.values())
    UNK_ID = id_max + 1
    PAD_ID = id_max + 2
    char2idx['UNK'] = UNK_ID
    char2idx['PAD'] = PAD_ID

    return UNK_ID, PAD_ID, voc_size


DATA = 'data/original/Train_Data_F.pickle'

with open(DATA, 'rb') as f:
    # (sentences, offsets)
    sentences = pickle.load(f)[0]
CHAR2IDX = count_char(sentences)

UNK_ID, PAD_ID, voc_size = get_init_embedding(CHAR2IDX)
