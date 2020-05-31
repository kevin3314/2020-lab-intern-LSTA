from collections import Counter
import pickle

import torch


def count_char(sentences):
    sentences = ''.join(sentences)
    co = Counter(sentences)

    sorted_co = sorted(co.items(), key=lambda x: x[1], reverse=True)
    print(sorted_co)
    sorted_co = [v for v in sorted_co if v[1] >= 2]

    char2idx = {}
    for i, v in enumerate(sorted_co):
        char2idx[v[0]] = i

    return char2idx


def get_init_embedding(char2idx, w_dim=128):
    num_words = len(char2idx) + 2  # UNK
    id_max = max(char2idx.values())
    char2idx['UNK'] = id_max + 1
    char2idx['PAD'] = id_max + 2

    emb = torch.randn(num_words, w_dim)
    return emb


if __name__ == '__main__':
    DATA = 'data/original/Train_Data_F.pickle'

    with open(DATA, 'rb') as f:
        sentences = pickle.load(f)
    char2idx = count_char(sentences)

    init_emb = get_init_embedding(char2idx)
    print(init_emb.shape)
    print(char2idx)
