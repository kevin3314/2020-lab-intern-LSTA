from collections import Counter
import pickle


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


if __name__ == '__main__':
    DATA = 'data/original/Train_Data_F.pickle'

    with open(DATA, 'rb') as f:
        sentences = pickle.load(f)
    char2idx = count_char(sentences)
    print(char2idx)
