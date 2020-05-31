from collections import Counter


def count_char(sentences):
    sentences = ''.join(sentences)
    co = Counter(sentences)
    co = [v for v in co if v[1] >= 2]
    sorted_co = sorted(co.items(), key=lambda x: x[1], reverse=True)
    return sorted_co
