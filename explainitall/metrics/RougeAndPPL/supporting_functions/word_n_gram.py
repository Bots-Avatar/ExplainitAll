import re


def words_n_gramm(text, n_gramm=3):
    tx = text.replace('\n', ' ')
    tx = re.sub(r' +', ' ', tx)
    w = tx.split(' ')

    ng = []

    for i, word in enumerate(w):
        ng.append(' '.join(w[i:i + n_gramm]))

    return list(set(ng))
