from itertools import islice


def split_into_overlapping_chunks(iterable, chunk_size):
    iterator = iter(iterable)
    res = tuple(islice(iterator, chunk_size))
    if len(res) == chunk_size:
        yield res
    for el in iterator:
        res = res[1:] + (el, )
        yield res
