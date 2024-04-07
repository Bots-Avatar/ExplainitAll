from itertools import islice


def split_into_overlapping_chunks(iterable, chunk_size):
    iterator = iter(iterable)
    res = tuple(islice(iterator, chunk_size))
    if len(res) == chunk_size:
        yield res
    for el in iterator:
        res = res[1:] + (el,)
        yield res


def list_intersection(l1, l2):
    """Подсчитывает и возвращает количество общих элементов между двумя списками (l1 и l2),
    удаляя найденные совпадения из второго списка.
    Используется когда необходимо определить количество уникальных совпадений между двумя наборами данных,
    не учитывая повторения во втором наборе.
    """
    l1_copy = l1[:]
    l2_copy = l2[:]

    res = 0
    for el in l1_copy:
        i = len(l2_copy) - 1
        while i >= 0:
            if el == l2_copy[i]:
                del l2_copy[i]
                res += 1
                break
            i -= 1

    return res
