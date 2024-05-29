from itertools import islice
from typing import List, Tuple, Iterator, Dict

def split_into_overlapping_chunks(iterable: List[int], chunk_size: int) -> Iterator[Tuple[int, ...]]:
    """
    Разбивает последовательность на перекрывающиеся блоки заданного размера
    """
    iterator = iter(iterable)
    res = tuple(islice(iterator, chunk_size))
    if len(res) == chunk_size:
        yield res
    for el in iterator:
        res = res[1:] + (el,)
        yield res

def list_intersection(l1: List[int], l2: List[int]) -> int:
    """
    Подсчитывает и возвращает количество общих элементов между двумя списками (l1 и l2),
    удаляя найденные совпадения из второго списка
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

def rouge_N(reference: List[int], candidate: List[int], n: int) -> Dict[str, float]:
    """
    Вычисляет Rouge-N оценку между эталонной последовательностью и кандидатом

    :param reference: Эталонная последовательность
    :param candidate: Кандидатская последовательность
    :param n: Размер n-грамм
    """
    reference_chunks = list(split_into_overlapping_chunks(reference, n))
    reference_length = len(reference_chunks)

    candidate_chunks = list(split_into_overlapping_chunks(candidate, n))
    candidate_length = len(candidate_chunks)

    if candidate_length == 0 or reference_length == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    intersection = list_intersection(reference_chunks, candidate_chunks)

    precision = intersection / candidate_length
    recall = intersection / reference_length
    f1 = 2*precision * recall / (precision + recall)

    return {'precision': precision, 'recall': recall, 'f1': f1}
