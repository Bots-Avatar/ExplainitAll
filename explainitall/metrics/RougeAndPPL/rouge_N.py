from supporting_functions import split_into_overlapping_chunks
from supporting_functions import list_intersection
from supporting_functions import fbeta_score

from typing import List


def rouge_N(reference: List[int], candidate: List[int], n):
    reference_chunks = list(split_into_overlapping_chunks(reference, n))
    reference_length = len(reference_chunks)
    
    candidate_chunks = list(split_into_overlapping_chunks(candidate, n))
    candidate_length = len(candidate_chunks)
    
    if candidate_length == 0 or reference_length == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    intersection = list_intersection(reference_chunks, candidate_chunks)
    
    precision = intersection / candidate_length
    recall = intersection / reference_length
    f1 = fbeta_score(precision, recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
