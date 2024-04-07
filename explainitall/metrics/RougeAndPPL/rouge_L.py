from typing import List

from explainitall.metrics.RougeAndPPL.helpers import fbeta_score
from explainitall.metrics.RougeAndPPL.rouge_l_helpers import get_element_positions, \
    update_existing_sequencies_with_next_element


def rouge_L(reference: List[int], candidate: List[int]):
    reference_length = len(reference)
    candidate_length = len(candidate)

    candidate_positions_in_reference = []
    for el in candidate:
        candidate_positions_in_reference.append(get_element_positions(reference, el))

    # remove elements not appeared in reference
    candidate_positions_in_reference = [x for x in candidate_positions_in_reference if x != []]

    existing_sequencies = []
    for element in candidate_positions_in_reference:
        existing_sequencies = update_existing_sequencies_with_next_element(existing_sequencies, element)

    if len(existing_sequencies) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    max_sequence_length = max(existing_sequencies, key=lambda x: x[1])[1]

    precision = max_sequence_length / candidate_length
    recall = max_sequence_length / reference_length
    f1 = fbeta_score(precision, recall)

    return {'precision': precision, 'recall': recall, 'f1': f1}
