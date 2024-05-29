from typing import List, Dict, Optional

def find_lowest_position_higher_then_current(pos_list: List[int], current_position: Optional[int]) -> Optional[int]:
    """Находит наименьшую позицию в списке, которая больше текущей"""
    for pos in pos_list:
        if current_position is None or pos > current_position:
            return pos
    return None

def get_element_positions(sequence: List[int], element: int) -> List[int]:
    """Возвращает список позиций, на которых элемент встречается в последовательности"""
    return [i for i, v in enumerate(sequence) if v == element]

def is_seq1_better_then_seq2(seq1: List[int], seq2: List[int]) -> bool:
    """Определяет, является ли первая последовательность лучше второй"""
    if seq1[1] < seq2[1]:
        return False
    if seq1[0] < seq2[0]:
        return True
    return False

def unite_sequencies(existing_sequencies: List[List[int]], new_sequencies: List[List[int]]) -> List[List[int]]:
    """Объединяет существующие и новые последовательности, оставляя только лучшие"""
    i = len(existing_sequencies) - 1
    while i >= 0:
        g = len(new_sequencies) - 1
        while g >= 0:
            if is_seq1_better_then_seq2(existing_sequencies[i], new_sequencies[g]):
                del new_sequencies[g]
            elif is_seq1_better_then_seq2(new_sequencies[g], existing_sequencies[i]):
                del existing_sequencies[i]
                break
            g -= 1
        i -= 1
    return existing_sequencies + new_sequencies

def update_existing_sequencies_with_next_element(existing_sequencies: List[List[int]], element: List[int]) -> List[List[int]]:
    """Обновляет существующие последовательности с учетом нового элемента"""
    new_sequencies = [[element[0], 1]]
    for existing_seq in existing_sequencies:
        appropriate_ind = find_lowest_position_higher_then_current(element, existing_seq[0])
        if appropriate_ind is not None:
            new_sequencies = unite_sequencies(new_sequencies, [[appropriate_ind, existing_seq[1] + 1]])
    return unite_sequencies(existing_sequencies, new_sequencies)

def fbeta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """Вычисляет FBeta оценку (по-умолчанию F1) на основе precision и recall"""
    if precision == 0.0 and recall == 0.0:
        return 0.0
    if beta == 1.0:
        return 2 * precision * recall / (precision + recall)
    return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)

def rouge_L(reference: List[int], candidate: List[int]) -> Dict[str, float]:
    """Вычисляет Rouge-L оценку между эталонной и кандидатской последовательностями."""
    reference_length = len(reference)
    candidate_length = len(candidate)

    if candidate_length == 0 or reference_length == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    candidate_positions_in_reference = [get_element_positions(reference, el) for el in candidate]
    
    candidate_positions_in_reference = [positions for positions in candidate_positions_in_reference if positions]

    # Когда после фильтрации нет элементов
    if not candidate_positions_in_reference:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    existing_sequences = []
    for element_positions in candidate_positions_in_reference:
        existing_sequences = update_existing_sequencies_with_next_element(existing_sequences, element_positions)

    # Если нет существующих последовательностей
    if not existing_sequences:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    max_sequence_length = max(existing_sequences, key=lambda seq: seq[1])[1]

    precision = max_sequence_length / candidate_length
    recall = max_sequence_length / reference_length
    f1 = fbeta_score(precision, recall)

    return {'precision': precision, 'recall': recall, 'f1': f1}
