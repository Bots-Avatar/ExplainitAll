def find_lowest_position_higher_then_current(pos_list, current_position):
    for pos in pos_list:
        if current_position is None or pos > current_position:
            return pos


def get_element_positions(sequence, element):
    res = []
    for i, v in enumerate(sequence):
        if v == element:
            res.append(i)
    return res


def is_seq1_better_then_seq2(seq1, seq2):
    if seq1[1] < seq2[1]:
        return False
    if seq1[0] < seq2[0]:
        return True
    return False


def unite_sequencies(existing_sequencies, new_sequencies):
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

    res = existing_sequencies + new_sequencies
    return res


def update_existing_sequencies_with_next_element(existing_sequencies, element):
    new_sequencies = [[element[0], 1]]

    for existing_seq in existing_sequencies:
        appropriate_ind = find_lowest_position_higher_then_current(element, existing_seq[0])
        if appropriate_ind is not None:
            new_sequencies = unite_sequencies(new_sequencies, [[appropriate_ind, existing_seq[1] + 1]])

    return unite_sequencies(existing_sequencies, new_sequencies)
