from .is_seq1_better_then_seq2 import is_seq1_better_then_seq2


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
