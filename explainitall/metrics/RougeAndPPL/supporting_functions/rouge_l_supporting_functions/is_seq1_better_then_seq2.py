def is_seq1_better_then_seq2(seq1, seq2):
    #if length of seq1 sequence lower then seq2 sequence length
    if seq1[1] < seq2[1]:
        return False
    
    # if index of seq1 lower then index in seq2
    if seq1[0] < seq2[0]:
        return True
    
    return False
