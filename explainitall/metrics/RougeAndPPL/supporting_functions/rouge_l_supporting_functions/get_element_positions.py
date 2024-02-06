def get_element_positions(sequence, element):
    res = []
    for i, v in enumerate(sequence):
        if v == element:
            res.append(i)
            
    return res
