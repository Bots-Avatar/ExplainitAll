def find_lowest_position_higher_then_current(pos_list, current_position):
    for pos in pos_list:
        if current_position is None or pos > current_position:
            return pos
    
    return None
