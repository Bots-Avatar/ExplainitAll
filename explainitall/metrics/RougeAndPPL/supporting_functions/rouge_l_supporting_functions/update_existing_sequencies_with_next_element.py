from .unite_sequencies import unite_sequencies
from .find_lowest_position_higher_then_current import find_lowest_position_higher_then_current


def update_existing_sequencies_with_next_element(existing_sequencies, element):
    
    new_sequencies = [[element[0], 1]]
    
    for existing_seq in existing_sequencies:
        appropriate_ind = find_lowest_position_higher_then_current(element, existing_seq[0])
        if appropriate_ind is not None:
            new_sequencies = unite_sequencies(new_sequencies, [[appropriate_ind, existing_seq[1] + 1]])
            
    return unite_sequencies(existing_sequencies, new_sequencies)
