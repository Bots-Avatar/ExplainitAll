def list_intersection(l1, l2):
    res = 0
    
    # count amount of  same elements in l1 and l2:
    # iterate through elements of l1 and iterate backwards (to have opportunity to remove elements from l2 in cycle)
    # to serch for equal element (remove it in this case)
    for el in l1:
        i = len(l2) - 1
        while i >= 0:
            if el == l2[i]:
                del l2[i]
                res += 1
                break
            i -= 1
    
    return res
