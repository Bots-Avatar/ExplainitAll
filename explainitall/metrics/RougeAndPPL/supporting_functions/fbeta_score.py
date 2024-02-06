def fbeta_score(precision, recall, beta=1):
    if precision == 0.0 and recall == 0.0:
        return 0.0
    
    if beta == 1:
        return 2 * precision * recall / (precision + recall)
    return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
