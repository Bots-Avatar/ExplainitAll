from Metric_rouge import Metric_rouge

from rouge_N import rouge_N


class Metric_rouge_N(Metric_rouge):
    # CONSTRUCTOR / DESTRUCTOR:
    
    def __init__(self, n):
        self.n_ = n
    
    def __del__(self):
        del self.n_
    
    # FUNCTIONS:
    
    def calculate(self, references_encodings, candidates_encodings):
        res = []
        
        for i, _ in enumerate(references_encodings):
            res.append(rouge_N(references_encodings[i], candidates_encodings[i], self.n_))
            
        return res
    
    # FIELDS:

    n_ = None
