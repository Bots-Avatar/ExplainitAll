from Metric import Metric


class Metric_standart(Metric):
    # CONSTRUCTOR / DESTRUCTOR:
    
    def __init__(self, metric_function):
        self.metric_function_ = metric_function
    
    def __del__(self):
        del self.metric_function_
    
    # FUNCTIONS:
    
    def calculate(self, references_encodings, candidates_encodings):
        res = []
        
        for i, _ in enumerate(references_encodings):
            res.append(self.metric_function_(references_encodings[i], candidates_encodings[i]))
            
        return res
    
    # FIELDS:

    metric_function_ = None
