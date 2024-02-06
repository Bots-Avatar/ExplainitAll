class Metric:
    # CONSTRUCTOR / DESTRUCTOR:
    
    def __init__(self):
        pass
    
    def __del__(self):
        pass
    
    # FUNCTIONS:

    @staticmethod
    def preprocess(contexts, references, candidates, tokenizer):
        pass
    
    def calculate(self, references_encodings, candidates_encodings):
        raise NotImplementedError()
    
    # FIELDS:
