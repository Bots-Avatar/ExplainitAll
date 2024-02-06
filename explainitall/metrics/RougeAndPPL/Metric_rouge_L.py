from Metric_rouge import Metric_rouge

from rouge_L import rouge_L


class Metric_rouge_L(Metric_rouge):
    # CONSTRUCTOR / DESTRUCTOR:

    def __init__(self):
        pass

    def __del__(self):
        pass

    # FUNCTIONS:

    def calculate(self, references_encodings, candidates_encodings):
        res = []

        for i, _ in enumerate(references_encodings):
            res.append(rouge_L(references_encodings[i], candidates_encodings[i]))

        return res

    # FIELDS:
