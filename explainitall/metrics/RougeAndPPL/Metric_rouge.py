from Metric import Metric

from supporting_functions import get_all_words_from_text


class Metric_rouge(Metric):
    # CONSTRUCTOR / DESTRUCTOR:

    def __init__(self, n):
        pass

    def __del__(self):
        pass

    # FUNCTIONS:

    @staticmethod
    def preprocess(contexts, references, candidates, tokenizer):
        res = {}

        res['references'] = []
        res['candidates'] = []
        for i, v in enumerate(contexts):
            res['references'].append(get_all_words_from_text(references[i][len(contexts[i]):]))
            res['candidates'].append(get_all_words_from_text(candidates[i][len(contexts[i]):]))

        return res

    # FIELDS:
