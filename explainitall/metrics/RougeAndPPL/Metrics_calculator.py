class Metrics_calculator:
    metrics_ = None
    preprocessing_functions_ = None
    tokenizer_ = None

    def __init__(self, tokenizer):
        self.metrics_ = {}
        self.preprocessing_functions_ = {}
        self.tokenizer_ = tokenizer

    def __del__(self):
        del self.metrics_
        del self.tokenizer_

    def calculate(self, contexts, references, candidates):
        res = {}

        assert len(contexts) == len(references) == len(candidates)

        preprocessed_sentences = {}
        for preproc in self.preprocessing_functions_:
            preprocessed = preproc(contexts, references, candidates, self.tokenizer_)
            preprocessed_sentences[frozenset(self.preprocessing_functions_[preproc])] = preprocessed

        for metric_name in self.metrics_:
            for metric_group in preprocessed_sentences:
                if metric_name in metric_group:
                    prep = preprocessed_sentences[metric_group]
                    prep_references = prep['references']
                    prep_candidates = prep['candidates']
                    res[metric_name] = self.metrics_[metric_name].calculate(prep_references, prep_candidates)
                    break
        return res

    def add_metric(self, name, metric):
        if name in self.metrics_:
            raise Exception('Metric name ' + name + ' already exists!')
        self.metrics_[name] = metric

        preproc = metric.preprocess
        if preproc not in self.preprocessing_functions_:
            self.preprocessing_functions_[preproc] = set()
        self.preprocessing_functions_[preproc].add(name)
