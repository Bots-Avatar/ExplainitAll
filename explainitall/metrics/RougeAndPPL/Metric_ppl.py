import copy

import torch

from Metric import Metric


class Metric_ppl(Metric):
    
    # CONSTRUCTOR / DESTRUCTOR:
    
    def __init__(self, model, stride):
        self.model_ = model
        self.stride_ = stride
    
    def __del__(self):
        del self.model_
        del self.stride_
    
    # FUNCTIONS:

    @staticmethod
    def preprocess(contexts, references, candidates, tokenizer):
        res = {}

        res['references'] = []
        res['candidates'] = []
        for i, v in enumerate(contexts):
            reference_encodings = tokenizer(references[i])
            res['references'].append(reference_encodings.input_ids)

        return res
    
    def calculate(self, references_encodings, candidates_encodings):
        res = []
        
        i = 0
        while i < len(references_encodings):
            res.append(self._calculate_for_one(references_encodings[i]))
            i += 1
            
        return res
    
    def _calculate_for_one(self, reference_encodings):
        max_length = self.model_.config.n_positions
        seq_len = len(reference_encodings)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride_):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = torch.tensor(reference_encodings[begin_loc:end_loc])
            input_ids = input_ids.to(self.model_.device)
            target_ids = input_ids.clone()
            target_ids[:-trg_len] = -100
            #target_ids = [-100] * (len(input_ids) - trg_len) + input_ids[trg_len:]

            with torch.no_grad():
                outputs = self.model_(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return {'value': ppl.item()}
    
    # FIELDS:
    
    model_ = None
    stride_ = None
    