from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict

import torch

from explainitall.metrics.RougeAndPPL.helpers import get_all_words_from_text
from explainitall.metrics.RougeAndPPL.rouge_L import rouge_L
from explainitall.metrics.RougeAndPPL.rouge_N import rouge_N


class Metric(ABC):
    @staticmethod
    def preprocess(contexts, references, candidates, tokenizer):
        raise NotImplementedError

    @abstractmethod
    def calculate(self, references_encodings, candidates_encodings):
        raise NotImplementedError


class Metric_rouge(Metric):

    def __init__(self, n):
        self.n = n

    @staticmethod
    def preprocess(contexts, references, candidates, tokenizer):
        res = {'references': [], 'candidates': []}
        for i, context in enumerate(contexts):
            res['references'].append(get_all_words_from_text(references[i][len(contexts[i]):]))
            res['candidates'].append(get_all_words_from_text(candidates[i][len(contexts[i]):]))
        return res


class MetricRougeL(Metric_rouge):

    def calculate(self, references_encodings, candidates_encodings):
        res = [rouge_L(reference, candidate) for reference, candidate in
               zip(references_encodings, candidates_encodings)]
        return res


class MetricRougeN(Metric_rouge):

    def calculate(self, references_encodings, candidates_encodings):
        res = [rouge_N(reference, candidate, self.n) for reference, candidate in
               zip(references_encodings, candidates_encodings)]
        return res


class MetricStandard(Metric):

    def __init__(self, metric_function):
        self.metric_function = metric_function

    def calculate(self, references_encodings, candidates_encodings):
        res = [self.metric_function(reference, candidate) for reference, candidate in
               zip(references_encodings, candidates_encodings)]
        return res


class Metric_ppl(Metric):

    def __init__(self, model: PreTrainedModel, stride: int):
        
        self.model = model
        self.stride = stride

    @staticmethod
    def preprocess(contexts: List[str], references: List[str], candidates: List[str], tokenizer: PreTrainedTokenizer) -> Dict[str, List[List[int]]]:
        """
        Предобработка данных
        """
        tokenized_data = {'references': [], 'candidates': []}

        for context, reference in zip(contexts, references):
            encoded_reference = tokenizer(reference)
            tokenized_data['references'].append(encoded_reference.input_ids)

        return tokenized_data

    def calculate(self, reference_encodings: List[List[int]], candidate_encodings: List[List[int]]) -> List[Dict[str, float]]:
        """
        Вычисление ppl для списка токенизированных текстов
        """
        perplexities = [self._calculate_perplexity(encodings) for encodings in reference_encodings]
        return perplexities

    def _calculate_perplexity(self, encodings: List[int]) -> Dict[str, float]:
        """
        Вычисление перплексии для одного токенизированного текста
        """
        max_length = self.model.config.n_positions
        sequence_length = len(encodings)

        neg_log_likelihoods = []
        prev_end_loc = 0

        for start_loc in range(0, sequence_length, self.stride):
            end_loc = min(start_loc + max_length, sequence_length)
            target_length = end_loc - prev_end_loc

            input_ids = torch.tensor(encodings[start_loc:end_loc], device=self.model.device).unsqueeze(0)
            target_ids = input_ids.clone()
            target_ids[:, :-target_length] = -1e-3

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * target_length

            neg_log_likelihoods.append(neg_log_likelihood)
            prev_end_loc = end_loc

            if end_loc == sequence_length:
                break

        total_neg_log_likelihood = torch.stack(neg_log_likelihoods).sum()
        perplexity = torch.exp(total_neg_log_likelihood / sequence_length)
        return {'value': perplexity.item()}
