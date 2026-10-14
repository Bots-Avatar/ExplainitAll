import re

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from explainitall.metrics import CheckingForHallucinations as chk

DOC = 'Кошки любят молоко. Собаки охраняют дом.'
SUPPORTED = 'Кошки любят молоко.'
ANSWER = SUPPORTED + ' Луна сделана из сыра.'


class BagOfWordsModel:
    """Детерминированный эмбеддер: мешок слов вместо sbert"""

    def __init__(self, dim=64):
        self.dim = dim
        self.vocab = {}

    def encode(self, sentences):
        vectors = np.zeros((len(sentences), self.dim))
        for i, sentence in enumerate(sentences):
            for word in re.findall(r'\w+', sentence.lower()):
                vectors[i, self.vocab.setdefault(word, len(self.vocab))] += 1
        return vectors


class ExactMatchCrossEncoder:
    def __init__(self):
        self.received = None

    def predict(self, pairs):
        self.received = pairs
        return [1.0 if a == b else 0.0 for a, b in pairs]


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.setattr(chk.nltk, 'download', lambda *args, **kwargs: True)
    monkeypatch.setattr(chk, 'sent_tokenize', lambda text, language=None: re.split(r'(?<=\.)\s+', text.strip()))
    return chk.RAGHallucinationsChecker(BagOfWordsModel(), cross_encoder=ExactMatchCrossEncoder())


@pytest.mark.parametrize("metric", ['cosine', 'euclidean', 'manhattan', 'cross_encoder'])
def test_support_found_only_for_matching_sentence(checker, metric):
    res = checker.get_support_seq([DOC], ANSWER, sim_metric=metric)

    assert [r['answer'] for r in res] == [SUPPORTED]
    assert res[0]['reference_texts'] == [SUPPORTED]


@pytest.mark.parametrize("metric", ['cosine', 'euclidean', 'manhattan', 'cross_encoder'])
def test_hallucination_prob_partial_support(checker, metric):
    prob = checker.get_hallucinations_prob([DOC], ANSWER, sim_metric=metric)

    assert prob == pytest.approx(1 - len(SUPPORTED) / len(ANSWER))


def test_hallucination_prob_without_support_is_one(checker):
    assert checker.get_hallucinations_prob([DOC], 'Луна сделана из сыра.') == pytest.approx(1.0)


def test_hallucination_prob_full_support_is_zero(checker):
    assert checker.get_hallucinations_prob([DOC], DOC) == pytest.approx(0.0)


def test_hallucination_prob_empty_answer(checker):
    assert checker.get_hallucinations_prob([DOC], '') == 0.0


def test_cross_encoder_receives_pairs(checker):
    checker.get_support_seq([DOC], ANSWER, sim_metric='cross_encoder')

    assert all(isinstance(p, tuple) and len(p) == 2 for p in checker.cross_encoder.received)


def test_distance_similarities_are_pairwise():
    a = np.array([[0., 0.], [3., 4.], [1., 1.]])
    b = np.array([[0., 0.], [3., 4.]])

    np.testing.assert_allclose(chk.sim_euclidean(a, b), [[1, 1 / 6], [1 / 6, 1], [1 / 2.414213562, 1 / (1 + np.sqrt(13))]],
                               rtol=1e-6)
    np.testing.assert_allclose(chk.sim_manhattan(a, b), [[1, 1 / 8], [1 / 8, 1], [1 / 3, 1 / 6]])
