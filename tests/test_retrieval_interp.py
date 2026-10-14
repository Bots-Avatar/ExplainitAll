import re

import numpy as np
import pytest

from explainitall.api_interp.cache import ResponseCache
from explainitall.api_interp.embedding_interp import APIEmbedder, EmbedderExplainer
from explainitall.api_interp.openrouter_client import (EmbeddingResult, OpenRouterClient, OpenRouterError,
                                                       RerankResult)
from explainitall.api_interp.rerank_interp import RerankerExplainer
from tests.test_openrouter_client import FakeResponse, FakeSession, URL

CLUSTERS = [{'name': 'Животные', 'centroid': ['кошка'], 'top_k': 1},
            {'name': 'Болезни', 'centroid': ['грипп'], 'top_k': 1}]


def stem(word):
    return word.lower()[:4]


class StemEmbedder:
    """Мешок основ слов: одинаковые основы дают общее измерение"""

    def __init__(self, dim=256):
        self.dim = dim
        self.vocab = {}
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for word in re.findall(r'\w+', text):
                vectors[i, self.vocab.setdefault(stem(word), len(self.vocab))] += 1
        return vectors


class OverlapReranker:
    """Релевантность — доля основ слов запроса, встречающихся в документе"""

    def __init__(self, normalize_in_request=False):
        self.normalize_in_request = normalize_in_request
        self.calls = []

    def rerank(self, query, documents, model, provider=None, **kwargs):
        self.calls.append((query, list(documents)))
        query_stems = {stem(w) for w in re.findall(r'\w+', query)}
        scores = []
        for document in documents:
            document_stems = {stem(w) for w in re.findall(r'\w+', document)}
            scores.append(len(query_stems & document_stems) / max(len(query_stems), 1))
        if self.normalize_in_request and max(scores) > 0:
            scores = [s / max(scores) for s in scores]
        return RerankResult(scores=scores, model=model, provider='Cohere', usage={}, cost=0.002, cached=False)


@pytest.fixture(scope="module")
def nlp_model():
    gensim = pytest.importorskip("gensim")
    pytest.importorskip("pymorphy3")
    words = {'кошка_NOUN': [1, 0, 0, 0], 'собака_NOUN': [0.9, 0.1, 0, 0],
             'грипп_NOUN': [0, 1, 0, 0], 'простуда_NOUN': [0, 0.9, 0.1, 0]}
    kv = gensim.models.KeyedVectors(vector_size=4)
    kv.add_vectors(list(words), np.array(list(words.values()), dtype=np.float32))
    return kv


# ------------------------------------------------------------------ client

def embeddings_response(vectors, cost=0.0001, shuffle=False):
    data = [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)]
    if shuffle:
        data = data[::-1]
    return FakeResponse(200, {'object': 'list', 'model': 'baai/bge-m3', 'data': data,
                              'usage': {'prompt_tokens': 8, 'total_tokens': 8, 'cost': cost}})


def rerank_response(scores, cost=0.001):
    results = sorted(({'index': i, 'relevance_score': s, 'document': {'text': f'd{i}'}} for i, s in scores.items()),
                     key=lambda r: -r['relevance_score'])
    return FakeResponse(200, {'id': 'gen-rerank-1', 'model': 'cohere/rerank-v3.5', 'provider': 'Cohere',
                              'results': results, 'usage': {'search_units': 1, 'total_tokens': 30, 'cost': cost}})


def make_client(*responses, **kwargs):
    session = FakeSession(*responses)
    return OpenRouterClient(api_key='test-key', session=session, sleep=lambda s: None, **kwargs), session


def test_embeddings_request_and_order():
    client, session = make_client(embeddings_response([[1.0, 0.0], [0.0, 1.0]], shuffle=True))

    result = client.embeddings(['кошка', 'грипп'], 'baai/bge-m3', dimensions=2, input_type='search_document')

    call = session.calls[0]
    assert call['url'] == URL + '/embeddings'
    assert call['json'] == {'model': 'baai/bge-m3', 'input': ['кошка', 'грипп'], 'encoding_format': 'float',
                            'dimensions': 2, 'input_type': 'search_document'}
    np.testing.assert_array_equal(result.vectors, [[1.0, 0.0], [0.0, 1.0]])
    assert result.cost == pytest.approx(0.0001) and client.total_cost == pytest.approx(0.0001)


def test_embeddings_validation():
    client, _ = make_client(embeddings_response([[1.0]]))

    with pytest.raises(ValueError):
        client.embeddings(['кошка', ''], 'm/x')
    with pytest.raises(OpenRouterError):
        client.embeddings(['кошка', 'грипп'], 'm/x')  # пришёл один вектор на два текста
    with pytest.raises(OpenRouterError):
        EmbeddingResult.from_response({'data': [{'index': 0, 'embedding': 'AAAA'}]}, 1)


def test_rerank_scores_follow_input_order():
    client, session = make_client(rerank_response({0: 0.1, 1: 0.9, 2: 0.5}))

    result = client.rerank('столица Франции', ['Берлин', 'Париж', 'Лион'], 'cohere/rerank-v3.5', top_n=3)

    assert session.calls[0]['url'] == URL + '/rerank'
    assert session.calls[0]['json'] == {'model': 'cohere/rerank-v3.5', 'query': 'столица Франции',
                                        'documents': ['Берлин', 'Париж', 'Лион'], 'top_n': 3}
    assert result.scores == [0.1, 0.9, 0.5]
    assert result.ranking == [1, 2, 0]
    assert result.provider == 'Cohere' and result.cost == pytest.approx(0.001)


def test_rerank_with_top_n_leaves_missing_scores_nan():
    client, _ = make_client(rerank_response({1: 0.9}))

    result = client.rerank('q', ['a', 'b', 'c'], 'm/x', top_n=1)

    assert np.isnan(result.scores[0]) and result.scores[1] == 0.9
    assert result.ranking == [1]


def test_rerank_rejects_bad_index():
    with pytest.raises(OpenRouterError):
        RerankResult.from_response({'results': [{'index': 5, 'relevance_score': 0.1}]}, 2)


def test_embeddings_and_rerank_are_cached_separately():
    client, session = make_client(embeddings_response([[1.0]]), rerank_response({0: 0.5}), cache=ResponseCache())

    for _ in range(2):
        client.embeddings('кошка', 'm/x')
        client.rerank('кошка', ['кошка'], 'm/x')

    assert len(session.calls) == 2
    assert client.n_cache_hits == 2


def test_list_models_by_output_modality():
    client, session = make_client(FakeResponse(200, {'data': [{'id': 'cohere/rerank-v3.5'}]}))

    client.list_models(output_modalities=['rerank'])

    assert session.calls[0]['params'] == {'output_modalities': 'rerank'}


# ---------------------------------------------------------------- embedder

class FakeEmbeddingsClient:
    def __init__(self):
        self.calls = []
        self.embedder = StemEmbedder()

    def embeddings(self, inputs, model, dimensions=None, input_type=None, provider=None):
        self.calls.append({'inputs': list(inputs), 'input_type': input_type})
        return EmbeddingResult(vectors=self.embedder.encode(inputs), model=model, provider=None, usage={},
                               cost=0.0001, cached=False)


def test_api_embedder_batches_requests():
    client = FakeEmbeddingsClient()
    embedder = APIEmbedder(client, 'baai/bge-m3', batch_size=2, input_type='search_query')

    vectors = embedder.encode(['a', 'b', 'c', 'd', 'e'])
    single = embedder.encode('a')

    assert vectors.shape[0] == 5 and single.ndim == 1
    assert [len(c['inputs']) for c in client.calls] == [2, 2, 1, 1]
    assert client.calls[0]['input_type'] == 'search_query'
    assert embedder.n_requests == 4 and embedder.total_cost == pytest.approx(0.0004)


def test_word_importance_uses_one_batch():
    embedder = StemEmbedder()
    df = EmbedderExplainer(embedder).word_importance('Кошка болеет гриппом')

    assert list(df['word']) == ['Кошка', 'болеет', 'гриппом']
    assert (df['importance'] > 0).all()
    assert df['share'].sum() == pytest.approx(1.0)
    assert len(embedder.calls) == 1 and len(embedder.calls[0]) == 4


def test_repeated_word_matters_less():
    df = EmbedderExplainer(StemEmbedder()).word_importance('грипп грипп грипп кошка')

    assert df.loc[3, 'importance'] > df.loc[0, 'importance']


def test_similarity_attribution_signs():
    explanation = EmbedderExplainer(StemEmbedder()).similarity_attribution('чем лечить грипп',
                                                                          'грипп лечат уколами')
    contributions = explanation.contributions.set_index(['side', 'label'])['contribution']

    assert 0 < explanation.similarity < 1
    assert contributions[('document', 'грипп')] > 0  # общее слово поддерживает близость
    assert contributions[('document', 'уколами')] < 0  # слово только из документа её размывает
    assert contributions[('query', 'грипп')] > 0
    assert explanation.top(1)['label'].iloc[0] in ('грипп', 'лечить', 'лечат')


def test_similarity_attribution_side_and_asymmetric_embedder():
    document_embedder, query_embedder = StemEmbedder(), StemEmbedder()
    query_embedder.vocab = document_embedder.vocab
    explainer = EmbedderExplainer(document_embedder, query_embedder=query_embedder)

    explanation = explainer.similarity_attribution('лечить грипп', 'грипп лечат', side='document')

    assert set(explanation.contributions['side']) == {'document'}
    assert query_embedder.calls == [['лечить грипп']]
    assert document_embedder.calls == [['грипп лечат', 'лечат', 'грипп']]
    with pytest.raises(ValueError):
        explainer.similarity_attribution('a', 'b', side='left')


def test_cluster_importance_and_similarity(nlp_model):
    explainer = EmbedderExplainer(StemEmbedder(), nlp_model=nlp_model)

    importance = explainer.cluster_importance('У кошки грипп, кошку жалко', CLUSTERS)
    assert set(importance['cluster']) == {'Животные', 'Болезни'}
    assert importance.set_index('cluster').loc['Животные', 'words'] == ['кошки', 'кошку']

    explanation = explainer.cluster_similarity_attribution('лечение кошки от гриппа', 'кошке помогут от гриппа',
                                                           CLUSTERS)
    assert set(explanation.contributions['label']) == {'Животные', 'Болезни'}
    assert (explanation.contributions['contribution'] > 0).all()


def test_cluster_importance_requires_nlp_model():
    with pytest.raises(ValueError, match='nlp_model'):
        EmbedderExplainer(StemEmbedder()).cluster_importance('кошка', CLUSTERS)


# ---------------------------------------------------------------- reranker

def test_document_word_importance_in_single_request():
    reranker = OverlapReranker()
    explainer = RerankerExplainer(reranker, 'cohere/rerank-v3.5')

    df = explainer.document_word_importance('лечение гриппа у кошки', 'Кошке при гриппе помогут уколы')

    assert len(reranker.calls) == 1
    assert reranker.calls[0][1][0] == 'Кошке при гриппе помогут уколы'
    importance = df.set_index('word')['importance']
    assert importance['гриппе'] > 0 and importance['Кошке'] > 0
    assert importance['уколы'] == 0
    assert explainer.n_requests == 1 and explainer.total_cost == pytest.approx(0.002)


def test_document_importance_chunks_keep_reference():
    reranker = OverlapReranker(normalize_in_request=True)
    explainer = RerankerExplainer(reranker, 'm/x', max_documents_per_request=3)

    df = explainer.document_word_importance('грипп кошка', 'кошка болеет гриппом сильно')

    assert len(reranker.calls) == 2
    assert all(call[1][0] == 'кошка болеет гриппом сильно' for call in reranker.calls)
    assert (df.set_index('word').loc[['кошка', 'гриппом'], 'importance'] > 0).all()


def test_query_word_importance_and_top1_change():
    reranker = OverlapReranker()
    explainer = RerankerExplainer(reranker, 'm/x')
    documents = ['кошка ест корм', 'грипп у кошки', 'погода сегодня']

    df = explainer.query_word_importance('кошка грипп', documents).set_index('word')

    assert len(reranker.calls) == 3
    assert df.loc['грипп', 'doc_0'] == pytest.approx(0.5 - 1.0)
    assert df.loc['грипп', 'doc_1'] == pytest.approx(0.0)
    assert df.loc['грипп', 'top1_changed']
    assert not df.loc['кошка', 'top1_changed']


def test_query_cluster_importance(nlp_model):
    explainer = RerankerExplainer(OverlapReranker(), 'm/x', nlp_model=nlp_model)

    df = explainer.query_cluster_importance('кошка и грипп', ['грипп у кошки', 'кошка ест'], CLUSTERS)

    assert list(df.columns[:2]) == ['cluster', 'words']
    assert set(df['cluster']) == {'Животные', 'Болезни'}


def test_reranker_settings_validation():
    with pytest.raises(ValueError):
        RerankerExplainer(OverlapReranker(), 'm/x', max_documents_per_request=1)
    with pytest.raises(ValueError):
        RerankerExplainer(OverlapReranker(), 'm/x').document_word_importance('q', '...')
