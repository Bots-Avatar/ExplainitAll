import numpy as np
import pytest

gensim = pytest.importorskip("gensim")
pytest.importorskip("pymorphy3")

from explainitall.clusters import ClusterManager
from explainitall.nlp import WordProcessor

VECTORS = {
    'кошка_NOUN': [1.0, 0.0, 0.0, 0.0],
    'собака_NOUN': [0.9, 0.1, 0.0, 0.0],
    'мяукать_VERB': [0.8, 0.2, 0.0, 0.0],
    'лечить_VERB': [0.0, 1.0, 0.0, 0.0],
    'колоть_VERB': [0.1, 0.9, 0.0, 0.0],
    'укол_NOUN': [0.2, 0.8, 0.0, 0.0],
    'красный_ADJ': [0.0, 0.0, 1.0, 0.0],
    'синий_ADJ': [0.0, 0.0, 0.9, 0.1],
    'быстро_ADV': [0.0, 0.0, 0.0, 1.0],
    'москва_PROPN': [0.0, 0.0, 0.1, 0.9],
    'хороший_ADJ': [0.0, 0.0, 0.5, 0.5],
}


@pytest.fixture(scope="module")
def embeddings():
    kv = gensim.models.KeyedVectors(vector_size=4)
    kv.add_vectors(list(VECTORS), np.array(list(VECTORS.values()), dtype=np.float32))
    return kv


@pytest.fixture(scope="module")
def manager(embeddings):
    return ClusterManager(embeddings=embeddings)


def build(manager, centroid, top_k=3, pos=None):
    descr = {'name': 'Кластер', 'centroid': centroid, 'top_k': top_k}
    if pos is not None:
        descr['pos'] = pos
    return manager.create_clusters([descr])[0]


@pytest.mark.parametrize("word,expected", [
    ('кошки', 'кошка_NOUN'),
    ('лечите', 'лечить_VERB'),
    ('красного', 'красный_ADJ'),
    ('лучше', 'хороший_ADJ'),
    ('быстро', 'быстро_ADV'),
    ('Москва', 'москва_PROPN'),
    ('абракадабра', None),
])
def test_embeddable_word_uses_upos_tags(embeddings, word, expected):
    assert WordProcessor(embeddings).get_embeddable_word_or_none(word) == expected


def test_noun_cluster_keeps_nouns_only_by_default(manager):
    cluster = build(manager, ['кошка'])

    assert cluster.words[0] == 'кошка'
    assert 'собака' in cluster.words
    assert 'мяукать' not in cluster.words


def test_verb_seed_words_are_kept(manager):
    cluster = build(manager, ['лечить'])

    assert cluster.words[:2] == ['лечить', 'колоть']
    assert 'укол' not in cluster.words
    assert manager.find_cluster_name('лечите', [cluster]) == 'Кластер'
    assert manager.find_cluster_name('колола', [cluster]) == 'Кластер'


def test_adjective_seed_words_are_kept(manager):
    cluster = build(manager, ['красный'], top_k=1)

    assert cluster.words == ['красный', 'синий']
    assert manager.find_cluster_name('красного', [cluster]) == 'Кластер'


def test_mixed_seed_words_allow_their_parts_of_speech(manager):
    cluster = build(manager, ['кошка', 'лечить'], top_k=4)

    assert {'кошка', 'лечить', 'колоть', 'укол', 'собака', 'мяукать'} >= set(cluster.words)
    assert {'кошка', 'лечить'} <= set(cluster.words)


def test_pos_all_includes_any_part_of_speech(manager):
    cluster = build(manager, ['кошка'], pos='all')

    assert 'мяукать' in cluster.words


def test_pos_list_restricts_neighbours_but_keeps_seed(manager):
    cluster = build(manager, ['кошка'], pos=['VERB'])

    assert cluster.words[0] == 'кошка'
    assert 'мяукать' in cluster.words
    assert 'собака' not in cluster.words


def test_cluster_words_are_unique(manager):
    cluster = build(manager, ['кошка', 'кошки'], top_k=5, pos='all')

    assert len(cluster.words) == len(set(cluster.words))
