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


# ------------------------------------------------------- aggregate_cluster_df

import pandas as pd  # noqa: E402

from explainitall.clusters import aggregate_cluster_df  # noqa: E402


def old_aggregate_cluster_df(cluster_df, aggr_f, drop_condition='unnamed', cl_names_col='Tokens'):
    """Прежняя реализация: верна, только если имена столбцов уникальны"""
    cluster_df = cluster_df[~cluster_df[cl_names_col].str.contains(drop_condition)]
    cluster_df = cluster_df.loc[:, ~cluster_df.columns.str.contains(drop_condition)]
    aggregation_functions = {col: aggr_f for col in cluster_df.columns if col != cl_names_col}
    return cluster_df.groupby(cl_names_col).agg(aggregation_functions).reset_index()


def cluster_frame(columns, rows):
    df = pd.DataFrame([values for _, values in rows])
    df.columns = columns
    df.insert(0, 'Tokens', [name for name, _ in rows])
    return df


@pytest.mark.parametrize('aggr_f', ['mean', 'max', 'min', 'sum', 'median', 'std'])
def test_aggregate_matches_old_version_on_unique_columns(aggr_f):
    df = cluster_frame(['Лекарства', 'unnamed', 'Болезни'],
                       [('Животные', [0.1, 0.9, 0.3]), ('unnamed', [0.5, 0.5, 0.5]), ('Болезни', [0.7, 0.2, 0.4]),
                        ('Животные', [0.3, 0.1, np.nan]), ('Лекарства', [0.8, 0.6, 0.2])])

    pd.testing.assert_frame_equal(aggregate_cluster_df(df, aggr_f), old_aggregate_cluster_df(df, aggr_f))


def test_aggregate_merges_columns_of_the_same_cluster():
    df = cluster_frame(['Лекарства', 'unnamed', 'Лекарства'],
                       [('Животные', [0.2, 0.9, 0.4]), ('Болезни', [0.6, 0.1, 1.0]), ('Животные', [0.0, 0.3, np.nan])])

    result = aggregate_cluster_df(df, 'mean').set_index('Tokens')

    assert list(result.columns) == ['Лекарства']
    assert result.loc['Животные', 'Лекарства'] == pytest.approx((0.2 + 0.4 + 0.0) / 3)
    assert result.loc['Болезни', 'Лекарства'] == pytest.approx(0.8)
    assert aggregate_cluster_df(df, 'max').set_index('Tokens').loc['Животные', 'Лекарства'] == 0.4


def test_aggregate_drops_only_exact_unnamed_label():
    df = cluster_frame(['unnamed_symptoms', 'unnamed'], [('unnamed_symptoms', [0.5, 0.1]), ('unnamed', [0.9, 0.9])])

    result = aggregate_cluster_df(df, 'mean')

    assert list(result['Tokens']) == ['unnamed_symptoms']
    assert list(result.columns) == ['Tokens', 'unnamed_symptoms']


def test_aggregate_keeps_column_order_of_first_appearance():
    df = cluster_frame(['Лекарства', 'Болезни', 'Лекарства', 'Животные'], [('Болезни', [0.1, 0.2, 0.3, 0.4])])

    assert list(aggregate_cluster_df(df, 'mean').columns) == ['Tokens', 'Лекарства', 'Болезни', 'Животные']


@pytest.mark.parametrize('columns,rows', [
    (['unnamed', 'unnamed'], [('Животные', [0.1, 0.2])]),
    (['Лекарства'], [('unnamed', [0.5])]),
])
def test_aggregate_without_clusters_returns_empty_frame(columns, rows):
    result = aggregate_cluster_df(cluster_frame(columns, rows), 'mean')

    assert result.empty and list(result.columns) == ['Tokens']


def test_aggregate_answer_with_repeated_clusters():
    """
    Таблица в формате ExplainerGPT2.cluster_imp_df для запроса «кошка грипп» и ответа
    «Кошке и собаке помогут уколы»: два слова кластера Животные дают одинаковые столбцы
    """
    tokens_generated = ('Животные', 'unnamed', 'Животные', 'unnamed', 'Лекарства')
    array = np.arange(7 * 5, dtype=float).reshape(7, 5)
    cluster_df = pd.DataFrame(array, columns=tokens_generated)
    cluster_df.insert(0, 'Tokens', ['Животные', 'Болезни', 'Животные', 'unnamed', 'Животные', 'unnamed', 'Лекарства'])

    result = aggregate_cluster_df(cluster_df, 'mean').set_index('Tokens')

    assert set(result.index) == {'Животные', 'Болезни', 'Лекарства'}
    assert list(result.columns) == ['Животные', 'Лекарства']
    assert result.loc['Болезни', 'Животные'] == pytest.approx((array[1, 0] + array[1, 2]) / 2)
