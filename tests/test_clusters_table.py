import pandas as pd
import pytest

from explainitall.gui.clusters_table import parse_centroid, make_clusters_from_dataframe, make_dataframe_from_clusters


def test_parse_centroid_from_string():
    assert parse_centroid("['кошка', 'собака']") == ['кошка', 'собака']


def test_parse_centroid_from_list():
    assert parse_centroid(['кошка']) == ['кошка']


@pytest.mark.parametrize("cell", [
    "__import__('os').system('echo pwned')",
    "open('secret.txt').read()",
    "['кошка', 1]",
    "'кошка'",
    "кошка, собака",
])
def test_parse_centroid_rejects_non_literal_lists(cell):
    with pytest.raises(ValueError):
        parse_centroid(cell)


def test_clusters_round_trip():
    clusters = [{'name': 'Животные', 'centroid': ['кошка', "кот д'Ивуара"], 'top_k': 140},
                {'name': 'Болезни', 'centroid': ['грипп'], 'top_k': 20}]

    df = make_dataframe_from_clusters(clusters)

    assert make_clusters_from_dataframe(df) == clusters


def test_make_clusters_skips_empty_rows_and_casts_top_k():
    df = pd.DataFrame({'name': ['Животные', ''], 'centroid': ["['кошка']", ''], 'top_k': ['5', '']})

    assert make_clusters_from_dataframe(df) == [{'name': 'Животные', 'centroid': ['кошка'], 'top_k': 5}]
