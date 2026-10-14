import ast
import copy

import pandas as pd


def parse_centroid(centroid):
    """
    Разбор центроида кластера из ячейки таблицы.
    Принимает список строк или его текстовую запись вида "['кошка', 'собака']".
    Код из ячейки не исполняется: допускаются только литералы Python.
    """
    if isinstance(centroid, str):
        try:
            centroid = ast.literal_eval(centroid)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Центроид должен быть списком строк, например ['кошка', 'собака']: {centroid!r}") from e

    if not isinstance(centroid, (list, tuple)) or not all(isinstance(w, str) for w in centroid):
        raise ValueError(f"Центроид должен быть списком строк, например ['кошка', 'собака']: {centroid!r}")
    return list(centroid)


def make_clusters_from_dataframe(df):
    clusters = []
    for row in df.itertuples():
        name = getattr(row, 'name')
        centroid = getattr(row, 'centroid')
        top_k = getattr(row, 'top_k')
        if name is None or name == "":
            continue
        d = {'name': name, 'centroid': parse_centroid(centroid), 'top_k': int(top_k)}
        clusters.append(d)
    return clusters


def make_dataframe_from_clusters(clusters):
    clusters = copy.deepcopy(clusters)
    for c in clusters:
        c['centroid'] = str(list(c['centroid']))
    res = pd.DataFrame(clusters)
    return res
