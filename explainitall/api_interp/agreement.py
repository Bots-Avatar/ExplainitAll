from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class RankAgreement:
    spearman: float  # корреляция рангов на общих кластерах; nan, если общих меньше трёх
    n_common: int
    top1_match: bool  # совпадает ли самый важный кластер


def api_cluster_importance(importance_df: pd.DataFrame, column: str = 'importance') -> pd.Series:
    """Важность кластеров из ExplainerAPIOutput.importance_df ('importance' или 'logprob_drop')"""
    return importance_df.set_index('cluster')[column].dropna().astype(float)


def rank_agreement(a: Mapping[str, float], b: Mapping[str, float]) -> RankAgreement:
    """
    Согласованность двух оценок важности кластеров, например двойника (Integrated Gradients)
    и чёрного ящика (исключение кластеров): совпадает ли порядок важности.
    """
    a, b = pd.Series(dict(a), dtype=float).dropna(), pd.Series(dict(b), dtype=float).dropna()
    common = [k for k in a.index if k in b.index]
    if not common:
        return RankAgreement(spearman=np.nan, n_common=0, top1_match=False)

    top1_match = a[common].idxmax() == b[common].idxmax()
    if len(common) < 3:
        return RankAgreement(spearman=np.nan, n_common=len(common), top1_match=bool(top1_match))
    rho = spearmanr(a[common].values, b[common].values).correlation
    return RankAgreement(spearman=float(rho), n_common=len(common), top1_match=bool(top1_match))
