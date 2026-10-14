import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .agreement import RankAgreement, api_cluster_importance, rank_agreement
from .explainer import ExplainerAPI, ExplainerAPIOutput, NoClusterWordsError
from .logprob_scoring import ForcedLogprobScorer
from .perturbations import ClusterMasker, Messages, normalize_messages
from .scorers import SemanticScorer

Measure = Callable[[List[dict]], float]


class LogprobDropMeasure:
    """Падение логарифма вероятности эталонного ответа (нат на токен) при изменённом промпте"""

    def __init__(self, scorer: ForcedLogprobScorer, reference_tokens: Sequence[str],
                 reference_logprobs: Sequence[float]):
        self.scorer = scorer
        self.reference_tokens = list(reference_tokens)
        self.reference_logprobs = list(reference_logprobs)
        self.cost = 0.0

    def __call__(self, messages: List[dict]) -> float:
        score = self.scorer.score(messages, self.reference_tokens)
        self.cost += score.cost
        if score.n_tokens == 0:
            return np.nan
        return (sum(self.reference_logprobs[:score.n_tokens]) - score.logprob) / score.n_tokens


class SemanticDistanceMeasure:
    """Смысловое отличие нового ответа от эталонного: 1 - косинусная близость"""

    def __init__(self, client, model: str, reference_text: str, scorer: SemanticScorer, provider=None,
                 max_tokens: Optional[int] = 256, temperature: Optional[float] = 0.0, seed: Optional[int] = None):
        self.client = client
        self.model = model
        self.reference_text = reference_text
        self.scorer = scorer
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.cost = 0.0

    def __call__(self, messages: List[dict]) -> float:
        result = self.client.chat(messages, self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                                  seed=self.seed, provider=self.provider)
        if not result.cached:
            self.cost += result.cost
        return self.scorer.distances(self.reference_text, [result.text])[0]


def ordering_from_importance(importance: pd.Series, names: Sequence[str]) -> List[str]:
    """Кластеры по убыванию важности; кластеры без оценки идут в конце в исходном порядке"""
    scored = [n for n in importance.sort_values(ascending=False).index if n in names]
    return scored + [n for n in names if n not in scored]


def deletion_curve(masker: ClusterMasker, messages: Messages, ordering: Sequence[str], measure: Measure,
                   cache: Optional[Dict[frozenset, float]] = None) -> List[float]:
    """
    Удаляющая кривая: изменение ответа после накопительного скрытия первых k кластеров ordering, k = 0..K.
    cache: общий для разных порядков кеш по набору скрытых кластеров
    """
    cache = {} if cache is None else cache
    curve = [0.0]
    for k in range(1, len(ordering) + 1):
        key = frozenset(ordering[:k])
        if key not in cache:
            cache[key] = measure(masker.mask_clusters(messages, ordering[:k]))
        curve.append(cache[key])
    return curve


def aopc(curve: Sequence[float]) -> float:
    """Площадь над удаляющей кривой (среднее изменение по шагам k >= 1): чем больше, тем точнее порядок"""
    values = np.asarray(curve[1:], dtype=float)
    return float(np.nanmean(values)) if len(values) and not np.all(np.isnan(values)) else np.nan


def random_orderings(names: Sequence[str], n: int, seed: int = 0) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    return [list(rng.permutation(list(names))) for _ in range(n)]


@dataclass
class PromptValidation:
    prompt: List[dict]
    clusters: List[str]
    answer: str
    rankings: Dict[str, pd.Series]
    agreements: Dict[Tuple[str, str], RankAgreement]
    curves: Dict[str, List[float]]
    aopc: Dict[str, float]
    random_aopc: List[float]
    cost: float
    warnings: List[str] = field(default_factory=list)

    @property
    def random_aopc_mean(self) -> float:
        return float(np.nanmean(self.random_aopc)) if self.random_aopc else np.nan


class ValidationExperiment:
    """
    Проверка методов интерпретации модели за API на наборе промптов:
    - согласованность рангов кластеров между методами (исключение кластеров по смыслу и по logprobs,
      атрибуции двойника);
    - удаляющие кривые: насколько сильнее меняется ответ, если скрывать кластеры в порядке,
      предложенном методом, а не в случайном (AOPC против случайного порядка).
    """

    def __init__(self, explainer: ExplainerAPI, twin=None, curve_measure: str = 'logprob', n_random: int = 5,
                 seed: int = 0):
        """
        explainer: ExplainerAPI; для curve_measure='logprob' нужен logprobs=True, для 'semantic' — semantic=True
        twin: SurrogateExplainer или None
        """
        if curve_measure not in ('logprob', 'semantic'):
            raise ValueError("curve_measure должен быть 'logprob' или 'semantic'")
        if curve_measure == 'logprob' and not explainer.logprobs:
            raise ValueError("Для curve_measure='logprob' нужен ExplainerAPI(logprobs=True)")
        if curve_measure == 'semantic' and not explainer.semantic:
            raise ValueError("Для curve_measure='semantic' нужен ExplainerAPI(semantic=True)")
        self.explainer = explainer
        self.twin = twin
        self.curve_measure = curve_measure
        self.n_random = n_random
        self.seed = seed

    def _rankings(self, prompt: Messages, clusters_description: List[dict], api_out: ExplainerAPIOutput):
        rankings = {}
        df = api_out.importance_df
        if self.explainer.semantic:
            rankings['occlusion_semantic'] = api_cluster_importance(df, 'importance')
        if self.explainer.logprobs:
            rankings['occlusion_logprob'] = api_cluster_importance(df, 'logprob_drop')
        if self.twin is not None:
            twin_out = self.twin.interpret(prompt, clusters_description, answer=api_out.reference_answer)
            rankings['twin_ig'] = twin_out.input_cluster_importance()
        return rankings

    def _measure(self, api_out: ExplainerAPIOutput):
        e = self.explainer
        if self.curve_measure == 'logprob':
            analysis = api_out.logprob_analysis
            scorer = e._make_logprob_scorer()
            if analysis.prefill is not None and analysis.prefill.supported is False:
                scorer.use_prefill = False
            reference_logprobs = [t.logprob for t in api_out.records[0].result.logprobs[:len(analysis.reference_tokens)]]
            return LogprobDropMeasure(scorer, analysis.reference_tokens, reference_logprobs)
        return SemanticDistanceMeasure(e.client, e.model, api_out.reference_answer, e.scorer, provider=e.provider,
                                       max_tokens=e.max_tokens, temperature=e.temperature, seed=e.seed)

    def run_one(self, prompt: Messages, clusters_description: List[dict]) -> PromptValidation:
        messages = normalize_messages(prompt)
        api_out = self.explainer.interpret(messages, clusters_description)
        names = [row.cluster for row in api_out.importance_df.itertuples() if row.words]

        rankings = self._rankings(messages, clusters_description, api_out)
        agreements = {(a, b): rank_agreement(rankings[a], rankings[b])
                      for a, b in itertools.combinations(sorted(rankings), 2)}

        masker = self.explainer._masker(clusters_description)
        measure = self._measure(api_out)
        cache: Dict[frozenset, float] = {}
        curves = {method: deletion_curve(masker, messages, ordering_from_importance(ranking, names), measure, cache)
                  for method, ranking in rankings.items()}
        random_aopc = [aopc(deletion_curve(masker, messages, ordering, measure, cache))
                       for ordering in random_orderings(names, self.n_random, self.seed)]

        return PromptValidation(prompt=messages, clusters=names, answer=api_out.reference_answer, rankings=rankings,
                                agreements=agreements, curves=curves,
                                aopc={m: aopc(c) for m, c in curves.items()}, random_aopc=random_aopc,
                                cost=api_out.total_cost + measure.cost, warnings=list(api_out.warnings))

    def run(self, dataset: Sequence[Tuple[Messages, List[dict]]],
            on_error: str = 'raise') -> List[PromptValidation]:
        """
        dataset: пары (промпт, описание кластеров)
        on_error: 'skip' — пропускать промпты без слов кластеров; 'raise' — прерывать на них. Прочие ошибки всегда прерывают
        """
        results = []
        for prompt, clusters_description in dataset:
            try:
                results.append(self.run_one(prompt, clusters_description))
            except NoClusterWordsError:
                if on_error != 'skip':
                    raise
        return results


def summarize(results: Sequence[PromptValidation]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Сводка по набору промптов:
    - по методам: средний AOPC, выигрыш над случайным порядком и доля промптов, где метод лучше случайного;
    - по парам методов: средняя корреляция рангов и доля совпадений самого важного кластера.
    Промпты с одним найденным кластером в удаляющих кривых не учитываются: там порядок не на что влиять.
    """
    method_rows, agreement_rows = {}, {}
    for r in results:
        informative = len(r.clusters) >= 2
        for method, value in r.aopc.items():
            row = method_rows.setdefault(method, {'aopc': [], 'gain': [], 'better': []})
            if informative and not np.isnan(value):
                row['aopc'].append(value)
                row['gain'].append(value - r.random_aopc_mean)
                row['better'].append(value > r.random_aopc_mean)
        for pair, agreement in r.agreements.items():
            row = agreement_rows.setdefault(pair, {'spearman': [], 'top1': []})
            if agreement.n_common:
                row['spearman'].append(agreement.spearman)
                row['top1'].append(agreement.top1_match)

    methods = pd.DataFrame([{'method': m, 'n_prompts': len(v['aopc']),
                             'aopc_mean': np.mean(v['aopc']) if v['aopc'] else np.nan,
                             'gain_over_random': np.mean(v['gain']) if v['gain'] else np.nan,
                             'better_than_random_share': np.mean(v['better']) if v['better'] else np.nan}
                            for m, v in method_rows.items()], columns=['method', 'n_prompts', 'aopc_mean',
                                                                       'gain_over_random', 'better_than_random_share'])
    agreements = pd.DataFrame([{'method_a': a, 'method_b': b, 'n_prompts': len(v['top1']),
                                'spearman_mean': np.nanmean(v['spearman']) if any(
                                    not np.isnan(s) for s in v['spearman']) else np.nan,
                                'top1_match_share': np.mean(v['top1']) if v['top1'] else np.nan}
                               for (a, b), v in agreement_rows.items()],
                              columns=['method_a', 'method_b', 'n_prompts', 'spearman_mean', 'top1_match_share'])
    return methods, agreements
