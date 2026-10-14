from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from explainitall.clusters import ClusterManager
from .logprob_scoring import ForcedLogprobScorer, ForcedScore, PrefillCheck
from .openrouter_client import ChatResult, OpenRouterClient, ProviderSettings
from .perturbations import ClusterMasker, DEFAULT_REPLACEMENT, Messages, normalize_messages
from .scorers import SemanticScorer

BASELINE = 'baseline'

SEMANTIC_COLUMNS = ['importance', 'distance', 'distance_std', 'significant', 'n_samples']
LOGPROB_COLUMNS = ['logprob_drop', 'logprob_drop_total', 'logprob_tokens', 'logprob_truncated',
                   'logprob_calls', 'logprob_significant']


@dataclass
class InterpretationPlan:
    """Что будет отправлено в API: исходный промпт и промпты без слов каждого найденного кластера"""
    messages: List[dict]
    cluster_words: Dict[str, List[str]]
    perturbed: Dict[str, List[dict]]
    n_baseline: int
    n_repeats: int
    semantic: bool = True
    logprobs: bool = False
    verify_prefill: bool = False
    max_scoring_calls: int = 0

    @property
    def n_generation_requests(self) -> int:
        return self.n_baseline + (len(self.perturbed) * self.n_repeats if self.semantic else 0)

    @property
    def n_requests(self) -> int:
        """Наименьшее число запросов"""
        extra = len(self.perturbed) if self.logprobs and not self.semantic else 0
        return self.n_generation_requests + extra

    @property
    def n_requests_max(self) -> int:
        """Наибольшее число запросов: оценка по logprobs добавляет запросы на каждом расхождении с эталоном"""
        if not self.logprobs:
            return self.n_requests
        scored_prompts = len(self.perturbed) + self.n_baseline - 1
        return self.n_generation_requests + (2 if self.verify_prefill else 0) + scored_prompts * self.max_scoring_calls


@dataclass
class GenerationRecord:
    kind: str  # 'baseline' или имя скрытого кластера
    sample_id: int
    messages: List[dict]
    text: str
    cost: float
    cached: bool
    provider: Optional[str]
    result: Optional[ChatResult] = field(default=None, repr=False)


@dataclass
class LogprobAnalysis:
    reference_tokens: List[str]
    prefill: Optional[PrefillCheck]
    scores: Dict[str, ForcedScore]
    noise_scores: List[ForcedScore]
    noise_mean: float
    noise_max: float
    warnings: List[str]

    @property
    def cost(self) -> float:
        scores = list(self.scores.values()) + self.noise_scores
        return sum(s.cost for s in scores) + (self.prefill.cost if self.prefill else 0.0)

    @property
    def n_api_requests(self) -> int:
        scores = list(self.scores.values()) + self.noise_scores
        n = sum(s.n_calls - s.n_cached for s in scores)
        return n + (self.prefill.n_calls - self.prefill.n_cached if self.prefill else 0)


class ExplainerAPIOutput:
    def __init__(self, importance_df: pd.DataFrame, cluster_imp_aggr_df: pd.DataFrame,
                 records: List[GenerationRecord], noise_mean: float, noise_max: float,
                 logprob_analysis: Optional[LogprobAnalysis] = None):
        self.importance_df = importance_df
        self.cluster_imp_aggr_df = cluster_imp_aggr_df
        self.records = records
        self.noise_mean = noise_mean
        self.noise_max = noise_max
        self.logprob_analysis = logprob_analysis

    @property
    def baseline_answers(self) -> List[str]:
        return [r.text for r in self.records if r.kind == BASELINE]

    @property
    def reference_answer(self) -> str:
        return self.baseline_answers[0]

    def answers_without(self, cluster_name: str) -> List[str]:
        return [r.text for r in self.records if r.kind == cluster_name]

    @property
    def warnings(self) -> List[str]:
        return self.logprob_analysis.warnings if self.logprob_analysis else []

    @property
    def total_cost(self) -> float:
        cost = sum(r.cost for r in self.records if not r.cached)
        return cost + (self.logprob_analysis.cost if self.logprob_analysis else 0.0)

    @property
    def n_api_requests(self) -> int:
        n = sum(not r.cached for r in self.records)
        return n + (self.logprob_analysis.n_api_requests if self.logprob_analysis else 0)

    def show_cluster_imp_aggr_heatmap(self):
        from explainitall.gpt_like_interp import viz
        viz.df_to_heatmap(self.cluster_imp_aggr_df, title="Влияние кластеров промпта на кластеры ответа")


def _run_parallel(jobs: List[Callable], max_workers: int) -> list:
    """Выполняет функции параллельно, сохраняя порядок результатов; первая ошибка прерывает остальные"""
    results = [None] * len(jobs)
    executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
    try:
        futures = {executor.submit(job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return results


class ExplainerAPI:
    """
    Интерпретация модели, доступной только по API (чёрный ящик), через исключение кластеров из промпта.

    Для каждого кластера слова промпта, относящиеся к нему, скрываются, и измеряется,
    как это меняет ответ модели:
    - importance — насколько сильнее меняется смысл ответа, чем от случайных колебаний модели;
    - logprob_drop (при logprobs=True) — на сколько натов на токен падает логарифм вероятности
      исходного ответа; exp(logprob_drop_total) — во сколько раз модель реже дала бы исходный ответ;
    - cluster_imp_aggr_df — насколько скрытие кластера промпта (строка) убирает из ответа слова
      кластера ответа (столбец): +1 — исчезает всегда, -1 — появляется всегда.
    """

    def __init__(self,
                 client: OpenRouterClient,
                 model: str,
                 nlp_model,
                 embedder=None,
                 provider: Optional[ProviderSettings] = None,
                 max_tokens: Optional[int] = 256,
                 temperature: Optional[float] = 0.0,
                 seed: Optional[int] = None,
                 n_baseline: int = 3,
                 n_repeats: int = 1,
                 replacement: str = DEFAULT_REPLACEMENT,
                 mask_roles: Iterable[str] = ('user',),
                 max_workers: int = 4,
                 semantic: bool = True,
                 logprobs: bool = False,
                 top_logprobs: int = 20,
                 max_answer_tokens: int = 32,
                 max_scoring_calls: int = 8,
                 use_prefill: Union[bool, str] = 'auto'):
        """
        nlp_model: векторная модель слов gensim (как в ExplainerGPT2) для построения кластеров
        embedder: модель предложений с методом encode, например SentenceTransformer; нужна при semantic=True
        provider: маршрутизация OpenRouter; для воспроизводимости и logprobs закрепите провайдера
            (capabilities.select_endpoint + pinned_provider)
        n_baseline: число генераций по исходному промпту для оценки шума (не меньше 1)
        n_repeats: число генераций на каждый промпт со скрытым кластером
        semantic: генерировать ответы без кластеров и сравнивать их по смыслу и по кластерам ответа
        logprobs: оценивать падение вероятности исходного ответа (нужен провайдер с logprobs)
        max_answer_tokens: сколько первых токенов исходного ответа оценивается по logprobs
        max_scoring_calls: предел запросов на оценку одного промпта по logprobs
        use_prefill: 'auto' — проверить поддержку prefill на исходном промпте; True/False — задать явно
        """
        if n_baseline < 1 or n_repeats < 1:
            raise ValueError('n_baseline и n_repeats должны быть не меньше 1')
        if not semantic and not logprobs:
            raise ValueError('Нужен хотя бы один способ оценки: semantic или logprobs')
        if semantic and embedder is None:
            raise ValueError('Для semantic=True нужен embedder')
        if use_prefill not in (True, False, 'auto'):
            raise ValueError("use_prefill должен быть True, False или 'auto'")

        self.client = client
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.n_baseline = n_baseline
        self.n_repeats = n_repeats
        self.replacement = replacement
        self.mask_roles = tuple(mask_roles)
        self.max_workers = max_workers
        self.semantic = semantic
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.max_answer_tokens = max_answer_tokens
        self.max_scoring_calls = max_scoring_calls
        self.use_prefill = use_prefill
        self.scorer = SemanticScorer(embedder) if embedder is not None else None
        self._cluster_manager = ClusterManager(embeddings=nlp_model)

    # ------------------------------------------------------------------ plan

    def _masker(self, clusters_description: List[dict]) -> ClusterMasker:
        clusters = self._cluster_manager.create_clusters(clusters_description)
        return ClusterMasker(self._cluster_manager, clusters, roles=self.mask_roles, replacement=self.replacement)

    def plan(self, prompt: Messages, clusters_description: List[dict]) -> InterpretationPlan:
        """План запросов без обращения к API: можно оценить стоимость до запуска"""
        return self._plan(self._masker(clusters_description), prompt)

    def _plan(self, masker: ClusterMasker, prompt: Messages) -> InterpretationPlan:
        messages = normalize_messages(prompt)
        cluster_words = masker.find_cluster_words(messages)
        perturbed = {name: masker.mask_cluster(messages, name) for name, words in cluster_words.items() if words}
        return InterpretationPlan(messages=messages, cluster_words=cluster_words, perturbed=perturbed,
                                  n_baseline=self.n_baseline, n_repeats=self.n_repeats, semantic=self.semantic,
                                  logprobs=self.logprobs, verify_prefill=self.logprobs and self.use_prefill == 'auto',
                                  max_scoring_calls=self.max_scoring_calls)

    # -------------------------------------------------------------- generate

    def _generate(self, kind: str, sample_id: int, messages: List[dict]) -> GenerationRecord:
        result = self.client.chat(messages, self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                                  seed=self.seed, provider=self.provider, sample_id=sample_id,
                                  top_logprobs=self.top_logprobs if self.logprobs else None)
        return GenerationRecord(kind=kind, sample_id=sample_id, messages=messages, text=result.text,
                                cost=result.cost, cached=result.cached, provider=result.provider, result=result)

    def _run(self, plan: InterpretationPlan) -> List[GenerationRecord]:
        jobs = [(BASELINE, i, plan.messages) for i in range(plan.n_baseline)]
        if plan.semantic:
            jobs += [(name, r, messages) for name, messages in plan.perturbed.items() for r in range(plan.n_repeats)]
        return _run_parallel([lambda job=job: self._generate(*job) for job in jobs], self.max_workers)

    def interpret(self, prompt: Messages, clusters_description: List[dict]) -> ExplainerAPIOutput:
        masker = self._masker(clusters_description)
        plan = self._plan(masker, prompt)
        if not plan.perturbed:
            raise ValueError('В промпте не найдено ни одного слова из заданных кластеров')

        records = self._run(plan)
        logprob_analysis = self._logprob_analysis(plan, records) if self.logprobs else None
        return self._analyze(masker, plan, records, logprob_analysis)

    # --------------------------------------------------------------- logprobs

    def _make_logprob_scorer(self) -> ForcedLogprobScorer:
        return ForcedLogprobScorer(self.client, self.model, provider=self.provider, top_logprobs=self.top_logprobs,
                                   max_answer_tokens=self.max_answer_tokens, max_calls=self.max_scoring_calls,
                                   temperature=self.temperature, seed=self.seed,
                                   use_prefill=self.use_prefill is not False)

    def _logprob_analysis(self, plan: InterpretationPlan, records: List[GenerationRecord]) -> LogprobAnalysis:
        scorer = self._make_logprob_scorer()
        baseline = [r for r in records if r.kind == BASELINE]
        reference = scorer.reference_tokens(baseline[0].result)
        warnings = []

        prefill = None
        if self.use_prefill == 'auto':
            prefill = scorer.verify_prefill(plan.messages, reference)
            if prefill.supported is False:
                scorer.use_prefill = False
                warnings.append('Провайдер не продолжает ответ с prefill: по logprobs оценивается только начало '
                                'ответа до первого расхождения с исходным')

        first_results = {}
        if plan.semantic:
            for r in records:
                if r.kind != BASELINE and r.sample_id == 0:
                    first_results[r.kind] = r.result

        jobs = [lambda r=r: scorer.score(plan.messages, reference, first_result=r.result) for r in baseline[1:]]
        names = list(plan.perturbed)
        jobs += [lambda n=n: scorer.score(plan.perturbed[n], reference, first_result=first_results.get(n))
                 for n in names]
        results = _run_parallel(jobs, self.max_workers)
        noise_scores, scores = results[:len(baseline) - 1], dict(zip(names, results[len(baseline) - 1:]))

        reference_logprobs = [t.logprob for t in baseline[0].result.logprobs[:len(reference)]]
        noise = [abs(self._drop(reference_logprobs, s)[0]) for s in noise_scores if s.n_tokens]
        if any(s.n_truncated for s in scores.values()):
            warnings.append('Часть эталонных токенов не попала в top_logprobs: для них logprob_drop занижен')

        return LogprobAnalysis(reference_tokens=reference, prefill=prefill, scores=scores, noise_scores=noise_scores,
                               noise_mean=float(np.mean(noise)) if noise else 0.0,
                               noise_max=float(np.max(noise)) if noise else 0.0, warnings=warnings)

    @staticmethod
    def _drop(reference_logprobs: List[float], score: ForcedScore):
        """Падение logprob исходного ответа на оценённых токенах: (на токен, всего)"""
        n = score.n_tokens
        if n == 0:
            return np.nan, np.nan
        total = float(sum(reference_logprobs[:n]) - score.logprob)
        return total / n, total

    # ---------------------------------------------------------------- analyze

    def _analyze(self, masker: ClusterMasker, plan: InterpretationPlan, records: List[GenerationRecord],
                 logprob_analysis: Optional[LogprobAnalysis]) -> ExplainerAPIOutput:
        names = masker.cluster_names
        baseline = [r.text for r in records if r.kind == BASELINE]

        noise_mean = noise_max = 0.0
        if self.semantic:
            noise = self.scorer.distances(baseline[0], baseline[1:])
            noise_mean = float(np.mean(noise)) if noise else 0.0
            noise_max = float(np.max(noise)) if noise else 0.0
            baseline_presence = self._mean_presence(masker, baseline)

        reference_logprobs = None
        if logprob_analysis is not None:
            reference_logprobs = [t.logprob for t in
                                  records[0].result.logprobs[:len(logprob_analysis.reference_tokens)]]

        rows, influence_rows = [], []
        for name in names:
            row = {'cluster': name, 'words': plan.cluster_words[name]}
            texts = [r.text for r in records if r.kind == name]

            if self.semantic and texts:
                distances = self.scorer.distances(baseline[0], texts)
                distance = float(np.mean(distances))
                row.update(importance=max(0.0, distance - noise_mean), distance=distance,
                           distance_std=float(np.std(distances)), significant=distance > noise_max,
                           n_samples=len(texts))
                perturbed_presence = self._mean_presence(masker, texts)
                influence_rows.append({'Tokens': name, **{out: baseline_presence[out] - perturbed_presence[out]
                                                          for out in names}})
            elif self.semantic:
                row.update(importance=np.nan, distance=np.nan, distance_std=np.nan, significant=False, n_samples=0)

            if logprob_analysis is not None:
                score = logprob_analysis.scores.get(name)
                if score is not None:
                    per_token, total = self._drop(reference_logprobs, score)
                    row.update(logprob_drop=per_token, logprob_drop_total=total, logprob_tokens=score.n_tokens,
                               logprob_truncated=score.n_truncated, logprob_calls=score.n_calls,
                               logprob_significant=bool(per_token > logprob_analysis.noise_max))
                else:
                    row.update(logprob_drop=np.nan, logprob_drop_total=np.nan, logprob_tokens=0,
                               logprob_truncated=0, logprob_calls=0, logprob_significant=False)
            rows.append(row)

        columns = ['cluster', 'words'] + (SEMANTIC_COLUMNS if self.semantic else []) + (
            LOGPROB_COLUMNS if logprob_analysis is not None else [])
        importance_df = pd.DataFrame(rows, columns=columns)
        sort_by = 'importance' if self.semantic else 'logprob_drop'
        importance_df = importance_df.sort_values(sort_by, ascending=False, na_position='last', ignore_index=True)
        cluster_imp_aggr_df = pd.DataFrame(influence_rows, columns=['Tokens', *names])
        return ExplainerAPIOutput(importance_df=importance_df, cluster_imp_aggr_df=cluster_imp_aggr_df,
                                  records=records, noise_mean=noise_mean, noise_max=noise_max,
                                  logprob_analysis=logprob_analysis)

    @staticmethod
    def _mean_presence(masker: ClusterMasker, texts: List[str]) -> Dict[str, float]:
        presences = [masker.cluster_presence(t) for t in texts]
        return {name: float(np.mean([p[name] for p in presences])) for name in masker.cluster_names}
