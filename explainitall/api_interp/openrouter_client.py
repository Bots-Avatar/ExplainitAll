import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

from .cache import ResponseCache, make_cache_key

OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
API_KEY_ENV = 'OPENROUTER_API_KEY'

# 408 — таймаут, 429 — лимит запросов, 502 — сбой провайдера, 503 — нет провайдера под условия маршрутизации
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_TOP_LOGPROBS = 20


class OpenRouterError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, metadata: Optional[dict] = None):
        super().__init__(f'[{status}] {message}' if status is not None else message)
        self.status = status
        self.metadata = metadata or {}


class BudgetExceededError(OpenRouterError):
    pass


@dataclass
class ProviderSettings:
    """
    Настройки маршрутизации OpenRouter (поле provider запроса).
    Для воспроизводимых экспериментов провайдер фиксируется: order=[tag], allow_fallbacks=False,
    а require_parameters=True не даёт провайдеру молча отбросить logprobs или seed.
    """
    order: Optional[List[str]] = None
    only: Optional[List[str]] = None
    allow_fallbacks: Optional[bool] = None
    require_parameters: Optional[bool] = None
    quantizations: Optional[List[str]] = None
    data_collection: Optional[str] = None  # 'deny' — только провайдеры, не сохраняющие данные
    zdr: Optional[bool] = None  # только провайдеры с нулевым хранением данных

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    top_logprobs: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class ChatResult:
    text: str
    finish_reason: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    usage: dict
    cost: float
    logprobs: Optional[List[TokenLogprob]]
    reasoning: Optional[str]
    cached: bool
    raw: dict

    @classmethod
    def from_response(cls, data: dict, cached: bool = False) -> 'ChatResult':
        choices = data.get('choices') or []
        if not choices:
            raise OpenRouterError('Ответ не содержит choices', metadata={'response': data})
        choice = choices[0]
        message = choice.get('message') or {}

        logprobs = None
        content_logprobs = (choice.get('logprobs') or {}).get('content')
        if content_logprobs is not None:
            logprobs = [TokenLogprob(token=t['token'], logprob=t['logprob'],
                                     top_logprobs=[(x['token'], x['logprob']) for x in t.get('top_logprobs') or []])
                        for t in content_logprobs]

        usage = data.get('usage') or {}
        return cls(text=message.get('content') or '',
                   finish_reason=choice.get('finish_reason'),
                   model=data.get('model'),
                   provider=data.get('provider'),
                   usage=usage,
                   cost=float(usage.get('cost') or 0.0),
                   logprobs=logprobs,
                   reasoning=message.get('reasoning'),
                   cached=cached,
                   raw=data)


@dataclass
class EmbeddingResult:
    vectors: np.ndarray  # (число текстов, размерность), в порядке входных текстов
    model: Optional[str]
    provider: Optional[str]
    usage: dict
    cost: float
    cached: bool

    @classmethod
    def from_response(cls, data: dict, n_inputs: int, cached: bool = False) -> 'EmbeddingResult':
        items = sorted(data.get('data') or [], key=lambda item: item.get('index', 0))
        if len(items) != n_inputs:
            raise OpenRouterError(f'Получено {len(items)} эмбеддингов вместо {n_inputs}', metadata={'response': data})
        if any(not isinstance(item.get('embedding'), list) for item in items):
            raise OpenRouterError('Эмбеддинги должны приходить списком чисел (encoding_format=float)')
        usage = data.get('usage') or {}
        return cls(vectors=np.asarray([item['embedding'] for item in items], dtype=float), model=data.get('model'),
                   provider=data.get('provider'), usage=usage, cost=float(usage.get('cost') or 0.0), cached=cached)


@dataclass
class RerankResult:
    scores: List[float]  # оценка релевантности каждого документа в порядке входного списка
    model: Optional[str]
    provider: Optional[str]
    usage: dict
    cost: float
    cached: bool

    @classmethod
    def from_response(cls, data: dict, n_documents: int, cached: bool = False) -> 'RerankResult':
        scores = [float('nan')] * n_documents
        for item in data.get('results') or []:
            index = item.get('index')
            if not isinstance(index, int) or not 0 <= index < n_documents:
                raise OpenRouterError(f'Некорректный индекс документа в ответе: {index!r}', metadata={'response': data})
            scores[index] = float(item['relevance_score'])
        usage = data.get('usage') or {}
        return cls(scores=scores, model=data.get('model'), provider=data.get('provider'), usage=usage,
                   cost=float(usage.get('cost') or 0.0), cached=cached)

    @property
    def ranking(self) -> List[int]:
        """Индексы документов по убыванию релевантности"""
        return sorted((i for i, s in enumerate(self.scores) if s == s), key=lambda i: -self.scores[i])


def _body_error(data: dict) -> Optional[dict]:
    """Ошибка из тела ответа: OpenRouter возвращает её на верхнем уровне или внутри choices"""
    if isinstance(data.get('error'), dict):
        return data['error']
    for choice in data.get('choices') or []:
        if isinstance(choice.get('error'), dict):
            return choice['error']
    return None


class OpenRouterClient:
    """
    Клиент OpenRouter для интерпретационных экспериментов: повторы при перегрузке,
    учёт стоимости с ограничением бюджета и кеширование ответов.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = OPENROUTER_BASE_URL,
                 cache: Optional[ResponseCache] = None,
                 max_cost: Optional[float] = None,
                 timeout: float = 120.0,
                 max_retries: int = 5,
                 backoff: float = 1.0,
                 app_url: Optional[str] = None,
                 app_title: Optional[str] = None,
                 session: Optional[requests.Session] = None,
                 sleep: Callable[[float], None] = time.sleep):
        """
        api_key: ключ API; по умолчанию берётся из переменной окружения OPENROUTER_API_KEY
        cache: кеш ответов; повторный одинаковый запрос не тратит деньги
        max_cost: предел расходов клиента в долларах; после его достижения новые запросы не отправляются
            (последний запрос может превысить предел на свою стоимость)
        """
        self.api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        self.base_url = base_url.rstrip('/')
        self.cache = cache
        self.max_cost = max_cost
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.app_url = app_url
        self.app_title = app_title
        self.session = session or requests.Session()
        self._sleep = sleep

        self._stats_lock = threading.Lock()
        self.total_cost = 0.0
        self.n_requests = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------ chat

    @staticmethod
    def build_payload(messages: Union[str, List[dict]],
                      model: str,
                      temperature: Optional[float] = 0.0,
                      max_tokens: Optional[int] = None,
                      seed: Optional[int] = None,
                      logprobs: bool = False,
                      top_logprobs: Optional[int] = None,
                      provider: Optional[ProviderSettings] = None,
                      extra: Optional[dict] = None) -> dict:
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        if top_logprobs is not None:
            if not 0 <= top_logprobs <= MAX_TOP_LOGPROBS:
                raise ValueError(f'top_logprobs должен быть от 0 до {MAX_TOP_LOGPROBS}')
            logprobs = True

        payload = {'model': model, 'messages': messages}
        optional = {'temperature': temperature, 'max_tokens': max_tokens, 'seed': seed,
                    'top_logprobs': top_logprobs}
        payload.update({k: v for k, v in optional.items() if v is not None})
        if logprobs:
            payload['logprobs'] = True
        if provider is not None and provider.to_dict():
            payload['provider'] = provider.to_dict()
        if extra:
            payload.update(extra)
        return payload

    def chat(self,
             messages: Union[str, List[dict]],
             model: str,
             temperature: Optional[float] = 0.0,
             max_tokens: Optional[int] = None,
             seed: Optional[int] = None,
             logprobs: bool = False,
             top_logprobs: Optional[int] = None,
             provider: Optional[ProviderSettings] = None,
             extra: Optional[dict] = None,
             sample_id: int = 0,
             use_cache: bool = True) -> ChatResult:
        """
        Запрос к /chat/completions.
        sample_id: номер независимой выборки для одного и того же запроса; влияет только на ключ кеша
        """
        payload = self.build_payload(messages, model, temperature, max_tokens, seed, logprobs, top_logprobs,
                                     provider, extra)

        key = make_cache_key(payload, sample_id)
        data, cached = self._post_cached('/chat/completions', payload, key, use_cache)
        result = ChatResult.from_response(data, cached=cached)
        if not cached and result.finish_reason != 'error':
            self._store(key, payload, data, use_cache)
        return result

    def embeddings(self,
                   inputs: Union[str, List[str]],
                   model: str,
                   dimensions: Optional[int] = None,
                   input_type: Optional[str] = None,
                   provider: Optional[ProviderSettings] = None,
                   use_cache: bool = True) -> EmbeddingResult:
        """
        Запрос к /embeddings: векторы текстов в порядке inputs.
        input_type: тип текста для асимметричных моделей, например 'search_query' или 'search_document'
        """
        inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        if not inputs or any(not text for text in inputs):
            raise ValueError('Тексты для эмбеддингов должны быть непустыми')
        payload = {'model': model, 'input': inputs, 'encoding_format': 'float'}
        optional = {'dimensions': dimensions, 'input_type': input_type}
        payload.update({k: v for k, v in optional.items() if v is not None})
        if provider is not None and provider.to_dict():
            payload['provider'] = provider.to_dict()

        key = make_cache_key({'endpoint': '/embeddings', **payload})
        data, cached = self._post_cached('/embeddings', payload, key, use_cache)
        result = EmbeddingResult.from_response(data, len(inputs), cached=cached)
        if not cached:
            self._store(key, payload, data, use_cache)
        return result

    def rerank(self,
               query: str,
               documents: List[str],
               model: str,
               top_n: Optional[int] = None,
               provider: Optional[ProviderSettings] = None,
               use_cache: bool = True) -> RerankResult:
        """Запрос к /rerank: оценки релевантности документов запросу в порядке documents"""
        documents = list(documents)
        if not documents:
            raise ValueError('Нужен хотя бы один документ')
        payload = {'model': model, 'query': query, 'documents': documents}
        if top_n is not None:
            payload['top_n'] = top_n
        if provider is not None and provider.to_dict():
            payload['provider'] = provider.to_dict()

        key = make_cache_key({'endpoint': '/rerank', **payload})
        data, cached = self._post_cached('/rerank', payload, key, use_cache)
        result = RerankResult.from_response(data, len(documents), cached=cached)
        if not cached:
            self._store(key, payload, data, use_cache)
        return result

    def _post_cached(self, path: str, payload: dict, key: str, use_cache: bool):
        """Ответ из кеша или из API: (данные, взят ли из кеша); стоимость ответа API учитывается в бюджете"""
        if self.cache is not None and use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                with self._stats_lock:
                    self.n_cache_hits += 1
                return cached, True

        self._check_budget()
        data = self._request('POST', path, json=payload, auth_required=True)
        with self._stats_lock:
            self.total_cost += float((data.get('usage') or {}).get('cost') or 0.0)
        return data, False

    def _store(self, key: str, payload: dict, data: dict, use_cache: bool):
        if self.cache is not None and use_cache:
            self.cache.set(key, payload, data)

    def _check_budget(self):
        if self.max_cost is not None and self.total_cost >= self.max_cost:
            raise BudgetExceededError(f'Исчерпан бюджет: потрачено ${self.total_cost:.6f} из ${self.max_cost:.6f}')

    # ---------------------------------------------------------------- models

    def list_models(self, supported_parameters: Optional[List[str]] = None,
                    output_modalities: Optional[List[str]] = None) -> List[dict]:
        """
        Список моделей. Фильтры на стороне сервера:
        supported_parameters — например ['logprobs'];
        output_modalities — например ['embeddings'] или ['rerank'] (по умолчанию сервер отдаёт текстовые модели)
        """
        params = {}
        if supported_parameters:
            params['supported_parameters'] = ','.join(supported_parameters)
        if output_modalities:
            params['output_modalities'] = ','.join(output_modalities)
        return self._request('GET', '/models', params=params or None)['data']

    def list_endpoints(self, model: str) -> dict:
        """Провайдеры (эндпоинты) модели с их параметрами, квантованием и ценами"""
        author, sep, slug = model.partition('/')
        if not sep or not author or not slug:
            raise ValueError(f'Идентификатор модели должен иметь вид author/slug: {model!r}')
        return self._request('GET', f'/models/{author}/{slug}/endpoints')['data']

    # --------------------------------------------------------------- network

    def _headers(self, auth_required: bool) -> Dict[str, str]:
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        elif auth_required:
            raise OpenRouterError(f'Не задан ключ API: передайте api_key или задайте {API_KEY_ENV}')
        if self.app_url:
            headers['HTTP-Referer'] = self.app_url
        if self.app_title:
            headers['X-OpenRouter-Title'] = self.app_title
        return headers

    def _retry_delay(self, attempt: int, response: Optional[requests.Response] = None) -> float:
        if response is not None:
            retry_after = response.headers.get('Retry-After')
            if retry_after is not None:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    pass
        return self.backoff * (2 ** attempt)

    def _request(self, method: str, path: str, auth_required: bool = False, **kwargs) -> dict:
        url = self.base_url + path
        headers = self._headers(auth_required)

        for attempt in range(self.max_retries + 1):
            is_last = attempt == self.max_retries
            try:
                with self._stats_lock:
                    self.n_requests += 1
                response = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as e:
                if is_last:
                    raise OpenRouterError(f'Сетевая ошибка: {e}') from e
                self._sleep(self._retry_delay(attempt))
                continue

            try:
                data = response.json()
            except ValueError:
                data = None

            error = _body_error(data) if isinstance(data, dict) else None
            status = response.status_code
            if error is not None and status < 400:
                status = error.get('code') if isinstance(error.get('code'), int) else status

            if status in RETRYABLE_STATUSES and not is_last:
                self._sleep(self._retry_delay(attempt, response))
                continue

            if error is not None or response.status_code >= 400:
                message = (error or {}).get('message') or response.text[:500] or 'Ошибка запроса'
                raise OpenRouterError(message, status=status, metadata=(error or {}).get('metadata'))
            if not isinstance(data, dict):
                raise OpenRouterError('Ответ не является JSON-объектом', status=status)
            return data

        raise AssertionError('недостижимо')
