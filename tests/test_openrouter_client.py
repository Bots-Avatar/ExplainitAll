import pytest
import requests

from explainitall.api_interp.cache import ResponseCache, make_cache_key
from explainitall.api_interp.openrouter_client import (OpenRouterClient, OpenRouterError, BudgetExceededError,
                                                       ProviderSettings, ChatResult)

URL = 'https://openrouter.ai/api/v1'


class FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None, text=''):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError('not json')
        return self._body


class FakeSession:
    """Отдаёт заранее заданные ответы и запоминает запросы"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append({'method': method, 'url': url, 'headers': headers, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def completion(text='Ответ', cost=0.001, logprobs=None, provider='Novita', finish_reason='stop'):
    choice = {'index': 0, 'finish_reason': finish_reason, 'message': {'role': 'assistant', 'content': text}}
    if logprobs is not None:
        choice['logprobs'] = {'content': logprobs, 'refusal': None}
    return FakeResponse(200, {'id': 'gen-1', 'model': 'meta-llama/llama-3.1-8b-instruct', 'provider': provider,
                              'choices': [choice],
                              'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12,
                                        'cost': cost}})


def make_client(*responses, **kwargs):
    session = FakeSession(*responses)
    sleeps = []
    kwargs.setdefault('api_key', 'test-key')
    client = OpenRouterClient(session=session, sleep=sleeps.append, **kwargs)
    return client, session, sleeps


def test_build_payload_defaults_and_optional_fields():
    payload = OpenRouterClient.build_payload('Привет', 'm/x', top_logprobs=5, seed=7,
                                             provider=ProviderSettings(order=['novita/fp8'], allow_fallbacks=False))

    assert payload == {'model': 'm/x', 'messages': [{'role': 'user', 'content': 'Привет'}],
                       'temperature': 0.0, 'seed': 7, 'top_logprobs': 5, 'logprobs': True,
                       'provider': {'order': ['novita/fp8'], 'allow_fallbacks': False}}


def test_build_payload_omits_none_values():
    payload = OpenRouterClient.build_payload([{'role': 'user', 'content': 'a'}], 'm/x', temperature=None)

    assert payload == {'model': 'm/x', 'messages': [{'role': 'user', 'content': 'a'}]}


@pytest.mark.parametrize('top_logprobs', [-1, 21])
def test_build_payload_rejects_invalid_top_logprobs(top_logprobs):
    with pytest.raises(ValueError):
        OpenRouterClient.build_payload('a', 'm/x', top_logprobs=top_logprobs)


def test_chat_parses_response_and_tracks_cost():
    logprobs = [{'token': 'От', 'logprob': -0.1, 'bytes': None,
                 'top_logprobs': [{'token': 'От', 'logprob': -0.1, 'bytes': None},
                                  {'token': 'Да', 'logprob': -2.5, 'bytes': None}]},
                {'token': 'вет', 'logprob': -0.2, 'bytes': None, 'top_logprobs': []}]
    client, session, _ = make_client(completion(cost=0.002, logprobs=logprobs))

    result = client.chat('Вопрос', 'meta-llama/llama-3.1-8b-instruct', top_logprobs=2)

    assert result.text == 'Ответ'
    assert result.provider == 'Novita'
    assert result.cost == pytest.approx(0.002)
    assert [t.token for t in result.logprobs] == ['От', 'вет']
    assert result.logprobs[0].top_logprobs == [('От', -0.1), ('Да', -2.5)]
    assert not result.cached
    assert client.total_cost == pytest.approx(0.002)

    call = session.calls[0]
    assert (call['method'], call['url']) == ('POST', URL + '/chat/completions')
    assert call['headers']['Authorization'] == 'Bearer test-key'
    assert call['json']['logprobs'] is True


def test_chat_without_logprobs_returns_none():
    client, _, _ = make_client(completion())

    assert client.chat('a', 'm/x').logprobs is None


def test_api_key_is_taken_from_environment(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'env-key')
    session = FakeSession(completion())

    OpenRouterClient(session=session).chat('a', 'm/x')

    assert session.calls[0]['headers']['Authorization'] == 'Bearer env-key'


def test_chat_without_api_key_fails_before_request(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    session = FakeSession()

    with pytest.raises(OpenRouterError):
        OpenRouterClient(session=session).chat('a', 'm/x')
    assert session.calls == []


def test_app_attribution_headers():
    client, session, _ = make_client(completion(), app_url='https://example.org', app_title='ExplainitAll')

    client.chat('a', 'm/x')

    assert session.calls[0]['headers']['HTTP-Referer'] == 'https://example.org'
    assert session.calls[0]['headers']['X-OpenRouter-Title'] == 'ExplainitAll'


def test_retry_on_rate_limit_uses_retry_after():
    client, session, sleeps = make_client(
        FakeResponse(429, {'error': {'code': 429, 'message': 'Rate limit'}}, headers={'Retry-After': '2'}),
        completion())

    assert client.chat('a', 'm/x').text == 'Ответ'
    assert sleeps == [2.0]
    assert len(session.calls) == 2


def test_retry_on_network_error_with_exponential_backoff():
    client, _, sleeps = make_client(requests.ConnectionError('boom'), requests.Timeout('slow'), completion(),
                                    backoff=0.5)

    assert client.chat('a', 'm/x').text == 'Ответ'
    assert sleeps == [0.5, 1.0]


def test_retryable_error_inside_successful_response():
    error_body = {'choices': [{'index': 0, 'finish_reason': 'error', 'message': {'content': ''},
                               'error': {'code': 502, 'message': 'Provider returned error'}}]}
    client, session, sleeps = make_client(FakeResponse(200, error_body), completion())

    assert client.chat('a', 'm/x').text == 'Ответ'
    assert len(session.calls) == 2


def test_non_retryable_error_is_raised_immediately():
    client, session, sleeps = make_client(
        FakeResponse(401, {'error': {'code': 401, 'message': 'No auth credentials found'}}))

    with pytest.raises(OpenRouterError) as exc:
        client.chat('a', 'm/x')

    assert exc.value.status == 401
    assert 'No auth credentials found' in str(exc.value)
    assert len(session.calls) == 1
    assert sleeps == []


def test_error_in_body_with_200_status_is_raised():
    client, _, _ = make_client(FakeResponse(200, {'error': {'code': 400, 'message': 'Bad request',
                                                            'metadata': {'provider_name': 'Novita'}}}))

    with pytest.raises(OpenRouterError) as exc:
        client.chat('a', 'm/x')

    assert exc.value.status == 400
    assert exc.value.metadata == {'provider_name': 'Novita'}


def test_retries_are_limited():
    unavailable = FakeResponse(503, {'error': {'code': 503, 'message': 'No endpoints found'}})
    client, session, sleeps = make_client(*[unavailable] * 3, max_retries=2, backoff=1.0)

    with pytest.raises(OpenRouterError) as exc:
        client.chat('a', 'm/x')

    assert exc.value.status == 503
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_non_json_error_response():
    client, _, _ = make_client(FakeResponse(400, None, text='<html>Bad Gateway</html>'))

    with pytest.raises(OpenRouterError, match='Bad Gateway'):
        client.chat('a', 'm/x')


def test_response_without_choices_is_rejected():
    with pytest.raises(OpenRouterError):
        ChatResult.from_response({'id': 'x', 'choices': []})


def test_cache_avoids_repeated_requests_and_cost():
    client, session, _ = make_client(completion(cost=0.003), completion(text='Другой', cost=0.003),
                                     cache=ResponseCache())

    first = client.chat('a', 'm/x')
    second = client.chat('a', 'm/x')
    third = client.chat('a', 'm/x', sample_id=1)

    assert (first.cached, second.cached, third.cached) == (False, True, False)
    assert second.text == first.text
    assert third.text == 'Другой'
    assert len(session.calls) == 2
    assert client.n_cache_hits == 1
    assert client.total_cost == pytest.approx(0.006)


def test_cache_hit_does_not_need_api_key(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    cache = ResponseCache()
    payload = OpenRouterClient.build_payload('a', 'm/x')
    cache.set(make_cache_key(payload), payload, completion()._body)

    result = OpenRouterClient(cache=cache, session=FakeSession()).chat('a', 'm/x')

    assert result.cached and result.text == 'Ответ'


def test_cache_can_be_bypassed():
    client, session, _ = make_client(completion(), completion(), cache=ResponseCache())

    client.chat('a', 'm/x')
    client.chat('a', 'm/x', use_cache=False)

    assert len(session.calls) == 2


def test_budget_stops_new_requests():
    client, session, _ = make_client(completion(cost=0.002), completion(cost=0.002), max_cost=0.001)

    client.chat('a', 'm/x')
    with pytest.raises(BudgetExceededError):
        client.chat('b', 'm/x')

    assert len(session.calls) == 1


def test_list_models_passes_filter():
    client, session, _ = make_client(FakeResponse(200, {'data': [{'id': 'm/x'}]}))

    assert client.list_models(['logprobs', 'seed']) == [{'id': 'm/x'}]
    assert session.calls[0]['url'] == URL + '/models'
    assert session.calls[0]['params'] == {'supported_parameters': 'logprobs,seed'}


def test_list_models_works_without_api_key(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    session = FakeSession(FakeResponse(200, {'data': []}))

    assert OpenRouterClient(session=session).list_models() == []
    assert 'Authorization' not in session.calls[0]['headers']


def test_list_endpoints_url_and_validation():
    client, session, _ = make_client(FakeResponse(200, {'data': {'id': 'meta-llama/llama-3.1-8b-instruct',
                                                                 'endpoints': []}}))

    assert client.list_endpoints('meta-llama/llama-3.1-8b-instruct')['endpoints'] == []
    assert session.calls[0]['url'] == URL + '/models/meta-llama/llama-3.1-8b-instruct/endpoints'
    with pytest.raises(ValueError):
        client.list_endpoints('llama')


def test_response_cache_persists_on_disk(tmp_path):
    path = str(tmp_path / 'cache.sqlite')
    cache = ResponseCache(path)
    cache.set('k', {'model': 'm/x'}, {'choices': []})
    cache.close()

    reopened = ResponseCache(path)
    assert reopened.get('k') == {'choices': []}
    assert reopened.get('missing') is None
    assert len(reopened) == 1
    reopened.close()


def test_cache_key_depends_on_payload_and_sample():
    payload = {'model': 'm/x', 'messages': [{'role': 'user', 'content': 'a'}]}

    assert make_cache_key(payload) == make_cache_key(dict(reversed(list(payload.items()))))
    assert make_cache_key(payload) != make_cache_key(payload, sample_id=1)
    assert make_cache_key(payload) != make_cache_key({**payload, 'seed': 1})
