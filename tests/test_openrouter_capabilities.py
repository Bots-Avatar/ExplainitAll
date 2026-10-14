import pytest

from explainitall.api_interp.capabilities import (EndpointInfo, ModelInfo, get_endpoints, get_models,
                                                  pinned_provider, select_endpoint)

BASE = ['max_tokens', 'temperature', 'top_p', 'stop', 'seed']
LOGPROBS = BASE + ['logprobs', 'top_logprobs']


def endpoint(tag, quantization, params, prompt='0.00000002', completion='0.00000004', status=0, uptime=100):
    return {'name': tag, 'model_id': 'meta-llama/llama-3.1-8b-instruct', 'provider_name': tag.split('/')[0],
            'tag': tag, 'quantization': quantization, 'supported_parameters': params,
            'pricing': {'prompt': prompt, 'completion': completion, 'discount': 0},
            'status': status, 'uptime_last_30m': uptime, 'context_length': 131072, 'max_completion_tokens': 16384}


# Наборы параметров повторяют ответ /models/meta-llama/llama-3.1-8b-instruct/endpoints
LLAMA_ENDPOINTS = [
    endpoint('deepinfra/fp8', 'fp8', BASE),
    endpoint('novita/fp8', 'fp8', LOGPROBS, completion='0.00000005'),
    endpoint('groq', 'unknown', BASE, prompt='0.00000005', completion='0.00000008'),
    endpoint('coreweave/bf16', 'bf16', LOGPROBS, prompt='0.00000022', completion='0.00000022'),
]


class FakeClient:
    def __init__(self, models=None, endpoints=None):
        self.models = models or []
        self.endpoints = endpoints or []
        self.models_filter = None

    def list_models(self, supported_parameters=None):
        self.models_filter = supported_parameters
        return self.models

    def list_endpoints(self, model):
        return {'id': model, 'endpoints': self.endpoints}


def parsed(endpoints=LLAMA_ENDPOINTS):
    return [EndpointInfo.from_dict(e) for e in endpoints]


def test_endpoint_info_parsing():
    e = EndpointInfo.from_dict(LLAMA_ENDPOINTS[3])

    assert e.tag == 'coreweave/bf16'
    assert e.quantization_rank < EndpointInfo.from_dict(LLAMA_ENDPOINTS[1]).quantization_rank
    assert e.supports(['logprobs', 'seed'])
    assert e.available
    assert e.estimate_cost(prompt_tokens=1000, completion_tokens=100, n_requests=10) == pytest.approx(
        10 * (1000 + 100) * 0.00000022)


def test_select_prefers_logprobs_then_precision():
    assert select_endpoint(parsed()).tag == 'coreweave/bf16'


def test_select_with_allowed_quantizations():
    assert select_endpoint(parsed(), allowed_quantizations=['FP8']).tag == 'novita/fp8'


def test_select_skips_unavailable_endpoints():
    endpoints = LLAMA_ENDPOINTS[:3] + [endpoint('coreweave/bf16', 'bf16', LOGPROBS, status=-2)]

    assert select_endpoint(parsed(endpoints)).tag == 'novita/fp8'
    assert select_endpoint(parsed(endpoints), include_unavailable=True).tag == 'coreweave/bf16'


def test_select_prefers_seed_support_over_precision():
    endpoints = [endpoint('a/bf16', 'bf16', ['logprobs', 'top_logprobs']), endpoint('b/fp8', 'fp8', LOGPROBS)]

    assert select_endpoint(parsed(endpoints)).tag == 'b/fp8'


def test_select_uses_uptime_then_price():
    endpoints = [endpoint('a/fp8', 'fp8', LOGPROBS, uptime=90), endpoint('b/fp8', 'fp8', LOGPROBS, uptime=99),
                 endpoint('c/fp8', 'fp8', LOGPROBS, uptime=99, prompt='0.000001')]

    assert select_endpoint(parsed(endpoints)).tag == 'b/fp8'


def test_select_returns_none_without_logprobs():
    assert select_endpoint(parsed(LLAMA_ENDPOINTS[:1] + LLAMA_ENDPOINTS[2:3])) is None


def test_select_without_requirements():
    assert select_endpoint(parsed(), required=(), preferred=()).tag == 'coreweave/bf16'


def test_pinned_provider_disables_fallbacks():
    settings = pinned_provider(EndpointInfo.from_dict(LLAMA_ENDPOINTS[1]), data_collection='deny', zdr=True)

    assert settings.to_dict() == {'order': ['novita/fp8'], 'allow_fallbacks': False, 'require_parameters': True,
                                  'data_collection': 'deny', 'zdr': True}


def test_get_endpoints_parses_client_response():
    tags = [e.tag for e in get_endpoints(FakeClient(endpoints=LLAMA_ENDPOINTS), 'meta-llama/llama-3.1-8b-instruct')]

    assert tags == ['deepinfra/fp8', 'novita/fp8', 'groq', 'coreweave/bf16']


def test_get_models_filters_locally_and_on_server():
    models = [{'id': 'meta-llama/llama-3.1-8b-instruct', 'name': 'Llama',
               'hugging_face_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
               'supported_parameters': LOGPROBS, 'pricing': {'prompt': '0.00000002', 'completion': '0.00000004'}},
              {'id': 'anthropic/claude', 'name': 'Claude', 'hugging_face_id': '',
               'supported_parameters': BASE, 'pricing': {'prompt': '0.000003', 'completion': '0.000015'}}]
    client = FakeClient(models=models)

    result = get_models(client, ['logprobs'])

    assert client.models_filter == ['logprobs']
    assert [m.id for m in result] == ['meta-llama/llama-3.1-8b-instruct']
    assert result[0].hugging_face_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct'


def test_model_info_handles_missing_fields():
    m = ModelInfo.from_dict({'id': 'x/y', 'pricing': {'prompt': 'bad'}})

    assert (m.name, m.hugging_face_id, m.prompt_price, m.supported_parameters) == ('x/y', None, 0.0, frozenset())
