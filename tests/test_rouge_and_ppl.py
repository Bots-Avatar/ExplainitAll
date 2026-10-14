import math

import pytest

from explainitall.metrics.RougeAndPPL.rouge_N import rouge_N

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from explainitall.metrics.RougeAndPPL.Metrics import Metric_ppl


def test_rouge_n_without_common_ngrams():
    assert rouge_N([1, 2, 3, 4], [5, 6, 7, 8], 2) == {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


def test_rouge_n_partial_overlap():
    res = rouge_N([1, 2, 3, 4], [1, 2, 3, 9], 2)
    assert res['precision'] == pytest.approx(2 / 3)
    assert res['recall'] == pytest.approx(2 / 3)
    assert res['f1'] == pytest.approx(2 / 3)


def test_rouge_n_identical():
    assert rouge_N([1, 2, 3], [1, 2, 3], 2)['f1'] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def tiny_gpt2():
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=50, n_positions=16, n_embd=16, n_layer=2, n_head=2)
    return transformers.GPT2LMHeadModel(config).eval()


def exact_perplexity(model, ids, stride):
    """Перплексия скользящим окном, посчитанная напрямую по log-вероятностям каждого токена"""
    max_length = model.config.n_positions
    nll, count, prev_end = 0.0, 0, 0
    for start in range(0, len(ids), stride):
        end = min(start + max_length, len(ids))
        with torch.no_grad():
            logits = model(torch.tensor(ids[start:end]).unsqueeze(0)).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        for pos in range(max(prev_end, start + 1), end):
            nll -= log_probs[pos - start - 1, ids[pos]].item()
            count += 1
        prev_end = end
        if end == len(ids):
            break
    return math.exp(nll / count)


@pytest.mark.parametrize("stride", [4, 8, 16, 20])
def test_ppl_matches_exact_sliding_window(tiny_gpt2, stride):
    ids = [(i * 7 + 3) % 50 for i in range(40)]
    ids[5] = ids[17] = 0  # токен 0 не должен влиять на маскирование контекста

    value = Metric_ppl(tiny_gpt2, stride=stride)._calculate_perplexity(ids)['value']

    assert value == pytest.approx(exact_perplexity(tiny_gpt2, ids, stride), rel=1e-4)


def test_ppl_single_window_equals_model_loss(tiny_gpt2):
    ids = list(range(1, 12))
    with torch.no_grad():
        loss = tiny_gpt2(torch.tensor(ids).unsqueeze(0), labels=torch.tensor(ids).unsqueeze(0)).loss

    value = Metric_ppl(tiny_gpt2, stride=16)._calculate_perplexity(ids)['value']

    assert value == pytest.approx(math.exp(loss.item()), rel=1e-5)


def test_ppl_single_token_is_nan(tiny_gpt2):
    assert math.isnan(Metric_ppl(tiny_gpt2, stride=8)._calculate_perplexity([3])['value'])
