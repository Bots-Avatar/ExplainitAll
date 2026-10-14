"""
Проверочный эксперимент интерпретации модели за API.

Сравнивает на наборе промптов три способа найти важные кластеры:
исключение кластеров с оценкой по смыслу, по падению logprobs и атрибуции открытого двойника (IG).
Для каждого строятся удаляющие кривые и сравниваются со случайным порядком скрытия кластеров.

Модель за OpenRouter (нужен OPENROUTER_API_KEY, провайдер с logprobs выбирается автоматически):
    python examples/validate_api_interpretation.py --backend openrouter --model qwen/qwen-2.5-7b-instruct \
        --twin Qwen/Qwen2.5-7B-Instruct --max-cost 0.5

Та же схема на локальной модели, бесплатно (модель служит и «API», и двойником):
    python examples/validate_api_interpretation.py --backend local --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLUSTERS = json.loads((ROOT / 'examples' / 'example_data' / 'clusters.json').read_text(encoding='utf-8'))

PROMPTS = [
    'У моей кошки грипп и аллергия на таблетки. Чем лечить кошку? Ответь кратко.',
    'Собака заболела простудой, ей назначили уколы. Можно ли гулять с собакой? Ответь кратко.',
    'Заяц чихает, похоже на простуду. Какие таблетки безопасны для зайца? Ответь кратко.',
    'У кошки аллергия на корм, а у собаки грипп. Кого из них нужно лечить уколами? Ответь кратко.',
    'При гриппе врач выписал микстуры и таблетки. Можно ли давать их собаке? Ответь кратко.',
    'У ребёнка ОРВИ и аллергия на антибиотики. Какие лекарства от простуды ему подойдут? Ответь кратко.',
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=['openrouter', 'local'], required=True)
    parser.add_argument('--model', required=True, help='id модели OpenRouter или репозиторий/путь HF для local')
    parser.add_argument('--twin', default='auto',
                        help="открытые веса двойника: репозиторий/путь HF, 'auto' (hugging_face_id или локальная "
                             "модель) или 'none'")
    parser.add_argument('--max-cost', type=float, default=0.5, help='предел расходов на API, $')
    parser.add_argument('--cache', default=None, help='файл кеша ответов API (повторный запуск бесплатен)')
    parser.add_argument('--n-baseline', type=int, default=3)
    parser.add_argument('--n-random', type=int, default=5)
    parser.add_argument('--max-answer-tokens', type=int, default=24)
    parser.add_argument('--ig-steps', type=int, default=16)
    parser.add_argument('--nlp-model-url', default='http://vectors.nlpl.eu/repository/20/180.zip')
    parser.add_argument('--embedder', default='FractalGPT/SbertDistil')
    parser.add_argument('--device', default=None)
    parser.add_argument('--out-dir', default='validation_results')
    return parser.parse_args(argv)


def load_nlp_model(url):
    import gensim
    from explainitall.gpt_like_interp.downloader import DownloadManager
    return gensim.models.KeyedVectors.load_word2vec_format(DownloadManager.load_zip(url), binary=True)


def build_backend(args):
    """
    Клиент «API», провайдер и двойник: (client, provider, готовая модель inseq или None, репозиторий HF или None).
    Если двойник не задан явно, SurrogateExplainer найдёт открытые веса по hugging_face_id модели OpenRouter.
    """
    explicit_twin = None if args.twin in ('auto', 'none') else args.twin

    if args.backend == 'local':
        from explainitall.api_interp.local_client import LocalChatClient
        client = LocalChatClient.from_pretrained(args.model, device=args.device)
        twin_model = None
        if args.twin == 'auto':  # двойник — та же локальная модель, загружать веса второй раз не нужно
            from inseq import load_model
            twin_model = load_model(client.model, attribution_method='integrated_gradients',
                                    tokenizer=client.tokenizer)
        return client, None, twin_model, explicit_twin

    from explainitall.api_interp.cache import ResponseCache
    from explainitall.api_interp.capabilities import get_endpoints, pinned_provider, select_endpoint
    from explainitall.api_interp.openrouter_client import OpenRouterClient

    client = OpenRouterClient(cache=ResponseCache(args.cache) if args.cache else None, max_cost=args.max_cost,
                              app_title='ExplainitAll validation')
    endpoint = select_endpoint(get_endpoints(client, args.model))
    if endpoint is None:
        raise SystemExit(f'У {args.model} нет провайдера с logprobs: выберите другую модель')
    print(f'Провайдер: {endpoint.tag} ({endpoint.quantization})')
    return client, pinned_provider(endpoint), None, explicit_twin


def main(argv=None):
    args = parse_args(argv)
    from sentence_transformers import SentenceTransformer
    from explainitall.api_interp.explainer import ExplainerAPI
    from explainitall.api_interp.surrogate import SurrogateExplainer
    from explainitall.api_interp.validation import ValidationExperiment, summarize

    nlp_model = load_nlp_model(args.nlp_model_url)
    embedder = SentenceTransformer(args.embedder)
    client, provider, twin_model, twin_hf = build_backend(args)

    explainer = ExplainerAPI(client, args.model, nlp_model=nlp_model, embedder=embedder, provider=provider,
                             seed=0, max_tokens=64, n_baseline=args.n_baseline, logprobs=True,
                             max_answer_tokens=args.max_answer_tokens, max_workers=4)
    twin = None
    if args.twin != 'none':
        twin = SurrogateExplainer(client, args.model, nlp_model=nlp_model, attribution_model=twin_model,
                                  hf_model=twin_hf, provider=provider, answer_max_tokens=args.max_answer_tokens,
                                  steps=args.ig_steps, batch_size=args.ig_steps)

    plan_total = sum(explainer.plan(p, CLUSTERS).n_requests_max for p in PROMPTS)
    print(f'Промптов: {len(PROMPTS)}, запросов к модели не больше {plan_total} (без учёта удаляющих кривых)')

    experiment = ValidationExperiment(explainer, twin=twin, curve_measure='logprob', n_random=args.n_random)
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        try:
            result = experiment.run_one(prompt, CLUSTERS)
        except ValueError as e:
            print(f'[{i}] пропущен: {e}')
            continue
        results.append(result)
        print(f'\n[{i}] {prompt}\n    ответ: {result.answer.strip()[:120]}')
        print(f'    кластеры: {result.clusters}; AOPC: '
              + ', '.join(f'{m}={v:.3f}' for m, v in result.aopc.items())
              + f'; случайный порядок={result.random_aopc_mean:.3f}; ${result.cost:.4f}')
        for warning in result.warnings:
            print(f'    ! {warning}')

    methods, agreements = summarize(results)
    pd.set_option('display.width', 200)
    print('\nМетоды (удаляющие кривые):\n', methods.to_string(index=False))
    print('\nСогласованность рангов:\n', agreements.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{'prompt': r.prompt[0]['content'], 'answer': r.answer, 'clusters': r.clusters, 'cost': r.cost,
             'random_aopc': r.random_aopc_mean, **{f'aopc_{m}': v for m, v in r.aopc.items()},
             **{f'rank_{m}': list(s.index) for m, s in r.rankings.items()}} for r in results]
    pd.DataFrame(rows).to_csv(out_dir / 'prompts.csv', index=False, encoding='utf-8')
    methods.to_csv(out_dir / 'methods.csv', index=False, encoding='utf-8')
    agreements.to_csv(out_dir / 'agreements.csv', index=False, encoding='utf-8')
    print(f'\nРезультаты сохранены в {out_dir.resolve()}')
    return results


if __name__ == '__main__':
    main()
