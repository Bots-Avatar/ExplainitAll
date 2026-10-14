[![Downloads](https://img.shields.io/pypi/dm/explainitall.svg)](https://pypi.org/project/explainitall/)

# **ExplainitAll**

**ExplainitAll** — это библиотека для интерпретируемого ИИ, предназначенная для интерпретации генеративных моделей (
GPT-like), и векторизаторов, например, Sbert. Библиотека предоставляет пользователям инструменты для анализа и понимания
работы этих сложных моделей. Кроме того, содержит модули RAG QA, fast_tuning и пользовательский интерфейс.

Через [OpenRouter](https://openrouter.ai) библиотека интерпретирует и закрытые модели, доступные только по API:
генеративные модели, модели векторных представлений и модели переранжирования (подробнее в разделе
[Интерпретация закрытых моделей через OpenRouter](#интерпретация-закрытых-моделей-через-openrouter)).

---

* [Примеры использования](https://github.com/Bots-Avatar/ExplainitAll/tree/main/examples)
* [Исходный код библиотеки](https://github.com/Bots-Avatar/ExplainitAll/tree/main/explainitall)
* [Документация](https://github.com/Bots-Avatar/ExplainitAll/wiki)

## Модели:

* Дистиллированный [Sbert](https://huggingface.co/FractalGPT/SbertDistil)
* Дистиллированный [Sbert](https://huggingface.co/FractalGPT/SbertSVDDistil) с применением SVD разложения, для ускорения
  инференса и обучения
* [FRED T5](https://huggingface.co/FractalGPT/FRED-T5-Interp), обученный под задачу RAG, для ответов на вопросы по
  интепретации генеративной gpt-подобной сети.
* [FRED T5](https://huggingface.co/FractalGPT/FredT5-Large-Instruct-Context), небольшой T5 обученный для instruct задач с учетом контекста
---

## Интерпретация закрытых моделей через OpenRouter

Библиотека работает с [OpenRouter](https://openrouter.ai): через единый API доступны сотни моделей, в том числе
закрытые (GPT, Claude, Gemini), к весам которых нет доступа. Градиентные методы к таким моделям неприменимы, поэтому
ExplainitAll интерпретирует их как черный ящик: скрывает слова и тематические кластеры во входном тексте и измеряет,
как меняется результат модели.

Возможности по типам моделей:

| Тип модели | Что показывает интерпретация | Класс |
|---|---|---|
| Генеративные модели, включая закрытые | важность кластеров запроса по изменению смысла ответа и влияние кластеров запроса на кластеры ответа | `ExplainerAPI` |
| Генеративные модели с логарифмами вероятностей (около 130 моделей: GPT-4o, Llama, Qwen, DeepSeek и другие) | во сколько раз падает вероятность исходного ответа без кластера | `ExplainerAPI(logprobs=True)` |
| Открытые модели, доступные через API | градиентные атрибуции (Integrated Gradients) на локальной копии модели для ответа, полученного через API | `SurrogateExplainer` |
| Модели векторных представлений (OpenAI, Gemini, bge-m3, multilingual-e5, Qwen3 Embedding и другие) | вклад слов и кластеров в вектор текста и в близость запроса и документа | `EmbedderExplainer`, `APIEmbedder` |
| Модели переранжирования (Cohere, Voyage, Qwen3 Reranker) | какие слова документа и запроса определяют релевантность и меняется ли самый релевантный документ | `RerankerExplainer` |

Каждый вывод проверяется на надежность. Эффект скрытия кластера сравнивается со случайными колебаниями модели,
которые оцениваются по нескольким прогонам исходного запроса. Модуль `validation` строит кривые удаления: они
показывают, насколько найденный порядок важности лучше случайного и насколько согласуются разные методы.

### Быстрый старт

Ключ API задается переменной окружения `OPENROUTER_API_KEY`.

```python
import gensim
from sentence_transformers import SentenceTransformer

from explainitall.api_interp.cache import ResponseCache
from explainitall.api_interp.capabilities import get_endpoints, pinned_provider, select_endpoint
from explainitall.api_interp.explainer import ExplainerAPI
from explainitall.api_interp.openrouter_client import OpenRouterClient
from explainitall.gpt_like_interp.downloader import DownloadManager

nlp_model = gensim.models.KeyedVectors.load_word2vec_format(
    DownloadManager.load_zip('http://vectors.nlpl.eu/repository/20/180.zip'), binary=True)
clusters = [
    {'name': 'Животные', 'centroid': ['собака', 'кошка'], 'top_k': 140},
    {'name': 'Болезни', 'centroid': ['простуда', 'грипп'], 'top_k': 20},
    {'name': 'Лекарства', 'centroid': ['уколы', 'таблетки'], 'top_k': 160},
]

client = OpenRouterClient(cache=ResponseCache('openrouter_cache.sqlite'), max_cost=1.0)
model = 'meta-llama/llama-3.1-8b-instruct'
endpoint = select_endpoint(get_endpoints(client, model))  # поставщик с поддержкой logprobs

explainer = ExplainerAPI(client, model, nlp_model=nlp_model,
                         embedder=SentenceTransformer('FractalGPT/SbertDistil'),
                         provider=pinned_provider(endpoint, data_collection='deny'),
                         seed=0, logprobs=True)
prompt = 'У моей кошки грипп. Чем лечить кошку?'
print(explainer.plan(prompt, clusters).n_requests_max)  # оценка числа запросов до запуска
result = explainer.interpret(prompt, clusters)
print(result.importance_df[['cluster', 'importance', 'logprob_drop', 'logprob_significant']])
result.show_cluster_imp_aggr_heatmap()
```

Для закрытой модели без логарифмов вероятностей (например, `anthropic/claude-sonnet-4.5`) укажите `logprobs=False`
и не передавайте `provider`.

Модели векторных представлений и переранжирования:

```python
from explainitall.api_interp.embedding_interp import APIEmbedder, EmbedderExplainer
from explainitall.api_interp.rerank_interp import RerankerExplainer

embedder = APIEmbedder(client, 'baai/bge-m3')
explanation = EmbedderExplainer(embedder, nlp_model=nlp_model).similarity_attribution(
    'чем лечить грипп у кошки', 'Кошке при гриппе назначают противовирусные препараты')
print(explanation.similarity, explanation.top(5))

reranker = RerankerExplainer(client, 'cohere/rerank-v3.5', nlp_model=nlp_model)
print(reranker.document_word_importance('лечение гриппа у кошки', 'Кошке при гриппе помогут уколы'))
print(reranker.query_cluster_importance('кошка и грипп', ['Грипп у кошек', 'Корм для собак'], clusters))
```

Проверочный эксперимент на наборе запросов находится в
[examples/validate_api_interpretation.py](examples/validate_api_interpretation.py). С ключом OpenRouter он работает
с моделью за API, а с параметром `--backend local` бесплатно повторяет ту же схему на локальной модели Hugging Face.

### Что важно знать

* Стоимость. Каждый ответ API содержит цену запроса, `OpenRouterClient(max_cost=...)` прекращает отправку запросов
  при достижении предела, а кеш `ResponseCache` делает повторный запуск бесплатным.
* Конфиденциальность. Тексты уходят стороннему поставщику. Для чувствительных данных задайте
  `pinned_provider(endpoint, data_collection='deny', zdr=True)`. Кеш хранит тексты на диске в открытом виде.
* Воспроизводимость. По умолчанию генерация идет с `temperature=0`, а `pinned_provider` закрепляет одного поставщика и
  не дает ему молча отбросить параметры запроса.
* Модель-двойник. Для моделей больше 3 млрд параметров нужна видеокарта, а закрытые веса (например, Llama 3) требуют
  принять лицензию на Hugging Face.

---

## Перечень направлений прикладного использования:

Результаты могут применяться в следующих областях: любые вопрос-ответные системы или классификаторы критических
отраслей (медицина, строительство, космос, право и т.п.). Типовой сценарий применения, например для медицины следующий:
разработчик конечного продукта, такого как например система поиска противопоказаний у лекарств в тесном взаимодействии в
заказчиком(врачом, поликлиникой и т.п.) создает набор кластеров тематической области, дообучает трансформерную модель (
GPT-like: например, семейств ruGPT3 и GPT2) на текстах вопрос-ответ, и на затем в режиме эксплуатации данной, уже
готовой Вопросно-ответной системы подключает библиотеку ExplainitAll для того, чтобы она давал аналитическую оценку –
насколько «надежными» и доверенными являются ответы вопросно-ответной системы на основе результата интерпретации –
действительно ли при ответе на вопросы пользователя система обращала внимание на важные для отрасли кластеры.

Разработанная библиотека может быть адаптирована как модуль конечного продукта - ассистента врача,
инженера-конструктора, юриста, бухгалтера. Для государственного сектора библиотека может быть полезна т.к. помогает
доверять RAG системам при ответах по налогам, регламентам проведения закупочных процедур, руководствам пользователей
информационных систем, нормативно-правовым актам регулирования. Для промышленных предприятий библиотека применима в
работе с регламентами, руководствами по эксплуатации и обслуживанию сложного технического оборудования, т.к. позволяет
оценивать учет в ответах QA систем понимание специальных, важных для отрасли сокращений, наименования, аббревиатур,
номенклатурных обозначений.

## Характеристики:

* Поддерживаемые версии Python: от 3.9 до 3.11

Тестовый стенд:

* Операционная система Ubuntu 22.04.3 LTS
* Драйвер: NVIDIA версия 535.104.05
* CUDA версия 12.2
* Python 3.10.12


* Процессор AMD Ryzen 3 3200G OEM (частота: 3600 МГц, количество ядер: 4)
* Оперативная память 16 GB

* Графический процессор
    * Модель: nVidia TU104GL [Tesla T4]
    * Видеопамять 16 GB

## Лицензия

* Лицензия [Apache-2.0 license](https://github.com/Bots-Avatar/ExplainitAll/tree/main#Apache-2.0-1-ov-file)
