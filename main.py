import gensim
from inseq import load_model

from explainitall.gpt_like_interp import viz, interp
from explainitall.gpt_like_interp.downloader import DownloadManager


def load_nlp_model(nlp_model_url):
    nlp_model_path = DownloadManager.load_zip(nlp_model_url)
    return gensim.models.KeyedVectors.load_word2vec_format(nlp_model_path, binary=True)


# 'ID': 180
# 'Размер вектора': 300
# 'Корпус': 'Russian National Corpus'
# 'Размер словаря': 189193
# 'Алгоритм': 'Gensim Continuous Bag-of-Words'
# 'Лемматизация': True

nlp_model = load_nlp_model('http://vectors.nlpl.eu/repository/20/180.zip')


def load_gpt_model(gpt_model_name):
    return load_model(model=gpt_model_name,
                      attribution_method="integrated_gradients")


# 'Фреймворк': 'transformers'
# 'Тренировочные токены': '80 млрд'
# 'Размер контекста': 2048
gpt_model = load_gpt_model("sberbank-ai/rugpt3small_based_on_gpt2")

clusters_discr = [
    {'name': 'Животные', 'centroid': ['собака', 'кошка', 'заяц'], 'top_k': 140},
    {'name': 'Лекарства', 'centroid': ['уколы', 'таблетки', 'микстуры'], 'top_k': 160},
    {'name': 'Болезни', 'centroid': ['простуда', 'орви', 'орз', 'грипп'], 'top_k': 20},
    {'name': 'Аллергия', 'centroid': ['аллергия'], 'top_k': 20}
]

explainer = interp.ExplainerGPT2(gpt_model=gpt_model, nlp_model=nlp_model)

expl_data = explainer.interpret(
    input_texts='у кошки грипп и аллергия на антибиотбиотики вопрос: чем лечить кошку? ответ:',
    generated_texts='лечичичичите ее уколами',
    clusters_description=clusters_discr,
    batch_size=50,
    steps=34,
    # max_new_tokens=19
)

print("\nКарта важности кластеров")
print(expl_data.cluster_imp_df)

print("\nТепловая карта важности кластеров")
expl_data.show_cluster_imp_heatmap()

print("\nКарта важности кластеров, группированная")
print(expl_data.cluster_imp_aggr_df)

print("\nТепловая карта важности кластеров, группированная")
expl_data.show_cluster_imp_aggr_heatmap()

print("\nКарта важности слов")
print(expl_data.word_imp_df)

print("\nТепловая карта важности слов")
expl_data.show_word_imp_heatmap()

print("\nКарта важности слов, нормированная")
print(expl_data.word_imp_norm_df)

print("\nТепловая карта важности слов, нормированная")
expl_data.show_word_imp_norm_heatmap()

print("\nГистограмма распределения")
viz.show_distribution_histogram(expl_data.attributions.array)
print("\nГрафик распределения")
viz.show_distribution_plot(expl_data.attributions.array)
