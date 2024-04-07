import os

import pandas as pd

os.environ['TEST_MODE_ON_LOW_SPEC_PC'] = 'True'
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from explainitall.metrics.RougeAndPPL.Metrics import Metric_ppl, MetricRougeL, MetricRougeN
from explainitall.metrics.RougeAndPPL.Metrics_calculator import Metrics_calculator
from explainitall.metrics.RougeAndPPL.create_database import data_table, ENGINE
from explainitall.metrics.RougeAndPPL.helpers import (
    get_max_dataset_version, generate_candidates, calculate_average_metric_values,
    insert_new_record)

# Параметры для загрузки модели и датасета
MODEL_NAME_OR_PATH = "distilgpt2"  # Пример: "gpt2" или путь к вашей модели
DATASET_FILE_PATH = "/home/laptop/PycharmProjects/codearchive/ExplainitAllBugfixes/examples/example_data/metrix.csv"  # Путь к вашему CSV файлу
DATASET_TITLE = "DatasetTitle"  # Название вашего датасета


def load_model(model_name_or_path):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    return model, tokenizer


def load_dataset(csv_file_path, title, conn):
    dataframe = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')
    dataset_name = title
    dataset_version = get_max_dataset_version(dataset_name, conn, data_table) + 1
    contexts = list(dataframe['context'].values)
    references = list(dataframe['reference'].values)
    return dataset_name, dataset_version, contexts, references


def main():
    conn = ENGINE.connect()

    model, tokenizer = load_model(MODEL_NAME_OR_PATH)
    dataset_name, dataset_version, contexts, references = load_dataset(DATASET_FILE_PATH, DATASET_TITLE, conn)

    calculator = Metrics_calculator(tokenizer)
    calculator.add_metric('PPL', Metric_ppl(model, stride=512))
    calculator.add_metric('R3', MetricRougeN(3))
    calculator.add_metric('R5', MetricRougeN(5))
    calculator.add_metric('R-L', MetricRougeL(3))

    candidates = generate_candidates(model, tokenizer, contexts, model.config.n_positions, max_new_tokens=10)

    res = calculator.calculate(contexts, references, candidates)
    metric_values = calculate_average_metric_values(res)

    insert_new_record(data_table, conn, MODEL_NAME_OR_PATH, dataset_name, dataset_version, metric_values)

    print("Metrics Calculated and Recorded:", metric_values)


if __name__ == "__main__":
    main()
