import re
import time
from datetime import datetime

import pandas as pd
import sqlalchemy


def calculate_average_metric_values(calculated_metrics):
    res = {}

    for metric in calculated_metrics:
        sub_metric_average_values = {}
        for text_evaluation_result in calculated_metrics[metric]:
            for sub_metric in text_evaluation_result:
                if sub_metric not in sub_metric_average_values:
                    sub_metric_average_values[sub_metric] = 0.0
                sub_metric_average_values[sub_metric] += text_evaluation_result[sub_metric]
        for sub in sub_metric_average_values:
            sub_metric_average_values[sub] = sub_metric_average_values[sub] / len(calculated_metrics[metric])
        if 'f1' in sub_metric_average_values:
            res[metric] = sub_metric_average_values['f1']
        elif 'value' in sub_metric_average_values:
            res[metric] = sub_metric_average_values['value']
        else:
            res[metric] = 0.0

    return res


def fbeta_score(precision, recall, beta=1):
    if precision == 0.0 and recall == 0.0:
        return 0.0

    if beta == 1:
        return 2 * precision * recall / (precision + recall)
    return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)


def generate_candidates(model, tokenizer, sentences, max_length=128, max_new_tokens=100):
    candidates = []
    for sentence in sentences:
        encoded_input = tokenizer(sentence, truncation=True, max_length=max_length, return_tensors='pt')
        encoded_input = encoded_input.to(model.device)
        res = model.generate(**encoded_input, max_new_tokens=max_new_tokens)
        candidate = tokenizer.decode(res[0], skip_special_tokens=True)
        candidates.append(candidate)
    return candidates


def get_all_words_from_text(text):
    words = re.findall(r"[\w]+", text)
    return words


def get_max_dataset_version(dataset_name, conn, data_table):
    statement = sqlalchemy.select(sqlalchemy.func.max(data_table.c.dataset_version)).where(
        data_table.c.dataset_name == dataset_name)

    records = []

    for row in conn.execute(statement):
        records.append(row)

    res = records[0][0]

    if res is None:
        return -1

    return records[0][0]


def get_records_from_database(data_table, conn, specific_column_value=None):
    records = []

    statement = data_table.select()

    if specific_column_value is not None:
        for k in specific_column_value:
            col = sqlalchemy.sql.column(k)
            statement = statement.where(col == specific_column_value[k])

    for row in conn.execute(statement):
        records.append(row)

    return records


def insert_new_record(data_table, conn, model_name, dataset_name, dataset_version, metric_values):
    record = {'model_name': model_name,
              'timestamp': int(time.time()),
              'dataset_name': dataset_name,
              'dataset_version': dataset_version}
    for m in metric_values:
        record[m] = metric_values[m]

    statement = data_table.insert().values(**record)
    conn.execute(statement)
    conn.commit()


def make_dataframe_from_history_records(records):
    columns = ['model_name', 'date', 'dataset_name', 'dataset_version', 'PPL', 'R3', 'R5', 'R-L']
    res_records = []

    for rec in records:
        r = list(rec[1:])
        d = datetime.fromtimestamp(r[1])
        r[1] = str(d.day) + '.' + str(d.month) + '.' + str(d.year)
        for i in range(len(columns[4:])):
            r[i + 4] = round(r[i + 4], 2)
        res_records.append(r)

    df = pd.DataFrame(res_records, columns=columns)

    return df


def split_text_by_whitespaces(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = re.split(r"\s", text)
    return tokens


def words_n_gramm(text, n_gramm=3):
    tx = text.replace('\n', ' ')
    tx = re.sub(r' +', ' ', tx)
    w = tx.split(' ')

    ng = []

    for i, word in enumerate(w):
        ng.append(' '.join(w[i:i + n_gramm]))

    return list(set(ng))
