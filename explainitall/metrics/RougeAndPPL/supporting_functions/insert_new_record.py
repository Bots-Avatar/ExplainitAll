import time
from .get_max_dataset_version import get_max_dataset_version


def insert_new_record(data_table, conn, model_name, dataset_name, dataset_version, metric_values):
    record = {}
    record['model_name'] = model_name
    record['timestamp'] = int(time.time())
    record['dataset_name'] = dataset_name
    record['dataset_version'] = dataset_version
    for m in metric_values:
        record[m] = metric_values[m]

    statement = data_table.insert().values(**record)
    conn.execute(statement)
    conn.commit()
