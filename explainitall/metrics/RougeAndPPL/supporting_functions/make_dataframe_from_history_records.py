from datetime import datetime

import pandas as pd


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
