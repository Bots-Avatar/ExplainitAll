from sqlalchemy import Table, Column, Float, Integer, String, MetaData, select, func


def get_max_dataset_version(dataset_name, conn, data_table):
    statement = select(func.max(data_table.c.dataset_version)).where(data_table.c.dataset_name == dataset_name)

    records = []

    for row in conn.execute(statement):
        records.append(row)

    res = records[0][0]

    if res is None:
        return -1

    return records[0][0]
