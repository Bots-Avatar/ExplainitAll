import sqlalchemy as db


def get_records_from_database(data_table, conn, specific_column_value=None):
    records = []

    statement = data_table.select()

    if specific_column_value is not None:
        for k in specific_column_value:
            col = db.sql.column(k)
            statement = statement.where(col == specific_column_value[k])

    for row in conn.execute(statement):
        records.append(row)

    return records
