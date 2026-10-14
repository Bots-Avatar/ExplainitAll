from sqlalchemy import create_engine, Table, Column, Float, Integer, String, MetaData


metadata = MetaData()


data_table = Table('data', metadata,
                   Column('id', Integer, primary_key=True, autoincrement=True),
                   Column('model_name', String),
                   Column('timestamp', Integer),
                   Column('dataset_name', String),
                   Column('dataset_version', Integer),
                   Column('PPL', Float),
                   Column('R3', Float),
                   Column('R5', Float),
                   Column('R-L', Float))


def get_engine(db_path: str = 'database.sqlite', echo: bool = False):
    """Подключение к базе истории метрик; таблица создаётся при первом вызове, а не при импорте модуля"""
    engine = create_engine(f'sqlite:///{db_path}', echo=echo)
    metadata.create_all(engine)
    return engine
