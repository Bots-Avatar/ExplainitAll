from sqlalchemy import create_engine, Table, Column, Float, Integer, String, MetaData


ENGINE = create_engine('sqlite:///database.sqlite', echo=True)


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


metadata.create_all(ENGINE)
