from sqlalchemy import Table, Column, Float, Integer, String, MetaData, select


data_table = Table('data', 
                   MetaData(),
                   Column('id', Integer, primary_key=True, autoincrement=True),
                   Column('model_name', String),
                   Column('timestamp', Integer),
                   Column('dataset_name', String),
                   Column('dataset_version', Integer),
                   Column('PPL', Float),
                   Column('R3', Float),
                   Column('R5', Float),
                   Column('R-L', Float))
