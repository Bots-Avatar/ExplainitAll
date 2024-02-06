import sqlalchemy

from models import data_table

engine = sqlalchemy.create_engine('sqlite:///database.sqlite', echo=True)

data_table.create(engine)
