from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from .settings import DATABASE_URI

from . import models

engine = create_engine(
    # connect_args={"check_same_thread": False} is only ment for SQLite dbs
    DATABASE_URI, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def create_db():
    models.Base.metadata.create_all(bind=engine)
