from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# "jdbc:sqlite:/Users/koraypoyraz/PycharmProjects/dataCompressorProject/data/images_db"
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/images_db"

engine = create_engine(
    # connect_args={"check_same_thread": False} is only ment for SQLite dbs
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()