from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest
import random

from app.main import app, title
from app.core.dependencies import get_db
from app.core.database import Base
from app.core.settings import DATABASE_TEST_URI
from . import fake_fact

engine = create_engine(
    DATABASE_TEST_URI, connect_args={"check_same_thread": False}
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture()
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

base_url = '/images'


def test_root(test_db):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": title}


def test_say_hello(test_db):
    name = 'Py'
    response = client.get(f"/hello/{name}")
    assert response.status_code == 200
    assert response.json() == {"message": f"Hello {name}"}


def test_get_images(test_db):
    expected = [client.post(
        base_url,
        json=fake
    ) for fake in fake_fact.generate()]

    response = client.get(base_url)

    assert response.status_code == 200
    assert len(response.json()) == len(expected)


def test_get_image(test_db):
    expected = client.post(
        base_url,
        json=random.sample(fake_fact.generate(), 1)[0]
    ).json()

    response = client.get(base_url + f'/{expected["id"]}')

    assert response.status_code == 200
    assert response.json() == expected


def test_create_image(test_db):
    expected = random.sample(fake_fact.generate(), 1)[0]

    response = client.post(
        base_url,
        json=expected
    )

    assert response.status_code == 200

    response = response.json()

    assert response['id']

    del response['id']

    assert response == expected
