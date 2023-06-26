from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import random

from ..main import app
from ..core.dependencies import get_db
from ..core.database import Base
from ..core.settings import DATABASE_TEST_URI
from ..tests import fake_obj

engine = create_engine(
    DATABASE_TEST_URI,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

base_url = '/images'


def test_get_images():
    expected = [client.post(
        base_url,
        json=fake
    ) for fake in fake_obj.generate()]

    response = client.get(base_url)

    assert response.status_code == 200
    assert len(response.json()) == len(expected)


def test_get_image():
    expected = client.post(
        base_url,
        json=random.sample(fake_obj.generate(), 1)[0]
    ).json()

    response = client.get(base_url + f'/{expected["id"]}')

    assert response.status_code == 200
    assert response.json() == expected


def test_create_image():
    expected = random.sample(fake_obj.generate(), 1)[0]

    response = client.post(
        base_url,
        json=expected
    )

    assert response.status_code == 200

    response = response.json()

    assert response['id']

    del response['id']

    assert response == expected
