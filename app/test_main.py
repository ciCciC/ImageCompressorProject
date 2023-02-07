from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_say_hello():
    name = 'Py'
    response = client.get(f"/hello/{name}")
    assert response.status_code == 200
    assert response.json() == {"message": f"Hello {name}"}

# def test_get_images():
#     assert False
#
#
# def test_get_image():
#     assert False
#
#
# def test_create_image():
#     assert False
