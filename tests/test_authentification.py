from fastapi.testclient import TestClient
from jose import jwt
from services.authentication.authenticate_api import app, SECRET_KEY, ALGORITHM

client = TestClient(app)

def test_login_success():
    response = client.post("/login", json={"username": "alice", "password": "secret"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_failure():
    response = client.post("/login", json={"username": "doc", "password": "wrong"})
    assert response.status_code == 401


def test_invalid_token():
    response = client.get("/protected", headers={"Authorization": "Bearer invalid"})
    assert response.status_code == 401

