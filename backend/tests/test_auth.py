from app.api import deps
from app.api.routers import auth as auth_router
from app.core.errors import AppError


def test_register_proxies_to_supabase(client, monkeypatch):
    async def fake_sign_up(email, password):
        return {"id": "user-1", "email": email}

    monkeypatch.setattr(auth_router, "sign_up", fake_sign_up)
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "a@example.com", "password": "strong-pass-123"},
    )
    assert response.status_code == 201
    assert response.json() == {
        "id": "user-1",
        "email": "a@example.com",
        "role": "user",
    }


def test_login_returns_access_token(client, monkeypatch):
    async def fake_sign_in(email, password):
        return {"access_token": "supabase-access-token"}

    monkeypatch.setattr(auth_router, "sign_in", fake_sign_in)
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "a@example.com", "password": "strong-pass-123"},
    )
    assert response.status_code == 200
    assert response.json()["access_token"] == "supabase-access-token"


def test_protected_route_requires_token(client):
    assert client.get("/api/v1/tasks").status_code == 401


def test_protected_route_rejects_invalid_token(client, monkeypatch):
    async def reject(token):
        raise AppError(401, "Invalid authentication credentials")

    monkeypatch.setattr(deps, "verify_access_token", reject)
    response = client.get(
        "/api/v1/tasks", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


def test_protected_route_with_valid_user(client, login_as):
    login_as("user-1")
    response = client.get("/api/v1/tasks")
    assert response.status_code == 200
    assert response.json() == []


def test_admin_route_requires_admin_role(client, login_as):
    login_as("user-1", role="user")
    assert client.get("/api/v1/tasks/admin/all").status_code == 403

    login_as("admin-1", role="admin")
    assert client.get("/api/v1/tasks/admin/all").status_code == 200
