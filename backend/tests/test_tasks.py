import pytest

from app.api.routers import tasks as tasks_router


@pytest.fixture()
def no_background(monkeypatch):
    async def fake_dispatch(*args, **kwargs):
        return None

    monkeypatch.setattr(tasks_router, "dispatch_inference", fake_dispatch)


def test_create_and_get_task(client, login_as, no_background):
    user = login_as("alice-1")

    create_response = client.post(
        "/api/v1/tasks",
        json={"object_key": "abc.png", "model": "trocr", "filename": "abc.png"},
    )
    assert create_response.status_code == 202
    task = create_response.json()
    assert task["status"] == "pending"
    assert task["model"] == "trocr"
    assert task["owner_id"] == user.id

    get_response = client.get(f"/api/v1/tasks/{task['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == task["id"]


def test_owner_isolation(client, login_as, no_background):
    login_as("alice-1")
    created = client.post(
        "/api/v1/tasks",
        json={"object_key": "a.png", "model": "trocr", "filename": "a.png"},
    ).json()

    login_as("bob-1")
    response = client.get(f"/api/v1/tasks/{created['id']}")
    assert response.status_code == 403
