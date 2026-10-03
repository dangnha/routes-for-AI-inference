import pytest

from app.api.routers import tasks as tasks_router
from app.core.config import settings


@pytest.fixture()
def no_background(monkeypatch):
    async def fake_dispatch(*args, **kwargs):
        return None

    monkeypatch.setattr(tasks_router, "dispatch_inference", fake_dispatch)


def _create_task(client, login_as):
    login_as("alice-1")
    return client.post(
        "/api/v1/tasks",
        json={"object_key": "a.png", "model": "trocr", "filename": "a.png"},
    ).json()


def test_webhook_updates_task(client, login_as, no_background):
    task = _create_task(client, login_as)

    response = client.post(
        "/api/v1/webhooks/inference",
        json={
            "task_id": task["id"],
            "model": "trocr",
            "status": "succeeded",
            "text": "hello world",
            "confidence": 0.97,
        },
        headers={"X-Webhook-Secret": settings.webhook_secret},
    )
    assert response.status_code == 200

    updated = client.get(f"/api/v1/tasks/{task['id']}").json()
    assert updated["status"] == "succeeded"
    assert updated["text"] == "hello world"
    assert updated["confidence"] == pytest.approx(0.97)
    assert updated["progress"] == 100


def test_webhook_marks_failure(client, login_as, no_background):
    task = _create_task(client, login_as)

    response = client.post(
        "/api/v1/webhooks/inference",
        json={
            "task_id": task["id"],
            "model": "trocr",
            "status": "failed",
            "error": "boom",
        },
        headers={"X-Webhook-Secret": settings.webhook_secret},
    )
    assert response.status_code == 200

    updated = client.get(f"/api/v1/tasks/{task['id']}").json()
    assert updated["status"] == "failed"
    assert updated["error"] == "boom"


def test_webhook_rejects_bad_secret(client, login_as, no_background):
    task = _create_task(client, login_as)

    response = client.post(
        "/api/v1/webhooks/inference",
        json={"task_id": task["id"], "model": "trocr", "status": "succeeded"},
        headers={"X-Webhook-Secret": "not-the-secret"},
    )
    assert response.status_code == 401
