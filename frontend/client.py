"""Thin httpx client for the FastAPI backend."""
from __future__ import annotations

import httpx


class BackendClient:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def register(self, email: str, password: str) -> dict:
        r = httpx.post(
            f"{self.base_url}/api/v1/auth/register",
            json={"email": email, "password": password},
        )
        r.raise_for_status()
        return r.json()

    def login(self, email: str, password: str) -> dict:
        r = httpx.post(
            f"{self.base_url}/api/v1/auth/token",
            data={"username": email, "password": password},
        )
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        return data

    def list_models(self) -> list[dict]:
        r = httpx.get(f"{self.base_url}/api/v1/models", headers=self._headers())
        r.raise_for_status()
        return r.json()

    def presign(self, filename: str, content_type: str) -> dict:
        r = httpx.post(
            f"{self.base_url}/api/v1/uploads/presign",
            json={"filename": filename, "content_type": content_type},
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()

    def upload(self, upload_url: str, content: bytes, content_type: str) -> None:
        r = httpx.put(upload_url, content=content, headers={"Content-Type": content_type})
        r.raise_for_status()

    def create_task(self, object_key: str, model: str, filename: str) -> dict:
        r = httpx.post(
            f"{self.base_url}/api/v1/tasks",
            json={"object_key": object_key, "model": model, "filename": filename},
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()

    def get_task(self, task_id: str) -> dict:
        r = httpx.get(f"{self.base_url}/api/v1/tasks/{task_id}", headers=self._headers())
        r.raise_for_status()
        return r.json()

    def list_tasks(self, offset: int = 0, limit: int = 50) -> list[dict]:
        r = httpx.get(
            f"{self.base_url}/api/v1/tasks",
            params={"offset": offset, "limit": limit},
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()
