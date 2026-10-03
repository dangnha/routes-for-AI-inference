"""Supabase Auth integration.

The backend no longer stores users or issues its own JWTs. Authentication is
delegated to Supabase Auth (GoTrue):

- ``sign_up`` / ``sign_in`` proxy the frontend's register/login calls so only the
  backend needs the Supabase credentials (the frontend stays backend-only).
- ``verify_access_token`` validates an incoming Supabase access token by asking
  GoTrue for the user; the user's ``app_metadata.role`` drives RBAC.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
from fastapi import status

from app.core.config import settings
from app.core.errors import AppError


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str | None
    role: str


def _auth_url(path: str) -> str:
    return f"{settings.supabase_url.rstrip('/')}/auth/v1{path}"


def _base_headers() -> dict[str, str]:
    return {
        "apikey": settings.supabase_anon_key,
        "Content-Type": "application/json",
    }


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Supabase auth error"
    return str(
        payload.get("msg")
        or payload.get("error_description")
        or payload.get("error")
        or payload
    )


async def sign_up(email: str, password: str) -> dict:
    """Create a Supabase Auth user."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            _auth_url("/signup"),
            json={"email": email, "password": password},
            headers=_base_headers(),
        )
    if response.status_code >= 400:
        code = 409 if response.status_code == 422 else response.status_code
        raise AppError(code, _error_detail(response))
    return response.json()


async def sign_in(email: str, password: str) -> dict:
    """Exchange email + password for a Supabase access token."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            _auth_url("/token?grant_type=password"),
            json={"email": email, "password": password},
            headers=_base_headers(),
        )
    if response.status_code >= 400:
        payload: dict = {}
        try:
            payload = response.json()
        except ValueError:
            pass
        if payload.get("error_code") == "email_not_confirmed":
            raise AppError(
                status.HTTP_403_FORBIDDEN,
                "Email not confirmed — confirm your address, or turn off email "
                "confirmation in Supabase (Authentication → Providers → Email).",
            )
        raise AppError(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return response.json()


async def verify_access_token(token: str) -> AuthUser:
    """Validate a Supabase access token and return the authenticated user."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            _auth_url("/user"),
            headers={
                "apikey": settings.supabase_anon_key,
                "Authorization": f"Bearer {token}",
            },
        )
    if response.status_code != 200:
        raise AppError(
            status.HTTP_401_UNAUTHORIZED, "Invalid authentication credentials"
        )

    data = response.json()
    app_metadata = data.get("app_metadata") or {}
    return AuthUser(
        id=data["id"],
        email=data.get("email"),
        role=app_metadata.get("role", "user"),
    )
