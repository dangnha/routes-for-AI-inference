"""Object storage on Supabase Storage: presigned uploads + downloads.

There is no local-disk fallback — the backend always talks to Supabase Storage,
so uploaded images live next to the data (Postgres) and auth.
"""
from __future__ import annotations

from uuid import uuid4

from app.core.config import settings

_supabase_client = None


def object_key_for(filename: str) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "img"
    return f"{uuid4().hex}.{suffix}"


def _client():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client

        _supabase_client = create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    return _supabase_client


def ensure_bucket() -> None:
    """Create the Supabase Storage bucket if it does not already exist."""
    storage = _client().storage
    try:
        names = {getattr(b, "name", None) for b in storage.list_buckets()}
    except Exception:
        names = set()
    if settings.supabase_storage_bucket not in names:
        storage.create_bucket(
            settings.supabase_storage_bucket, options={"public": False}
        )


def create_presigned_upload(object_key: str, content_type: str) -> str:
    """Return a presigned upload URL (client PUTs the image straight to Storage)."""
    result = (
        _client()
        .storage.from_(settings.supabase_storage_bucket)
        .create_signed_upload_url(object_key)
    )
    return result["signed_url"] if isinstance(result, dict) else result.signed_url


def download_object(object_key: str) -> bytes:
    return (
        _client()
        .storage.from_(settings.supabase_storage_bucket)
        .download(object_key)
    )
