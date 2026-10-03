import os

# Deterministic config for tests. The app requires Supabase settings, but tests
# never reach Supabase: auth is overridden and the DB is in-memory SQLite.
os.environ["DATABASE_URL"] = "postgresql+psycopg://test:test@localhost:5432/test"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_ANON_KEY"] = "test-anon-key"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-service-role-key"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user
from app.core.auth import AuthUser
from app.core.database import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    # In-memory SQLite is a *test fixture* only; the app itself runs on Supabase.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def login_as():
    """Bypass Supabase Auth by overriding the current-user dependency."""

    def _login(user_id: str, role: str = "user", email: str | None = None) -> AuthUser:
        user = AuthUser(id=user_id, email=email or f"{user_id}@example.com", role=role)
        app.dependency_overrides[get_current_user] = lambda: user
        return user

    yield _login
    app.dependency_overrides.pop(get_current_user, None)
