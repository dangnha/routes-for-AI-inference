from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.auth import sign_in, sign_up
from app.schemas import Token, UserCreate, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate) -> UserRead:
    """Create the user in Supabase Auth (email + password)."""
    data = await sign_up(payload.email, payload.password)
    # GoTrue's /signup returns the user object directly; some clients wrap it.
    user = data.get("user") or data
    if not user.get("id"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed — check your email to confirm the account.",
        )
    return UserRead(id=user["id"], email=user.get("email"), role="user")


@router.post("/token", response_model=Token)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """Exchange email + password for a Supabase access token."""
    data = await sign_in(form.username, form.password)
    return Token(access_token=data["access_token"])
