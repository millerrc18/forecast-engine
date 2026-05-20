"""Session middleware and auth dependencies."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from itsdangerous import BadSignature, TimestampSigner

from forecast_engine.config import settings

COOKIE_NAME = "fe_session"


def add_session_middleware(app: FastAPI):
    """Add signed-cookie session handling via middleware."""

    @app.middleware("http")
    async def session_middleware(request: Request, call_next):
        signer = TimestampSigner(settings.secret_key)
        request.state.user = None

        cookie = request.cookies.get(COOKIE_NAME)
        if cookie:
            try:
                raw = signer.unsign(cookie, max_age=settings.session_max_age)
                request.state.user = json.loads(raw)
            except (BadSignature, json.JSONDecodeError):
                pass

        response = await call_next(request)
        return response


def get_current_user(request: Request) -> dict | None:
    """Extract current user from session. Returns None if not logged in."""
    return getattr(request.state, "user", None)


def require_auth(request: Request) -> dict:
    """Dependency that requires authentication."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user


def require_role(*roles: str):
    """Dependency factory that requires specific role(s)."""
    def checker(user: Annotated[dict, Depends(require_auth)]) -> dict:
        if user.get("role") not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return user
    return checker


def set_session_cookie(response, user_data: dict):
    """Sign and set the session cookie."""
    signer = TimestampSigner(settings.secret_key)
    signed = signer.sign(json.dumps(user_data)).decode()
    response.set_cookie(
        COOKIE_NAME,
        signed,
        max_age=settings.session_max_age,
        httponly=True,
        samesite="lax",
        secure=not settings.auth_dev_mode,
        path="/",
    )


def clear_session_cookie(response):
    response.delete_cookie(COOKIE_NAME)
