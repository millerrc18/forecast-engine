"""Auth routes — login, logout, dev-mode role picker."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from forecast_engine.templating import templates
from forecast_engine.auth.ldap import ldap_authenticate, resolve_role
from forecast_engine.auth.middleware import (
    clear_session_cookie,
    get_current_user,
    set_session_cookie,
)
from forecast_engine.config import settings
from forecast_engine.models.base import async_session
from forecast_engine.models.program import User
from sqlalchemy import select

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse("/dashboard", status_code=302)

    if settings.auth_dev_mode:
        # Dev mode: show role picker
        async with async_session() as session:
            result = await session.execute(select(User).where(User.is_active == True))
            users = result.scalars().all()

        return templates.TemplateResponse(request, "login.html", {
            "dev_mode": True,
            "users": users,
        })

    return templates.TemplateResponse(request, "login.html", {
        "dev_mode": False,
    })


@router.post("/login")
async def login_submit(request: Request):
    form = await request.form()

    if settings.auth_dev_mode:
        # Dev mode: pick user from dropdown
        user_id = form.get("user_id")
        async with async_session() as session:
            user = await session.get(User, user_id)

        if not user:
            return RedirectResponse("/auth/login?error=1", status_code=302)

        user_data = {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        }
    else:
        # LDAP mode
        username = form.get("username", "")
        password = form.get("password", "")
        ldap_info = await ldap_authenticate(username, password)

        if not ldap_info:
            return RedirectResponse("/auth/login?error=1", status_code=302)

        role = resolve_role(ldap_info["groups"])

        # Upsert user in DB
        async with async_session() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                user = User(
                    username=username,
                    display_name=ldap_info["display_name"],
                    email=ldap_info["email"],
                    role=role,
                )
                session.add(user)
                await session.commit()

        user_data = {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "role": role,
        }

    response = RedirectResponse("/dashboard", status_code=302)
    set_session_cookie(response, user_data)
    return response


@router.get("/logout")
async def logout(request: Request):
    response = RedirectResponse("/auth/login", status_code=302)
    clear_session_cookie(response)
    return response


@router.get("/me")
async def me(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"authenticated": False}, status_code=401)
    return JSONResponse({"authenticated": True, **user})
