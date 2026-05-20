"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from forecast_engine.config import settings
from forecast_engine.models.base import init_db

STATIC_DIR = Path(__file__).parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    await init_db()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Register route modules
    from forecast_engine.auth.middleware import add_session_middleware
    from forecast_engine.routes import auth, pages, programs
    from forecast_engine.routes.demand import page_router as demand_page_router, router as demand_router
    from forecast_engine.routes.imports import page_router as import_page_router, router as import_router
    from forecast_engine.routes.staffing import page_router as staffing_page_router, router as staffing_router
    from forecast_engine.routes.models import page_router as model_page_router, router as model_router
    from forecast_engine.routes.forecasts import router as forecast_router

    add_session_middleware(app)
    app.include_router(auth.router)
    app.include_router(pages.router)
    app.include_router(programs.router)
    app.include_router(demand_page_router)
    app.include_router(demand_router)
    app.include_router(import_page_router)
    app.include_router(import_router)
    app.include_router(staffing_page_router)
    app.include_router(staffing_router)
    app.include_router(model_page_router)
    app.include_router(model_router)
    app.include_router(forecast_router)

    return app


app = create_app()
