"""Model management page and API routes."""

from __future__ import annotations

import json
import logging
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from forecast_engine.auth.middleware import get_current_user, require_auth, require_role
from forecast_engine.models.base import async_session
from forecast_engine.models.ml import FeatureImportance, ModelMetric, ModelVersion
from forecast_engine.services.training import TrainingService
from forecast_engine.templating import templates

logger = logging.getLogger(__name__)

_training_service = TrainingService()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decimal_to_float(val) -> float | None:
    """Safely convert a Decimal or numeric to float."""
    if val is None:
        return None
    return float(val)


def _serialize_model(mv: ModelVersion, overall_metrics: dict | None = None) -> dict:
    """Serialize a ModelVersion to a JSON-safe dict."""
    return {
        "id": mv.id,
        "version_tag": mv.version_tag,
        "algorithm": mv.algorithm,
        "hyperparameters": json.loads(mv.hyperparameters) if mv.hyperparameters else {},
        "training_data_start": mv.training_data_start.isoformat() if mv.training_data_start else None,
        "training_data_end": mv.training_data_end.isoformat() if mv.training_data_end else None,
        "n_observations": mv.n_observations,
        "n_programs": mv.n_programs,
        "artifact_path": mv.artifact_path,
        "is_active": mv.is_active,
        "trained_at": mv.trained_at.isoformat() if mv.trained_at else None,
        "trained_by": mv.trained_by,
        "metrics": overall_metrics or {},
    }


def _get_overall_metrics(metrics: list[ModelMetric]) -> dict[str, float]:
    """Extract overall (program_id=NULL) metrics into a flat dict."""
    result = {}
    for m in metrics:
        if m.program_id is None:
            result[m.metric_name] = _decimal_to_float(m.metric_value)
    return result


# ---------------------------------------------------------------------------
# Page router -- HTML views
# ---------------------------------------------------------------------------

page_router = APIRouter(tags=["model-pages"])


@page_router.get("/models", response_class=HTMLResponse)
async def model_management_page(request: Request):
    """Model management page (admin only)."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/auth/login", status_code=302)
    if user.get("role") != "admin":
        return RedirectResponse("/dashboard", status_code=302)

    async with async_session() as session:
        result = await session.execute(
            select(ModelVersion)
            .options(selectinload(ModelVersion.metrics))
            .order_by(ModelVersion.trained_at.desc())
        )
        model_versions = result.scalars().all()

    # Build template-friendly list
    models_data = []
    active_model = None
    for mv in model_versions:
        overall = _get_overall_metrics(mv.metrics)
        entry = {
            "id": mv.id,
            "version_tag": mv.version_tag,
            "algorithm": mv.algorithm,
            "n_observations": mv.n_observations,
            "n_programs": mv.n_programs,
            "is_active": mv.is_active,
            "trained_at": mv.trained_at,
            "mape": overall.get("mape"),
            "mae": overall.get("mae"),
            "rmse": overall.get("rmse"),
            "r2": overall.get("r2"),
        }
        models_data.append(entry)
        if mv.is_active:
            active_model = entry

    return templates.TemplateResponse(request, "models.html", {
        "user": user,
        "models": models_data,
        "active_model": active_model,
    })


# ---------------------------------------------------------------------------
# API router -- JSON endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/models", tags=["models"])


class RetrainRequest(BaseModel):
    """Optional body for /retrain."""
    version_tag: str | None = None
    hyperparameters: dict | None = None


@router.get("/")
async def list_models(request: Request):
    """List all model versions with summary metrics. Admin only."""
    require_role("admin")(require_auth(request))

    async with async_session() as session:
        result = await session.execute(
            select(ModelVersion)
            .options(selectinload(ModelVersion.metrics))
            .order_by(ModelVersion.trained_at.desc())
        )
        model_versions = result.scalars().all()

    return [
        _serialize_model(mv, _get_overall_metrics(mv.metrics))
        for mv in model_versions
    ]


@router.get("/active")
async def get_active_model(request: Request):
    """Get the currently active model. Any authenticated user."""
    require_auth(request)

    async with async_session() as session:
        result = await session.execute(
            select(ModelVersion)
            .options(selectinload(ModelVersion.metrics))
            .where(ModelVersion.is_active == True)  # noqa: E712
        )
        mv = result.scalars().first()

    if not mv:
        raise HTTPException(status_code=404, detail="No active model found.")

    return _serialize_model(mv, _get_overall_metrics(mv.metrics))


@router.post("/retrain")
async def retrain_model(request: Request):
    """Trigger model retraining. Admin only.

    Accepts optional JSON body: {version_tag, hyperparameters}.
    """
    user = require_role("admin")(require_auth(request))

    # Parse optional JSON body
    body = RetrainRequest()
    try:
        raw = await request.json()
        body = RetrainRequest(**raw)
    except Exception:
        pass  # empty body is fine — defaults are used

    try:
        async with async_session() as session:
            mv = await _training_service.train_model(
                session=session,
                user_id=user["id"],
                version_tag=body.version_tag,
                hyperparameters=body.hyperparameters,
            )
            await session.commit()

            # Re-query with metrics loaded
            result = await session.execute(
                select(ModelVersion)
                .options(selectinload(ModelVersion.metrics))
                .where(ModelVersion.id == mv.id)
            )
            mv_full = result.scalars().first()

        return _serialize_model(mv_full, _get_overall_metrics(mv_full.metrics))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Model training failed")
        raise HTTPException(status_code=500, detail="Model training failed. Check server logs.")


@router.get("/{model_id}/metrics")
async def get_model_metrics(request: Request, model_id: str):
    """All metrics for a model version. Admin only."""
    require_role("admin")(require_auth(request))

    async with async_session() as session:
        result = await session.execute(
            select(ModelMetric)
            .where(ModelMetric.model_version_id == model_id)
            .order_by(ModelMetric.metric_name)
        )
        metrics = result.scalars().all()

    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found for this model.")

    return [
        {
            "id": m.id,
            "model_version_id": m.model_version_id,
            "program_id": m.program_id,
            "metric_name": m.metric_name,
            "metric_value": _decimal_to_float(m.metric_value),
            "cv_method": m.cv_method,
        }
        for m in metrics
    ]


@router.get("/{model_id}/features")
async def get_feature_importances(request: Request, model_id: str):
    """Feature importances for a model version. Admin only."""
    require_role("admin")(require_auth(request))

    async with async_session() as session:
        result = await session.execute(
            select(FeatureImportance)
            .where(FeatureImportance.model_version_id == model_id)
            .order_by(FeatureImportance.rank_importance)
        )
        features = result.scalars().all()

    if not features:
        raise HTTPException(status_code=404, detail="No feature importances found for this model.")

    return [
        {
            "id": f.id,
            "feature_name": f.feature_name,
            "importance": _decimal_to_float(f.importance),
            "mean_shap": _decimal_to_float(f.mean_shap),
            "rank_importance": f.rank_importance,
            "rank_shap": f.rank_shap,
        }
        for f in features
    ]


@router.post("/{model_id}/activate")
async def activate_model(request: Request, model_id: str):
    """Set a model version as the active production model. Admin only."""
    user = require_role("admin")(require_auth(request))

    async with async_session() as session:
        # Verify the model exists
        mv = await session.get(ModelVersion, model_id)
        if not mv:
            raise HTTPException(status_code=404, detail="Model version not found.")

        await _training_service.activate_model(session, model_id)
        await session.commit()

    return {"status": "ok", "model_id": model_id, "message": f"Model {mv.version_tag} is now active."}
