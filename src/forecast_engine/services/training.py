"""GBM training service with Leave-One-Group-Out cross-validation.

Trains a ``GradientBoostingRegressor`` on the feature matrix produced by
:class:`FeatureEngineer`, evaluates via LOGO-CV (one fold per program),
persists the model artifact as ``.joblib``, and stores metrics and feature
importances in the database.

Public API
----------
TrainingService
    .train_model(session, user_id, ...) -> ModelVersion
    .activate_model(session, model_version_id)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from forecast_engine.config import settings
from forecast_engine.models.ml import FeatureImportance, ModelMetric, ModelVersion
from forecast_engine.services.features import FEATURE_NAMES, FeatureEngineer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_HYPERPARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_samples_leaf": 10,
    "subsample": 0.8,
}

TARGET_COL = "total_support_hrs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%), skipping rows where actual == 0."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, MAPE, RMSE for a set of predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
    }


def _auto_version_tag(session_result_count: int) -> str:
    """Generate a version tag like ``v3.0-2026Q2``."""
    now = datetime.now(timezone.utc)
    q = (now.month - 1) // 3 + 1
    return f"v{session_result_count + 1}.0-{now.year}Q{q}"


# ---------------------------------------------------------------------------
# Core service
# ---------------------------------------------------------------------------

class TrainingService:
    """Trains a GBM model and persists results to the database."""

    def __init__(self) -> None:
        self._feature_engineer = FeatureEngineer()

    # ------------------------------------------------------------------
    # Public: train_model
    # ------------------------------------------------------------------

    async def train_model(
        self,
        session: AsyncSession,
        user_id: str,
        version_tag: str | None = None,
        hyperparameters: dict | None = None,
    ) -> ModelVersion:
        """Build features, train a GBM, evaluate with LOGO-CV, and persist.

        Parameters
        ----------
        session:
            Async SQLAlchemy session (caller manages commit/rollback).
        user_id:
            ID of the user triggering training.
        version_tag:
            Optional human-readable tag. Auto-generated if omitted.
        hyperparameters:
            Override default GBM hyperparameters.

        Returns
        -------
        ModelVersion
            The newly created, fully-populated model version record.
        """
        # 1. Build features
        df = await self._feature_engineer.build_features(session)
        if df.empty:
            raise ValueError("No training data available — upload actuals first.")

        X = df[FEATURE_NAMES].copy()
        y = df[TARGET_COL].copy()
        groups = df["program_id"].copy()

        logger.info(
            "Training data: %d rows, %d programs, %d features",
            len(X), groups.nunique(), len(FEATURE_NAMES),
        )

        # 2. Resolve hyperparams
        params = {**DEFAULT_HYPERPARAMS, **(hyperparameters or {})}

        # 3. Train final model on all data
        model = GradientBoostingRegressor(**params, random_state=42)
        model.fit(X.values, y.values)

        # 4. Auto-generate version tag if needed
        if version_tag is None:
            count_result = await session.execute(select(ModelVersion))
            existing_count = len(count_result.scalars().all())
            version_tag = _auto_version_tag(existing_count)

        # 5. Save artifact
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = settings.model_dir / f"{version_tag}.joblib"
        artifact = {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "version_tag": version_tag,
        }
        joblib.dump(artifact, artifact_path)
        logger.info("Model artifact saved to %s", artifact_path)

        # 6. Determine training data date range
        periods_in_data = df["label"].unique()
        training_start = df["label"].min()
        training_end = df["label"].max()

        # Try to extract actual dates from the period labels if possible,
        # otherwise fall back to today.
        try:
            # period labels are typically like "2026-01" etc.
            training_data_start = date.fromisoformat(training_start + "-01")
            training_data_end = date.fromisoformat(training_end + "-01")
        except (ValueError, TypeError):
            training_data_start = date.today()
            training_data_end = date.today()

        # 7. Create ModelVersion record
        model_version = ModelVersion(
            version_tag=version_tag,
            algorithm="gradient_boosting",
            hyperparameters=json.dumps(params),
            training_data_start=training_data_start,
            training_data_end=training_data_end,
            n_observations=len(df),
            n_programs=int(groups.nunique()),
            artifact_path=str(artifact_path),
            is_active=False,
            trained_by=user_id,
        )
        session.add(model_version)
        await session.flush()  # populate model_version.id

        # 8. Run LOGO-CV
        cv_results = self._run_logo_cv(X, y, groups, params)

        # Build program_id -> program_id mapping (identity, but keeps API clean)
        program_map = {pid: pid for pid in groups.unique()}

        # 9. Store metrics and feature importances
        await self._store_metrics(session, model_version.id, cv_results, program_map)
        await self._store_feature_importances(
            session, model_version.id, model, FEATURE_NAMES, X,
        )

        await session.flush()

        logger.info(
            "Model %s trained: MAE=%.2f, MAPE=%.2f%%, RMSE=%.2f, R²=%.4f",
            version_tag,
            cv_results["overall"]["mae"],
            cv_results["overall"]["mape"],
            cv_results["overall"]["rmse"],
            cv_results["overall"].get("r2", 0),
        )

        return model_version

    # ------------------------------------------------------------------
    # Public: activate_model
    # ------------------------------------------------------------------

    async def activate_model(
        self, session: AsyncSession, model_version_id: str
    ) -> None:
        """Set *model_version_id* as the sole active model.

        Deactivates all other versions first.
        """
        # Deactivate all
        await session.execute(
            update(ModelVersion).values(is_active=False)
        )

        # Activate the selected one
        await session.execute(
            update(ModelVersion)
            .where(ModelVersion.id == model_version_id)
            .values(is_active=True)
        )

        await session.flush()
        logger.info("Model %s activated.", model_version_id)

    # ------------------------------------------------------------------
    # LOGO-CV
    # ------------------------------------------------------------------

    def _run_logo_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        model_params: dict,
    ) -> dict:
        """Leave-One-Group-Out CV where group = program_id.

        Returns
        -------
        dict with keys ``overall`` and ``per_program``.
        ``overall`` contains MAE, MAPE, RMSE, R².
        ``per_program`` maps ``{program_id: {mae, mape, rmse}}``.
        """
        unique_groups = groups.unique()
        logger.info("LOGO-CV: %d folds (one per program)", len(unique_groups))

        # Collect all predictions for overall metrics
        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []
        per_program: dict[str, dict[str, float]] = {}

        for held_out in unique_groups:
            mask = groups == held_out
            X_train, y_train = X.loc[~mask], y.loc[~mask]
            X_test, y_test = X.loc[mask], y.loc[mask]

            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(
                    "LOGO-CV: skipping group %s (train=%d, test=%d)",
                    held_out, len(X_train), len(X_test),
                )
                continue

            fold_model = GradientBoostingRegressor(
                **model_params, random_state=42
            )
            fold_model.fit(X_train.values, y_train.values)
            y_pred = fold_model.predict(X_test.values)

            # Per-program metrics
            y_true_np = y_test.values
            per_program[str(held_out)] = _compute_metrics(y_true_np, y_pred)

            all_y_true.append(y_true_np)
            all_y_pred.append(y_pred)

        # Overall metrics across all held-out predictions
        if all_y_true:
            y_true_all = np.concatenate(all_y_true)
            y_pred_all = np.concatenate(all_y_pred)
            overall = _compute_metrics(y_true_all, y_pred_all)
            overall["r2"] = float(r2_score(y_true_all, y_pred_all))
        else:
            overall = {"mae": 0.0, "mape": 0.0, "rmse": 0.0, "r2": 0.0}

        return {"overall": overall, "per_program": per_program}

    # ------------------------------------------------------------------
    # DB persistence helpers
    # ------------------------------------------------------------------

    async def _store_metrics(
        self,
        session: AsyncSession,
        model_version_id: str,
        cv_results: dict,
        program_map: dict,
    ) -> None:
        """Persist LOGO-CV metrics as :class:`ModelMetric` records."""
        cv_method = "logo"

        # Overall metrics (program_id=None)
        for metric_name, metric_value in cv_results["overall"].items():
            session.add(
                ModelMetric(
                    model_version_id=model_version_id,
                    program_id=None,
                    metric_name=metric_name,
                    metric_value=Decimal(str(round(metric_value, 4))),
                    cv_method=cv_method,
                )
            )

        # Per-program metrics
        for prog_id, metrics in cv_results["per_program"].items():
            resolved_id = program_map.get(prog_id, prog_id)
            for metric_name, metric_value in metrics.items():
                session.add(
                    ModelMetric(
                        model_version_id=model_version_id,
                        program_id=resolved_id,
                        metric_name=metric_name,
                        metric_value=Decimal(str(round(metric_value, 4))),
                        cv_method=cv_method,
                    )
                )

    async def _store_feature_importances(
        self,
        session: AsyncSession,
        model_version_id: str,
        model: GradientBoostingRegressor,
        feature_names: list[str],
        X_train: pd.DataFrame,
    ) -> None:
        """Compute Gini + SHAP importances and persist as DB records."""
        # --- Gini importances ---
        gini_importances = model.feature_importances_

        # --- SHAP values ---
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train.values)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        except Exception:
            logger.warning(
                "SHAP computation failed — falling back to zeros.",
                exc_info=True,
            )
            mean_abs_shap = np.zeros(len(feature_names))

        # --- Ranks (1 = most important) ---
        gini_ranks = np.argsort(-gini_importances) + 1  # descending order
        shap_ranks = np.argsort(-mean_abs_shap) + 1

        # Build rank lookup: feature index -> rank
        gini_rank_lookup = {int(idx): int(rank) for rank, idx in enumerate(np.argsort(-gini_importances), 1)}
        shap_rank_lookup = {int(idx): int(rank) for rank, idx in enumerate(np.argsort(-mean_abs_shap), 1)}

        for i, name in enumerate(feature_names):
            session.add(
                FeatureImportance(
                    model_version_id=model_version_id,
                    feature_name=name,
                    importance=Decimal(str(round(float(gini_importances[i]), 6))),
                    mean_shap=Decimal(str(round(float(mean_abs_shap[i]), 2))),
                    rank_importance=gini_rank_lookup[i],
                    rank_shap=shap_rank_lookup[i],
                )
            )
