"""SHAP explainability engine for per-prediction feature attribution.

Wraps ``shap.TreeExplainer`` to produce JSON-serializable explanations
that power the waterfall charts and narrative text in the forecast UI.

Public API
----------
ShapEngine
    .explain(feature_vector)         -> dict   (single prediction)
    .explain_batch(feature_dicts)    -> list    (multiple predictions)
    .load(model_version)             -> ShapEngine  (class method)

shap_to_json(shap_result) -> str
json_to_shap(json_str)    -> dict
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from forecast_engine.config import settings
from forecast_engine.models.ml import ModelVersion
from forecast_engine.services.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ShapEngine:
    """Compute SHAP explanations for GBM support-hours forecasts."""

    def __init__(self, model_artifact: dict) -> None:
        """Initialize from a loaded joblib artifact dict.

        Parameters
        ----------
        model_artifact:
            Dict with keys ``model``, ``feature_names``, ``version_tag``
            as persisted by :class:`TrainingService`.
        """
        self._model = model_artifact["model"]
        self._feature_names: list[str] = model_artifact["feature_names"]
        self._version_tag: str = model_artifact.get("version_tag", "unknown")

        self._explainer = shap.TreeExplainer(self._model)
        logger.info(
            "ShapEngine initialized for model %s (%d features)",
            self._version_tag,
            len(self._feature_names),
        )

    # ------------------------------------------------------------------
    # Public: single-row explanation
    # ------------------------------------------------------------------

    def explain(self, feature_vector: dict) -> dict:
        """Explain a single prediction.

        Parameters
        ----------
        feature_vector:
            Dict mapping feature name -> numeric value.  Keys must match
            ``FEATURE_NAMES`` (order does not matter).

        Returns
        -------
        dict
            JSON-serializable explanation::

                {
                    "base_value": 210.5,
                    "features": [
                        {"name": "sup_rolling3", "value": 385.2,
                         "shap": 142.3, "direction": "up"},
                        ...
                    ]
                }

            Features are sorted by ``|shap|`` descending.
        """
        # Build ordered numpy array matching the model's expected feature order
        x = np.array(
            [feature_vector[name] for name in self._feature_names],
            dtype=np.float64,
        ).reshape(1, -1)

        shap_values = self._explainer.shap_values(x)

        # TreeExplainer may return 2-D (n_samples, n_features) — take first row
        if shap_values.ndim == 2:
            sv = shap_values[0]
        else:
            sv = shap_values

        base_value = self._extract_base_value()

        return self._format_result(base_value, sv, feature_vector)

    # ------------------------------------------------------------------
    # Public: batch explanation
    # ------------------------------------------------------------------

    def explain_batch(self, feature_dicts: list[dict]) -> list[dict]:
        """Explain multiple predictions in a single SHAP call.

        Parameters
        ----------
        feature_dicts:
            List of dicts, each mapping feature name -> value.

        Returns
        -------
        list[dict]
            One explanation dict per input row, same schema as
            :meth:`explain`.
        """
        if not feature_dicts:
            return []

        # Build DataFrame in model feature order
        df = pd.DataFrame(feature_dicts, columns=self._feature_names)
        X = df.values.astype(np.float64)

        shap_values = self._explainer.shap_values(X)

        # shap_values shape: (n_samples, n_features)
        if shap_values.ndim == 1:
            # Edge case: single row submitted via batch
            shap_values = shap_values.reshape(1, -1)

        base_value = self._extract_base_value()

        results: list[dict] = []
        for i, fdict in enumerate(feature_dicts):
            results.append(
                self._format_result(base_value, shap_values[i], fdict)
            )

        return results

    # ------------------------------------------------------------------
    # Class method: load from ModelVersion
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_version: ModelVersion) -> ShapEngine:
        """Load a ShapEngine from a persisted :class:`ModelVersion`.

        Parameters
        ----------
        model_version:
            A ``ModelVersion`` ORM instance whose ``artifact_path``
            points to a ``.joblib`` file.

        Returns
        -------
        ShapEngine
        """
        artifact_path = Path(model_version.artifact_path).resolve()
        model_dir = settings.model_dir.resolve()
        if not str(artifact_path).startswith(str(model_dir)):
            raise ValueError(
                f"Artifact path {artifact_path} is outside allowed model_dir {model_dir}"
            )
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {artifact_path}"
            )

        artifact: dict = joblib.load(artifact_path)
        logger.info(
            "Loaded model artifact from %s (version: %s)",
            artifact_path,
            artifact.get("version_tag", "unknown"),
        )
        return cls(artifact)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_base_value(self) -> float:
        """Extract the base (expected) value from the TreeExplainer.

        The explainer stores this as ``expected_value``, which may be a
        scalar or a single-element array depending on the SHAP version.
        """
        bv = self._explainer.expected_value
        if isinstance(bv, np.ndarray):
            bv = float(bv.item())
        return round(float(bv), 1)

    def _format_result(
        self,
        base_value: float,
        shap_row: np.ndarray,
        feature_vector: dict,
    ) -> dict:
        """Build the JSON-serializable explanation dict for one row."""
        features: list[dict] = []
        for j, name in enumerate(self._feature_names):
            sv = float(shap_row[j])
            features.append({
                "name": name,
                "value": round(float(feature_vector[name]), 2),
                "shap": round(sv, 1),
                "direction": "up" if sv > 0 else "down",
            })

        # Sort by absolute SHAP value descending
        features.sort(key=lambda f: abs(f["shap"]), reverse=True)

        return {
            "base_value": base_value,
            "features": features,
        }


# ---------------------------------------------------------------------------
# Module-level serialization helpers
# ---------------------------------------------------------------------------

def shap_to_json(shap_result: dict) -> str:
    """Serialize a SHAP explanation dict to a JSON string.

    Intended for storage in the ``Forecast.shap_values`` Text column.
    """
    return json.dumps(shap_result, separators=(",", ":"))


def json_to_shap(json_str: str) -> dict:
    """Deserialize a JSON string back to a SHAP explanation dict."""
    return json.loads(json_str)
