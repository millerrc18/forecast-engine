"""ModelVersion, ModelMetric, FeatureImportance models — ML tracking layer."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forecast_engine.models.base import Base, UUIDMixin


class ModelVersion(UUIDMixin, Base):
    """Tracks trained model versions, hyperparams, and artifact paths."""

    __tablename__ = "model_versions"

    version_tag: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False
    )  # e.g., "v1.0-2026Q2"
    algorithm: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "gradient_boosting"
    hyperparameters: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # JSON string: {n_estimators, max_depth, learning_rate, ...}
    training_data_start: Mapped[date] = mapped_column(Date, nullable=False)
    training_data_end: Mapped[date] = mapped_column(Date, nullable=False)
    n_observations: Mapped[int] = mapped_column(Integer, nullable=False)
    n_programs: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_path: Mapped[str] = mapped_column(
        String(500), nullable=False
    )  # path to .joblib file
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    trained_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))

    # Relationships
    trained_by_user = relationship("User", backref="model_versions")
    metrics = relationship("ModelMetric", back_populates="model_version")
    feature_importances = relationship("FeatureImportance", back_populates="model_version")

    def __repr__(self) -> str:
        return f"<ModelVersion {self.version_tag} ({self.algorithm}) active={self.is_active}>"


class ModelMetric(UUIDMixin, Base):
    """Cross-validation metrics per model version, optionally per program."""

    __tablename__ = "model_metrics"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )
    program_id: Mapped[str | None] = mapped_column(
        ForeignKey("programs.id")
    )  # NULL = overall metric
    metric_name: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # mae, mape, r2, rmse
    metric_value: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    cv_method: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "logo" (leave-one-group-out)

    # Relationships
    model_version = relationship("ModelVersion", back_populates="metrics")
    program = relationship("Program", backref="model_metrics")

    def __repr__(self) -> str:
        return f"<ModelMetric {self.metric_name}={self.metric_value}>"


class FeatureImportance(UUIDMixin, Base):
    """Feature importance and SHAP values for a trained model."""

    __tablename__ = "feature_importances"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )
    feature_name: Mapped[str] = mapped_column(String(50), nullable=False)
    importance: Mapped[Decimal] = mapped_column(
        Numeric(8, 6), nullable=False
    )  # Gini importance
    mean_shap: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False
    )  # Mean |SHAP| in hours
    rank_importance: Mapped[int] = mapped_column(Integer, nullable=False)
    rank_shap: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    model_version = relationship("ModelVersion", back_populates="feature_importances")

    def __repr__(self) -> str:
        return f"<FeatureImportance {self.feature_name} rank={self.rank_importance}>"
