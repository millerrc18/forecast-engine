"""DemandSignal and StaffingAllocation models."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forecast_engine.models.base import Base, UUIDMixin, TimestampMixin


DEMAND_STATUSES = ("draft", "submitted", "acknowledged")
MILESTONE_TYPES = ("FAI", "NRE", "ECO", "PROD", "QUAL")


class DemandSignal(UUIDMixin, TimestampMixin, Base):
    """Quarterly production demand signal from a program PM."""

    __tablename__ = "demand_signals"

    program_id: Mapped[str] = mapped_column(
        ForeignKey("programs.id"), nullable=False
    )
    submitted_by: Mapped[str] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    units_in_flow: Mapped[int | None] = mapped_column(Integer)
    milestones: Mapped[str | None] = mapped_column(
        Text
    )  # JSON: [{"name": str, "target_date": "YYYY-MM-DD", "type": str}]
    scope_changes: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="draft"
    )  # draft, submitted, acknowledged

    # Relationships
    program = relationship("Program", backref="demand_signals")
    submitter = relationship("User", backref="demand_signals")
    allocations = relationship(
        "StaffingAllocation", back_populates="demand_signal"
    )

    def __repr__(self) -> str:
        return (
            f"<DemandSignal {self.program_id} "
            f"{self.period_start}–{self.period_end} [{self.status}]>"
        )


STAFFING_STATUSES = ("draft", "submitted", "accepted")
COST_POOLS = ("BAMTL", "ENGTL")


class StaffingAllocation(UUIDMixin, Base):
    """Functional manager response to a program demand signal."""

    __tablename__ = "staffing_allocations"

    demand_signal_id: Mapped[str] = mapped_column(
        ForeignKey("demand_signals.id"), nullable=False
    )
    program_id: Mapped[str] = mapped_column(
        ForeignKey("programs.id"), nullable=False
    )
    submitted_by: Mapped[str] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    cost_pool: Mapped[str] = mapped_column(String(10), nullable=False)  # BAMTL | ENGTL
    fte_count: Mapped[float] = mapped_column(Numeric(4, 1), nullable=False)
    avg_allocation_pct: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    blended_rate: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    planned_hrs_per_month: Mapped[float | None] = mapped_column(
        Numeric(8, 1), nullable=True
    )  # computed: fte_count * (avg_allocation_pct/100) * 168
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="draft"
    )  # draft, submitted, accepted
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    demand_signal = relationship("DemandSignal", back_populates="allocations")
    program = relationship("Program")
    submitter = relationship("User")

    def __repr__(self) -> str:
        return (
            f"<StaffingAllocation {self.cost_pool} "
            f"{self.fte_count}FTE@{self.avg_allocation_pct}% [{self.status}]>"
        )
