"""Program, User, ForecastPeriod, and ProgramStaff models."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forecast_engine.models.base import Base, UUIDMixin, TimestampMixin


class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str | None] = mapped_column(String(200), unique=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # pm, func_mgr, leadership, admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    programs: Mapped[list[Program]] = relationship(back_populates="pm_user")

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


class Program(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "programs"

    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    pm_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    pop_start: Mapped[date | None] = mapped_column(Date)
    pop_end: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), default="active")  # active, closing, closed
    site: Mapped[str] = mapped_column(String(10), default="59")
    contract_type: Mapped[str | None] = mapped_column(String(50))

    # Relationships
    pm_user: Mapped[User | None] = relationship(back_populates="programs")
    staff: Mapped[list[ProgramStaff]] = relationship(back_populates="program")

    def __repr__(self) -> str:
        return f"<Program {self.code} ({self.name})>"


class ForecastPeriod(UUIDMixin, Base):
    __tablename__ = "forecast_periods"

    label: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)  # e.g. "JUN-26"
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    workdays: Mapped[int] = mapped_column(Integer, nullable=False)
    quarter: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"<ForecastPeriod {self.label}>"


class ProgramStaff(UUIDMixin, TimestampMixin, Base):
    """Maps people to programs with allocation percentages."""

    __tablename__ = "program_staff"
    __table_args__ = (
        UniqueConstraint(
            "program_id", "person_name", "cost_pool", "effective_date",
            name="uq_staff_prog_person_pool_date",
        ),
    )

    program_id: Mapped[str] = mapped_column(ForeignKey("programs.id"), nullable=False)
    person_name: Mapped[str] = mapped_column(String(200), nullable=False)
    cost_pool: Mapped[str] = mapped_column(String(10), nullable=False)  # BAMTL, ENGTL, MFGTL
    allocation_pct: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    hourly_rate: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    labor_bid_code: Mapped[str | None] = mapped_column(String(20))
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date)
    updated_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))

    # Relationships
    program: Mapped[Program] = relationship(back_populates="staff")
    updated_by_user: Mapped[User | None] = relationship()

    def __repr__(self) -> str:
        return f"<ProgramStaff {self.person_name} → {self.cost_pool} {self.allocation_pct}%>"
