"""Application configuration — env-based settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'forecast.db'}",
        )
    )

    # Auth
    secret_key: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY", "")
    )
    session_max_age: int = 28800  # 8 hours

    # LDAP / AD
    ldap_server: str = field(default_factory=lambda: os.getenv("LDAP_SERVER", ""))
    ldap_base_dn: str = field(default_factory=lambda: os.getenv("LDAP_BASE_DN", ""))
    ldap_domain: str = field(default_factory=lambda: os.getenv("LDAP_DOMAIN", ""))
    auth_dev_mode: bool = field(
        default_factory=lambda: os.getenv("AUTH_DEV_MODE", "false").lower() == "true"
    )

    # Upload constraints
    max_upload_mb: int = 10

    # Role → AD group mapping
    role_group_map: dict[str, str] = field(default_factory=lambda: {
        "pm": os.getenv("AD_GROUP_PM", "GRP-FE-PM"),
        "func_mgr": os.getenv("AD_GROUP_FUNCMGR", "GRP-FE-FUNCMGR"),
        "leadership": os.getenv("AD_GROUP_LEADERSHIP", "GRP-FE-LEADERSHIP"),
        "admin": os.getenv("AD_GROUP_ADMIN", "GRP-FE-ADMIN"),
    })

    # ML model artifacts
    model_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("MODEL_DIR", str(BASE_DIR / "data" / "models"))
        )
    )

    # Upload staging
    upload_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
        )
    )

    # App
    app_title: str = "Forecast Engine"
    app_version: str = "0.1.0"
    site_code: str = "59"


settings = Settings()
