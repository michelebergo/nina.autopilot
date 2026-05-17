"""Runtime configuration — loaded from environment, with sane defaults.

Single source of truth for env-derived settings; everything else (Conductor,
HttpNinaClient) takes a Config dataclass and reads from it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .safety import SafetyThresholds


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass
class Config:
    nina_host: str = "localhost"
    nina_port: int = 1888
    discord_webhook_url: Optional[str] = None
    discord_user_id: Optional[str] = None
    state_db_path: Path = field(default_factory=lambda: Path("session.sqlite"))
    safety_tick_s: float = 5.0
    thresholds: SafetyThresholds = field(default_factory=SafetyThresholds)
    # Phase 4 additions
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765
    dashboard_enabled: bool = True
    nightly_budget_usd: Optional[float] = None
    ts_db_path: Optional[str] = None  # None = default %LOCALAPPDATA% location
    ts_profile_id: Optional[str] = None

    @property
    def nina_base_url(self) -> str:
        return f"http://{self.nina_host}:{self.nina_port}/v2/api"


def _env_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    return float(raw) if raw else None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> Config:
    """Reads .env into os.environ then builds a Config from it."""
    load_dotenv()
    th = SafetyThresholds(
        wind_max_kmh=_env_float("SAFETY_WIND_MAX", 30.0),
        cloud_max_pct=_env_float("SAFETY_CLOUD_MAX", 80.0),
        dew_margin_min_c=_env_float("SAFETY_DEW_MARGIN_MIN", 2.0),
        humidity_warn_pct=_env_float("SAFETY_HUMIDITY_WARN", 95.0),
        cooler_delta_warn_c=_env_float("SAFETY_COOLER_DELTA_WARN", 2.0),
        cooler_delta_unsafe_c=_env_float("SAFETY_COOLER_DELTA_UNSAFE", 10.0),
    )
    return Config(
        nina_host=os.getenv("NINA_HOST", "localhost"),
        nina_port=int(os.getenv("NINA_PORT", "1888")),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL") or None,
        discord_user_id=os.getenv("DISCORD_USER_ID") or None,
        state_db_path=Path(os.getenv("STATE_DB_PATH", "session.sqlite")),
        safety_tick_s=_env_float("SAFETY_TICK_S", 5.0),
        thresholds=th,
        dashboard_host=os.getenv("DASHBOARD_HOST", "127.0.0.1"),
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "8765")),
        dashboard_enabled=_env_bool("DASHBOARD_ENABLED", True),
        nightly_budget_usd=_env_optional_float("NIGHTLY_BUDGET_USD"),
        ts_db_path=os.getenv("TS_DB_PATH") or None,
        ts_profile_id=os.getenv("TS_PROFILE_ID") or None,
    )
