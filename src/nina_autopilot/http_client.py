"""HttpNinaClient — production NinaClient backed by NINA's Advanced API.

Maps the Conductor's small control surface to the actual HTTP endpoints. The
Advanced API wraps the real ASCOM/driver response in `{"Response": ...}`, so
this client unwraps it and converts to the SafetyReading dataclass the
supervisor expects.

Discord alerting is in-lined here (small enough — avoids a hard dep on
nina_mcp_server's alerter module).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
from urllib.parse import quote

import aiohttp

from .operator import SubFrameStats
from .safety import SafetyReading


logger = logging.getLogger(__name__)


def _wind_ms_to_kmh(v: Optional[float]) -> Optional[float]:
    return v * 3.6 if v is not None else None


def _f(value: Any) -> Optional[float]:
    """Coerce to float, returning None on missing / NaN / sentinel values."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # NINA / ASCOM often uses NaN or large negative sentinels for "no data".
    if f != f:  # NaN check
        return None
    return f


def _b(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return None


class HttpNinaClient:
    def __init__(
        self,
        base_url: str,
        discord_webhook_url: Optional[str] = None,
        discord_user_id: Optional[str] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._webhook = discord_webhook_url
        self._user_id = discord_user_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "HttpNinaClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _get(self, path: str) -> dict[str, Any]:
        assert self._session is not None, "HttpNinaClient must be used as async context manager"
        url = f"{self._base_url}/{path.lstrip('/')}"
        async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            data = await r.json()
        return data

    @staticmethod
    def _response(data: dict[str, Any]) -> dict[str, Any]:
        """Unwrap the Advanced API's {Response: ...} envelope."""
        if isinstance(data, dict) and "Response" in data and isinstance(data["Response"], dict):
            return data["Response"]
        return data

    # ---- NinaClient methods -------------------------------------------------

    async def get_safety_reading(self) -> SafetyReading:
        """Aggregate safety + weather + camera signals into one snapshot.

        Each sub-call is independent — a missing driver returns None for those
        fields rather than failing the whole reading. This keeps the rig usable
        even if e.g. you don't have a Safety Monitor wired up.
        """
        async def _safe(coro):
            try:
                return await coro
            except Exception as e:
                logger.debug("safety reading sub-call failed: %s", e)
                return None

        safety_raw = await _safe(self._get("equipment/safetymonitor/info"))
        weather_raw = await _safe(self._get("equipment/weather/info"))
        camera_raw = await _safe(self._get("equipment/camera/info"))
        mount_raw = await _safe(self._get("equipment/mount/info"))
        dome_raw = await _safe(self._get("equipment/dome/info"))

        is_safe = None
        if safety_raw is not None:
            r = self._response(safety_raw)
            is_safe = _b(r.get("IsSafe"))

        cloud_pct = wind_kmh = humidity = rain = dew_margin = None
        if weather_raw is not None:
            w = self._response(weather_raw)
            cloud_pct = _f(w.get("CloudCover"))
            wind_kmh = _wind_ms_to_kmh(_f(w.get("WindSpeed")))
            humidity = _f(w.get("Humidity"))
            rate = _f(w.get("RainRate"))
            rain = (rate is not None and rate > 0)
            ambient = _f(w.get("Temperature"))
            dew_point = _f(w.get("DewPoint"))
            if ambient is not None and dew_point is not None:
                dew_margin = ambient - dew_point

        cooler_delta = None
        if camera_raw is not None:
            c = self._response(camera_raw)
            t_actual = _f(c.get("Temperature"))
            t_set = _f(c.get("TemperatureSetPoint"))
            if t_actual is not None and t_set is not None:
                cooler_delta = t_actual - t_set

        mount_parked = None
        if mount_raw is not None:
            mount_parked = _b(self._response(mount_raw).get("AtPark"))

        dome_open = None
        if dome_raw is not None:
            d = self._response(dome_raw)
            shutter = d.get("ShutterStatus")
            if shutter is not None:
                # ASCOM ShutterStatus: 0=Open, 1=Closed, 2=Opening, 3=Closing, 4=Error
                dome_open = (shutter == 0 or str(shutter).lower() == "open")

        return SafetyReading(
            safety_is_safe=is_safe,
            cloud_cover_pct=cloud_pct,
            wind_kmh=wind_kmh,
            rain=rain,
            humidity_pct=humidity,
            dew_margin_c=dew_margin,
            cooler_delta_c=cooler_delta,
            mount_at_park=mount_parked,
            dome_shutter_open=dome_open,
            power_ok=None,  # site-specific; left for future switch integration
        )

    async def get_sequence_state(self) -> dict[str, Any]:
        raw = await self._get("sequence/state")
        return self._response(raw) if isinstance(raw, dict) else {"State": "Unknown"}

    async def load_sequence(self, name: str) -> dict[str, Any]:
        return await self._get(f"sequence/load?sequenceName={quote(name)}")

    async def start_sequence(self) -> dict[str, Any]:
        return await self._get("sequence/start")

    async def stop_sequence(self) -> dict[str, Any]:
        return await self._get("sequence/stop")

    async def close_dome_shutter(self) -> dict[str, Any]:
        return await self._get("equipment/dome/close-shutter")

    async def park_mount(self) -> dict[str, Any]:
        return await self._get("equipment/mount/park")

    async def unpark_mount(self) -> dict[str, Any]:
        return await self._get("equipment/mount/unpark")

    async def stop_cooling(self) -> dict[str, Any]:
        return await self._get("equipment/camera/cool?cancel=true")

    async def get_latest_sub_stats(self) -> Optional[dict[str, Any]]:
        """Fetch the most recent captured sub's stats from NINA's image history.

        Returns {"index": int, "stats": SubFrameStats} or None if no captures yet.
        The orchestrator tracks `last_sub_index` and only feeds Operator on new subs.
        """
        try:
            raw = await self._get("image-history?count=1")
        except Exception as e:
            logger.debug("image-history fetch failed: %s", e)
            return None
        images = self._response(raw)
        # NINA returns the history as a list under various keys depending on version;
        # accept both common shapes defensively.
        if isinstance(images, list):
            entries = images
        elif isinstance(images, dict):
            entries = images.get("Images") or images.get("Response") or []
        else:
            entries = []
        if not entries:
            return None
        img = entries[-1]
        index = img.get("Id") or img.get("Index") or img.get("ImageId") or 0
        stats = SubFrameStats(
            hfr=_f(img.get("HFR")),
            star_count=img.get("StarCount") or img.get("Stars"),
            mean=_f(img.get("Mean")),
            median=_f(img.get("Median")),
            filter_name=img.get("Filter") or img.get("FilterName"),
            exposure_s=_f(img.get("ExposureTime") or img.get("Duration")),
        )
        return {"index": int(index), "stats": stats}

    async def alert(
        self,
        severity: str,
        message: str,
        image_path: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self._webhook:
            logger.info("alert (no webhook configured): [%s] %s", severity, message)
            return {"Success": False, "Reason": "No webhook configured"}

        sev = severity.lower()
        prefix = {"info": "**[INFO]**", "alert": "**[ALERT]**", "panic": "🚨 **[PANIC]**"}.get(sev, "**[?]**")
        mention = ""
        if sev == "alert" and self._user_id:
            mention = f"<@{self._user_id}> "
        elif sev == "panic":
            mention = "@everyone "
        content = f"{mention}{prefix} {message}"
        if len(content) > 2000:
            content = content[:1999] + "…"
        payload = {"username": "NINA Autopilot", "content": content}

        assert self._session is not None
        if image_path:
            data = aiohttp.FormData()
            data.add_field("payload_json", json.dumps(payload), content_type="application/json")
            with open(image_path, "rb") as fh:
                data.add_field("files[0]", fh.read(), filename="image.jpg",
                               content_type="application/octet-stream")
            async with self._session.post(self._webhook, data=data) as r:
                return {"Success": r.status < 400, "HttpStatus": r.status}
        async with self._session.post(self._webhook, json=payload) as r:
            return {"Success": r.status < 400, "HttpStatus": r.status}
