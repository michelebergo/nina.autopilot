"""Entry point — runs one autopilot session end-to-end.

Modes:
  python -m nina_autopilot                  # Phase 3 — Planner picks target from TS DB
  python -m nina_autopilot --sequence X.json # Phase 2 — load named sequence
  python -m nina_autopilot --no-dashboard   # skip web UI (headless)
  python -m nina_autopilot --no-doctor      # skip LLM Doctor (rule-only behaviour)

The dashboard runs as a background uvicorn server alongside the conductor.
When the conductor returns, the dashboard is signalled to shut down.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from functools import partial

import uvicorn

from .conductor import Conductor, ConductorConfig
from .config import load_config
from .dashboard import create_app
from .doctor import Doctor
from .http_client import HttpNinaClient
from .llm import LLMClient
from .planner import plan_next
from .state import open_store


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


async def _serve_dashboard(app, host: str, port: int) -> uvicorn.Server:
    """Start the dashboard as a background asyncio task."""
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve(), name="dashboard")
    # Wait briefly for startup so /api/status is reachable when the human checks.
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.1)
    logger.info("dashboard ready at http://%s:%d/", host, port)
    return server


async def _run(args: argparse.Namespace) -> int:
    cfg = load_config()
    store = open_store(cfg.state_db_path)

    # LLM client only created if Doctor is enabled (saves an API-key requirement
    # for users in Phase-2 mode).
    llm = None
    doctor = None
    if not args.no_doctor:
        llm = LLMClient(nightly_budget_usd=cfg.nightly_budget_usd)
        doctor = Doctor(llm=llm, model=args.doctor_model)

    # Planner — used unless --sequence overrides it.
    planner_fn = None
    if not args.sequence:
        planner_fn = partial(
            plan_next,
            sequence_name=args.ts_sequence_name,
            ts_db_path=cfg.ts_db_path,
            profile_id=cfg.ts_profile_id,
        )

    async with HttpNinaClient(
        cfg.nina_base_url,
        discord_webhook_url=cfg.discord_webhook_url,
        discord_user_id=cfg.discord_user_id,
    ) as client:
        conductor = Conductor(
            client,
            store,
            ConductorConfig(
                sequence_file=args.sequence,
                planner=planner_fn,
                doctor=doctor,
                safety_tick_s=cfg.safety_tick_s,
                thresholds=cfg.thresholds,
            ),
        )

        # Optionally launch the dashboard alongside the Conductor.
        dashboard_server = None
        if cfg.dashboard_enabled and not args.no_dashboard:
            app = create_app(conductor=conductor, store=store, llm=llm)
            dashboard_server = await _serve_dashboard(app, cfg.dashboard_host, cfg.dashboard_port)

        try:
            await conductor.run()
        finally:
            if dashboard_server is not None:
                dashboard_server.should_exit = True

        return 0 if conductor.phase.value == "DONE" else 1


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="nina-autopilot")
    parser.add_argument(
        "--sequence", default=None,
        help="Phase 2 mode: skip Planner, load a named NINA sequence directly.",
    )
    parser.add_argument(
        "--ts-sequence-name", default="autopilot_ts_sequence",
        help="When Planner is used, the NINA sequence name to load on IMAGE decision.",
    )
    parser.add_argument(
        "--no-doctor", action="store_true",
        help="Disable the LLM Doctor — Conductor will abort on first fault.",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable the web dashboard.",
    )
    parser.add_argument(
        "--doctor-model", default="claude-sonnet-4-6",
        help="Anthropic model the Doctor uses.",
    )
    args = parser.parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
