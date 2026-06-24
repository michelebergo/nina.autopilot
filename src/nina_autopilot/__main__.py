"""Entry point — runs one autopilot session end-to-end.

Modes:
  python -m nina_autopilot                  # Phase 3 — Planner picks target from TS DB
  python -m nina_autopilot --sequence X.json # Phase 2 — load named sequence
  python -m nina_autopilot --build-sequence  # Multi-target: plan_all → write JSON → load
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
from .planner import plan_all, plan_next
from .sequence_builder import SequenceBuildConfig, write_sequence
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
    # In --build-sequence mode, plan_all() feeds the sequence builder.
    planner_fn = None
    built_sequence_name: str | None = None
    if args.build_sequence:
        decision = plan_all(
            sequence_name=args.ts_sequence_name,
            ts_db_path=cfg.ts_db_path,
            profile_id=cfg.ts_profile_id,
        )
        if decision.action.value == "no_work":
            logger.info("Planner found no actionable targets — nothing to build.")
            return 0
        build_cfg = SequenceBuildConfig(sequences_dir=args.sequences_dir)
        out_path = write_sequence(decision, args.ts_sequence_name, build_cfg)
        built_sequence_name = out_path.stem  # NINA loads by name without .json
        logger.info(
            "Built multi-target sequence '%s' with %d target(s): %s",
            built_sequence_name,
            len(decision.targets),
            decision.summary,
        )
        # Use the built sequence as the conductor's sequence_file
        planner_fn = None
    elif not args.sequence:
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
                sequence_file=built_sequence_name or args.sequence,
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
            if dashboard_server is not None and args.keep_alive_seconds > 0:
                logger.info(
                    "session done — keeping dashboard alive at http://%s:%d for %ds (Ctrl-C to exit)",
                    cfg.dashboard_host, cfg.dashboard_port, args.keep_alive_seconds,
                )
                try:
                    await asyncio.sleep(args.keep_alive_seconds)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    logger.info("keep-alive interrupted, shutting down")
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
        "--build-sequence", action="store_true",
        help=(
            "Multi-target mode: run plan_all() to collect every actionable "
            "Target Scheduler target, build a NINA sequence JSON (Start → "
            "Targets → End containers), write it to the sequences dir, and "
            "load it into NINA."
        ),
    )
    parser.add_argument(
        "--sequences-dir", default=None,
        help="Override the NINA sequences directory (default: %LOCALAPPDATA%\\NINA\\Sequences).",
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
    parser.add_argument(
        "--keep-alive-seconds", type=int, default=0,
        help=(
            "After the session ends, keep the dashboard alive for N seconds "
            "(useful for inspecting the final state in a browser or letting "
            "the in-NINA panel connect for UI testing without a live session). "
            "Default 0 = exit immediately."
        ),
    )
    args = parser.parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
