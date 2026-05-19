# nina.autopilot

Autonomous astrophotography orchestrator for N.I.N.A. (Nighttime Imaging 'N' Astronomy).

Status: **Phase 4.1 — Operator + Scout (rule-based) shipped**. Phase 4 added
web dashboard, nightly LLM-spend circuit breaker, full Doctor action coverage
(RETRY / REPLAN / PARK_AND_WAIT / ABORT), and dashboard-routed E-STOP. Phase
4.1 added the per-sub Operator and the world-state Scout — both rule-based
(zero LLM tokens in steady state) and standalone for now; Conductor wiring
comes in a follow-up turn.

The full architecture spec lives at
`%USERPROFILE%\.claude\plans\you-are-an-expert-linked-quiche.md`.

## What works today

**Phase 2 — Safety supervisor + Conductor skeleton:**
- Loads + starts an Advanced Sequencer sequence via NINA Advanced API
- Polls every 5s (configurable):
  - **Safety Monitor** (`IsSafe`)
  - **Weather** (cloud, wind, rain, humidity, dew margin)
  - **Camera cooler** (delta from set-point)
- On UNSAFE → stops sequence → closes dome → parks mount → stops cooling → Discord PANIC alert
- On sequence completion → same close-down chain + Discord INFO
- Manual `request_stop()` → close-down chain + Discord ALERT
- Persists session phase + event log in SQLite (`session.sqlite` by default)

**Phase 3 — Planner + Doctor (NEW):**
- **Planner** (algorithmic, no LLM in v1) reads `%LOCALAPPDATA%\NINA\SchedulerPlugin\schedulerdb.sqlite`,
  walks Target Scheduler projects by priority desc, returns the first actionable target.
  If nothing's actionable, the Conductor exits cleanly without ever touching the rig
  (`end_reason="no_work"`).
- **Doctor** (LLM-driven, default `claude-sonnet-4-6`) is invoked when the sequencer reports
  `State="Error"`. It receives a structured `FaultContext` and returns one of
  `{retry, replan, park_and_wait, abort}` as JSON. System prompt is sent with `cache_control`
  so subsequent diagnoses in the same session cost ~10% of normal input tokens.
- Conductor wires both in: Planner runs at session start (replaces hand-written sequence files
  when configured); Doctor runs on fault with a `max_retries` cap (default 2) and a fail-safe
  to ABORT on malformed/unknown LLM responses.
- Cost tracking + nightly budget hook: every LLM call accumulates in `LLMClient.usage_total`
  and `cost_estimate_usd()`; the budget circuit-breaker plugs in here in Phase 4.

**Phase 4 — Dashboard + budget + full Doctor coverage (NEW):**
- **Web dashboard** (FastAPI + HTMX, bound to `127.0.0.1`, default port 8765).
  Single page polling `/api/status` and `/api/events` every 5s; live phase, target,
  recent events, LLM budget bar, and a big red E-STOP button (routed to
  `conductor.request_stop()`). Designed for Tailscale/WireGuard remote access —
  do not publish to the open internet.
- **Nightly LLM budget circuit breaker** (`NIGHTLY_BUDGET_USD`):
  - `NORMAL` until 80% spent
  - `DEMOTED` at 80% (caller should downgrade model — surfaced on dashboard)
  - `HALTED` at 100% — subsequent `LLMClient.complete()` raises `BudgetExceeded`,
    which the Conductor catches and treats as Doctor-driven ABORT
- **REPLAN handler**: Doctor returns `REPLAN` → Conductor re-runs Planner →
  stops current sequence → loads new sequence → resets retry counter → resumes.
  Without a Planner configured, REPLAN cleanly aborts.
- **PARK_AND_WAIT handler**: Doctor returns `PARK_AND_WAIT` with optional
  `retry_after_s` (default 300s) → stops sequence → parks mount → sleeps →
  re-checks safety → unparks + resumes if conditions cleared, otherwise ends
  with `park_and_wait_unsafe` reason.
- **E-STOP from the dashboard** is a `POST /api/estop` that calls
  `conductor.request_stop()`; the next imaging tick picks it up and runs the
  normal close-down chain.

**Phase 4.1 — Operator + Scout (NEW):**
- **Operator** ([`operator.py`](src/nina_autopilot/operator.py)) — per-sub
  image-quality decider. Inputs `SubFrameStats` (HFR, star count, guide RMS,
  mean ADU); returns `OperatorDecision` in `{ACCEPT, RESHOOT, REQUEST_AF,
  REQUEST_DITHER}`. Maintains a rolling HFR baseline (default 10 subs) to
  catch focus drift without per-night calibration. Rule priority:
  hard reshoot rules (HFR over max / low star count / catastrophic guide RMS)
  → focus-drift AF request → mild-guide dither suggestion → ACCEPT. Standalone
  for now; Conductor wiring waits on per-sub event subscription.
- **Scout** ([`scout.py`](src/nina_autopilot/scout.py)) — world-state delta
  detector. Takes successive `SafetyReading` snapshots, emits compact
  human-readable `ScoutSummary` (text + severity OK/WARN/ALERT + per-field
  prev→cur pairs). Per-field noise floors prevent jiggle from spamming the
  log. Severity-promotion rules: safety-monitor flips false → ALERT;
  rain appears → ALERT; power-loss → ALERT; signal disappears → WARN;
  numeric crossing of a warn threshold → WARN.

**Still deliberately NOT here yet:**
- LLM second-opinion for Operator / Scout — wired only when token budget allows.
- Operator/Scout integration into the Conductor imaging loop — needs per-sub
  IMAGE-SAVE event polling against NINA's event-websocket (Phase 1 has the
  infrastructure in `nina_mcp_server`; wiring to `nina.autopilot` is next).
- LLM-smart Planner (altitude/moon/weather scoring) — algorithmic Planner is
  sufficient for TS-driven setups.
- Sequencer JSON synthesis — relies on user's pre-saved TS-driven NINA sequence.

## Setup (dev)

```bash
cd c:\Users\miche\Desktop\github\nina.autopilot
py -m pip install -e .[dev]
cp .env.example .env
# edit .env: set DISCORD_WEBHOOK_URL at minimum
```

Required env:
- `NINA_HOST` / `NINA_PORT` — defaults to `localhost:1888`
- `DISCORD_WEBHOOK_URL` — required for lights-out alerting; orchestrator runs
  without it but `nina_alert_human` becomes a no-op (logged only)

Optional safety threshold overrides (see `.env.example`):
- `SAFETY_WIND_MAX` (km/h, default 30)
- `SAFETY_CLOUD_MAX` (%, default 80)
- `SAFETY_DEW_MARGIN_MIN` (°C, default 2.0)
- `SAFETY_COOLER_DELTA_WARN` / `SAFETY_COOLER_DELTA_UNSAFE` (°C)
- `SAFETY_TICK_S` (poll interval, default 5)

## Run

```bash
# Phase 3+ — Planner picks target from Target Scheduler, Doctor diagnoses faults,
# dashboard on http://127.0.0.1:8765/
py -m nina_autopilot

# Phase 2 — load a named sequence directly (no Planner)
py -m nina_autopilot --sequence "my_sequence_name"

# Headless (no dashboard)
py -m nina_autopilot --no-dashboard

# Skip the LLM Doctor — Conductor aborts on first fault
py -m nina_autopilot --no-doctor

# Override Doctor model (default: claude-sonnet-4-6)
py -m nina_autopilot --doctor-model claude-opus-4-7
```

The process exits 0 on a clean completion, 1 on FAULT.

The dashboard binds to 127.0.0.1 by default. To reach it from a phone/laptop
while the rig is unattended, run [Tailscale](https://tailscale.com) (or
WireGuard) on the NINA PC and hit `http://<tailscale-ip>:8765/` — never
publish the dashboard to the open internet, there's no auth layer.

## Architecture (Phase 2)

```
        ┌─────────────────────────────────────────────────┐
        │                  Conductor                       │
        │   ┌───────────────────────────────────────┐     │
        │   │  Phase state machine                  │     │
        │   │  BOOT → STARTING → IMAGING → ABORTING │     │
        │   │              ↑          ↓     ↓       │     │
        │   │              └─safety   └→ CLOSING → DONE
        │   └───────────────────────────────────────┘     │
        │   ┌───────────────────────┐                     │
        │   │  Safety supervisor    │  ← pure rules, no LLM
        │   │  (safety.evaluate)    │                     │
        │   └───────────────────────┘                     │
        └────────────────────┬────────────────────────────┘
                             │ NinaClient Protocol
                             ▼
        ┌─────────────────────────────────────────┐
        │  HttpNinaClient                          │
        │  - GET equipment/safetymonitor/info      │
        │  - GET equipment/weather/info            │
        │  - GET equipment/camera/info             │
        │  - GET sequence/{load,start,stop,state}  │
        │  - GET equipment/{dome,mount,camera}/... │
        │  - POST Discord webhook (alerts)         │
        └─────────────────────────────────────────┘
                             │ HTTP
                             ▼
                    NINA + Advanced API plugin
```

The Conductor depends on the `NinaClient` Protocol; tests inject a
`FakeNinaClient` that records every call and lets you script reactions.

## Layout

```
src/nina_autopilot/
├── conductor.py     # state machine + close-down + Planner/Doctor/REPLAN/PARK wiring
├── safety.py        # pure rule engine (SafetyDecision)
├── planner.py       # Phase 3: TS DB → PlannerDecision (algorithmic)
├── doctor.py        # Phase 3: LLM fault diagnosis (single-turn JSON out)
├── operator.py      # Phase 4.1: per-sub quality decision (rule-based)
├── scout.py         # Phase 4.1: world-state delta summaries (rule-based)
├── llm.py           # Phase 3+4: Anthropic wrapper, prompt caching, budget circuit breaker
├── dashboard.py     # Phase 4: FastAPI app — /api/status, /api/events, /api/estop, /
├── nina_client.py   # NinaClient Protocol + FakeNinaClient
├── http_client.py   # production NINA HTTP impl (incl. unpark_mount)
├── state.py         # SQLite session store
├── config.py        # env-driven Config (incl. dashboard + budget settings)
└── __main__.py      # CLI entrypoint — launches Conductor + dashboard concurrently
tests/
├── test_safety.py                  # 24 tests, pure rules
├── test_conductor.py               # 28 tests, state machine + Planner + Doctor +
│                                   #   REPLAN + PARK_AND_WAIT handlers
├── test_state.py                   # 9 tests, session store
├── test_planner.py                 # 7 tests, TS DB Planner
├── test_doctor.py                  # 15 tests, LLM Doctor (fake Anthropic)
├── test_operator.py                # 17 tests, per-sub Operator
├── test_scout.py                   # 13 tests, world-state Scout
├── test_llm.py                     # 8 tests, LLM wrapper + prompt caching
├── test_budget.py                  # 7 tests, nightly-budget circuit breaker
├── test_dashboard.py               # 9 tests, FastAPI endpoints via httpx ASGI client
├── test_phase2_exit_criterion.py   # 1 e2e — Phase 2 gate (injected unsafe)
├── test_phase3_exit_criterion.py   # 3 e2e — Phase 3 gate (Planner + Doctor unattended)
└── test_phase4_exit_criterion.py   # 3 e2e — Phase 4 gate (REPLAN + budget + dashboard)
```

## Tests

```bash
py -m pytest tests/ -v
```

**146 tests, all green** (Phase 1 in `nina_mcp_server` + Phase 2–4.1 in this repo).

Headline gates:
- `test_phase2_exit_criterion.py` — injects UNSAFE rain mid-IMAGING; close-down + PANIC alert.
- `test_phase3_exit_criterion.py::test_phase3_full_night_with_synthetic_fault_zero_human_intervention`
  — real Planner + real Doctor; injected plate-solve fault → RETRY → clean finish, single INFO alert.
- `test_phase4_exit_criterion.py::test_phase4_full_replan_with_budget_and_dashboard`
  — same plus Doctor REPLAN → switches target → dashboard `/api/status` shows DONE + budget;
  plus a budget-halt path (BudgetExceeded → Doctor-driven abort) and an E-STOP via
  dashboard test that fires while the Conductor is running.

## Next

**Phase 4.2 (carry-over):**
- Wire **Operator** into the imaging loop — poll NINA's
  `nina_get_capture_statistics` or subscribe to IMAGE-SAVE events (Phase 1
  infrastructure already exists in `nina_mcp_server`) and feed each sub's
  stats to `Operator.evaluate()`.
- Wire **Scout** into the Conductor's safety/weather tick — emit summaries
  to the session log and post WARN/ALERT summaries to Discord.
- Optional LLM second-opinion for both: local Ollama default, Haiku escalation
  on ambiguous cases (gated by budget circuit breaker).

**Phase 5 — In-NINA panel:**
- `nina.plugin.aiassistant` mode toggle: Standalone Chat ↔ Connected to Orchestrator.
- Live status panel + E-STOP inside NINA, mirroring the web dashboard.

See the plan file for the full Phase 5–6 rollout.
