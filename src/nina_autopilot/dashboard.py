"""Web dashboard — FastAPI app that surfaces session state to the human.

Bound to 127.0.0.1 in production; remote access via Tailscale/WireGuard
(never published to the open internet — per the locked plan decisions).

Endpoints:
  GET  /             → HTMX page (single small HTML doc)
  GET  /api/status   → JSON snapshot: phase, current session, budget
  GET  /api/events   → recent events from the current session
  POST /api/estop    → triggers conductor.request_stop()
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .conductor import Phase
from .llm import LLMClient
from .state import SessionStore


@runtime_checkable
class DashboardConductor(Protocol):
    """Read-only-plus-estop interface the dashboard depends on.

    The real Conductor instance satisfies this naturally; tests use a stub.
    """
    @property
    def phase(self) -> Phase: ...
    async def request_stop(self) -> None: ...


_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NINA Autopilot</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; background:#0e1116; color:#e6edf3;
         margin: 0; padding: 1rem; line-height: 1.4; }
  .card { background:#161b22; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;
          border: 1px solid #30363d; }
  h1 { margin: 0 0 1rem 0; font-size: 1.4rem; }
  h2 { margin: 0 0 .5rem 0; font-size: 1rem; color:#7d8590; text-transform: uppercase;
       letter-spacing: 0.05em; }
  .phase { font-size: 2rem; font-weight: 600; color: #58a6ff; }
  .small { font-size: .85rem; color:#7d8590; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th, td { text-align: left; padding: .25rem .5rem; border-bottom: 1px solid #21262d; }
  th { color:#7d8590; font-weight: normal; }
  .estop { width: 100%; padding: 1rem; font-size: 1.2rem; font-weight: 700;
           background: #da3633; color:white; border: none; border-radius: 8px;
           cursor:pointer; }
  .estop:hover { background: #b62324; }
  .budget-bar { height: 8px; background:#21262d; border-radius: 4px; overflow: hidden;
                margin-top: .5rem; }
  .budget-fill { height: 100%; background: #3fb950; transition: width 0.5s; }
  .budget-fill.demoted { background:#d29922; }
  .budget-fill.halted  { background:#f85149; }
  .badge { display:inline-block; padding: .15rem .5rem; border-radius: 4px;
           font-size: .75rem; font-weight: 600; }
  .badge.normal  { background:#1f6f43; color:#7ee787; }
  .badge.demoted { background:#9e6a03; color:#f8e3a1; }
  .badge.halted  { background:#a40e26; color:#ffabab; }
</style>
</head>
<body>
<h1>🔭 NINA Autopilot</h1>

<div class="card" hx-get="/api/status" hx-trigger="load, every 5s" hx-target="this" hx-swap="innerHTML">
  Loading status…
</div>

<div class="card">
  <h2>Recent events</h2>
  <div hx-get="/api/events?limit=15" hx-trigger="load, every 5s" hx-target="this" hx-swap="innerHTML">
    Loading…
  </div>
</div>

<div class="card">
  <h2>Emergency</h2>
  <button class="estop"
          hx-post="/api/estop"
          hx-confirm="E-STOP will stop the sequence, close the dome, park the mount, warm the camera, and end the session. Proceed?"
          hx-swap="none">
    🛑 E-STOP
  </button>
</div>

<script>
// Render status JSON into something readable.
document.body.addEventListener("htmx:afterRequest", (ev) => {
  const url = ev.detail.requestConfig.path || "";
  const target = ev.detail.target;
  if (!target || ev.detail.xhr.status !== 200) return;
  try {
    if (url.startsWith("/api/status")) {
      const d = JSON.parse(ev.detail.xhr.response);
      const phaseClass = (d.phase || "?").toLowerCase();
      let budgetHtml = "";
      if (d.budget) {
        const pct = d.budget.budget_usd
          ? Math.min(100, (d.budget.spent_usd / d.budget.budget_usd) * 100) : 0;
        budgetHtml = `
          <h2 style="margin-top:1rem">LLM budget</h2>
          <div>$${d.budget.spent_usd.toFixed(2)} / $${(d.budget.budget_usd || 0).toFixed(2)}
          <span class="badge ${d.budget.state}">${d.budget.state}</span></div>
          <div class="budget-bar"><div class="budget-fill ${d.budget.state}" style="width:${pct}%"></div></div>
        `;
      }
      const sess = d.session
        ? `<div class="small">session ${d.session.id} · ${d.session.sequence_file || ""} · started ${d.session.started_at}</div>`
        : `<div class="small">no active session</div>`;
      target.innerHTML = `
        <h2>Current phase</h2>
        <div class="phase">${d.phase}</div>
        ${sess}
        ${budgetHtml}
      `;
    } else if (url.startsWith("/api/events")) {
      const events = JSON.parse(ev.detail.xhr.response);
      if (!events.length) { target.innerHTML = "<div class='small'>no events yet</div>"; return; }
      const rows = events.map(e => `<tr>
        <td class="small">${e.timestamp.replace("T", " ").slice(0, 19)}</td>
        <td>${e.kind}</td>
        <td class="small">${JSON.stringify(e.payload).slice(0, 160)}</td>
      </tr>`).join("");
      target.innerHTML = `<table>
        <tr><th>time</th><th>kind</th><th>payload</th></tr>${rows}</table>`;
    }
  } catch (e) { /* swallow render errors */ }
});
</script>
</body>
</html>
"""


def create_app(
    *,
    conductor: DashboardConductor,
    store: SessionStore,
    llm: Optional[LLMClient] = None,
) -> FastAPI:
    app = FastAPI(title="NINA Autopilot Dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return _HTML_PAGE

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        session = store.current_session()
        out: dict[str, Any] = {
            "phase": conductor.phase.value,
            "session": session,
        }
        if llm is not None:
            out["budget"] = llm.budget_snapshot()
        return out

    @app.get("/api/events")
    async def events(limit: int = 50) -> list[dict[str, Any]]:
        session = store.current_session()
        if session is None:
            return []
        rows = store.list_events(session["id"])
        # Most recent first, capped at `limit`
        return list(reversed(rows[-limit:]))

    @app.post("/api/estop")
    async def estop() -> dict[str, Any]:
        await conductor.request_stop()
        return {"ok": True, "message": "Stop requested — close-down will run on next tick."}

    return app
