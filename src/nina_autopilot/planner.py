"""Phase 3 algorithmic Planner.

At session start the Conductor calls plan_next() to decide:
  - IMAGE: there's an actionable Target Scheduler target → load the user's
    TS-driven NINA sequence and start it
  - NO_WORK: nothing to image → end the session cleanly

The Planner does NOT generate sequencer JSON in Phase 3 — the user has a
saved NINA Advanced Sequence containing a TargetSchedulerContainer, and
Target Scheduler handles target selection inside NINA at run time. The
Planner's job is to verify there's work to do AND give the human (via
Discord) a one-line summary of what the night is supposed to capture.

LLM-based re-planning lives in a future phase. This is deterministic Python.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


STATE_ACTIVE = 1


def _default_ts_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise RuntimeError("LOCALAPPDATA is not set — cannot locate Target Scheduler DB")
    return Path(base) / "NINA" / "SchedulerPlugin" / "schedulerdb.sqlite"


def _connect_ro(path: Optional[str]) -> sqlite3.Connection:
    db_path = Path(path) if path else _default_ts_path()
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class PlannerAction(str, Enum):
    IMAGE = "image"
    NO_WORK = "no_work"


@dataclass
class TargetPlan:
    """One actionable target plus its remaining exposure plans.

    Mirrors a single ``*_Container`` block in a NINA multi-target sequence:
    a target with coordinates and the per-filter exposure work still owed.
    """

    project: dict[str, Any]
    target: dict[str, Any]
    plans: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


@dataclass
class PlannerDecision:
    action: PlannerAction
    sequence_name: Optional[str] = None
    project: Optional[dict[str, Any]] = None
    target: Optional[dict[str, Any]] = None
    plans: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    # Multi-target: every actionable target in priority order. The first entry
    # mirrors the single-target fields above for backward compatibility.
    targets: list[TargetPlan] = field(default_factory=list)


# ---- DB readers (intentionally inlined, not imported from nina_mcp_server,
#                  to keep nina.autopilot self-contained) -------------------

def _active_projects(conn, profile_id: Optional[str]) -> list[dict]:
    sql = "SELECT * FROM project WHERE state = ?"
    params: list[Any] = [STATE_ACTIVE]
    if profile_id:
        sql += " AND profileId = ?"
        params.append(profile_id)
    sql += " ORDER BY priority DESC, Id ASC"
    return [dict(r) for r in conn.execute(sql, params)]


def _active_targets(conn, project_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM target WHERE projectid = ? AND active = 1 ORDER BY Id ASC",
        (project_id,),
    )
    return [dict(r) for r in rows]


def _exposure_plans(conn, target_id: int) -> list[dict]:
    sql = """
        SELECT
            ep.Id AS plan_id, ep.exposure, ep.desired, ep.acquired,
            ep.accepted, ep.enabled,
            et.name AS template_name, et.filtername AS filter_name,
            et.gain, et.offset, et.bin
        FROM exposureplan ep
        LEFT JOIN exposuretemplate et ON et.Id = ep.exposureTemplateId
        WHERE ep.targetid = ?
        ORDER BY ep.Id ASC
    """
    out = []
    for r in conn.execute(sql, (target_id,)):
        desired = r["desired"] or 0
        acquired = r["acquired"] or 0
        out.append({
            "plan_id": r["plan_id"],
            "template_name": r["template_name"],
            "filter_name": r["filter_name"],
            "exposure": r["exposure"],
            "gain": r["gain"],
            "offset": r["offset"],
            "bin": r["bin"],
            "desired": desired,
            "acquired": acquired,
            "accepted": r["accepted"] or 0,
            "remaining": max(desired - acquired, 0),
            "enabled": bool(r["enabled"]),
        })
    return out


# ---- Actionable-target collection ----------------------------------------

def _collect_actionable(conn, profile_id: Optional[str]) -> list[TargetPlan]:
    """Walk projects (priority desc) → targets → every actionable target.

    'Actionable' = target.active = 1 AND at least one enabled exposure plan
    with desired > acquired. Returns one TargetPlan per actionable target,
    in the order NINA would image them.
    """
    out: list[TargetPlan] = []
    for project in _active_projects(conn, profile_id):
        for target in _active_targets(conn, project["Id"]):
            plans = _exposure_plans(conn, target["Id"])
            actionable = [p for p in plans if p["enabled"] and p["remaining"] > 0]
            if not actionable:
                continue
            summary_parts = [
                f"target={target['name']}",
                f"project={project['name']}",
            ]
            for p in actionable:
                summary_parts.append(
                    f"{p['filter_name']}×{p['remaining']} ({p['exposure']:.0f}s)"
                )
            out.append(
                TargetPlan(
                    project={
                        "id": project["Id"],
                        "name": project["name"],
                        "priority": project["priority"],
                    },
                    target={
                        "id": target["Id"],
                        "name": target["name"],
                        "ra": target["ra"],
                        "dec": target["dec"],
                    },
                    plans=actionable,
                    summary=" | ".join(summary_parts),
                )
            )
    return out


# ---- Planner entry points ------------------------------------------------

def plan_next(
    *,
    sequence_name: str,
    ts_db_path: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> PlannerDecision:
    """First actionable Target Scheduler target (single-target front door).

    Backward-compatible: populates the flat project/target/plans fields and a
    one-element ``targets`` list.
    """
    conn = _connect_ro(ts_db_path)
    try:
        actionable = _collect_actionable(conn, profile_id)
    finally:
        conn.close()

    if not actionable:
        return PlannerDecision(
            action=PlannerAction.NO_WORK,
            summary="No actionable Target Scheduler targets",
        )

    first = actionable[0]
    return PlannerDecision(
        action=PlannerAction.IMAGE,
        sequence_name=sequence_name,
        project=first.project,
        target=first.target,
        plans=first.plans,
        summary=first.summary,
        targets=[first],
    )


def plan_all(
    *,
    sequence_name: str,
    ts_db_path: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> PlannerDecision:
    """Every actionable Target Scheduler target, in NINA imaging order.

    Models astro5's ``jan_2026_north_multitarget`` shape: a Targets_Container
    holding one self-contained block per target. The flat project/target/plans
    fields mirror the first target for callers that still read them.
    """
    conn = _connect_ro(ts_db_path)
    try:
        actionable = _collect_actionable(conn, profile_id)
    finally:
        conn.close()

    if not actionable:
        return PlannerDecision(
            action=PlannerAction.NO_WORK,
            summary="No actionable Target Scheduler targets",
        )

    first = actionable[0]
    summary = f"{len(actionable)} target(s): " + " → ".join(
        tp.target["name"] for tp in actionable
    )
    return PlannerDecision(
        action=PlannerAction.IMAGE,
        sequence_name=sequence_name,
        project=first.project,
        target=first.target,
        plans=first.plans,
        summary=summary,
        targets=actionable,
    )
