"""Build a NINA Advanced Sequence JSON from a multi-target PlannerDecision.

Inspired by astro5's ``jan_2026_north_multitarget`` reference sequence, which
uses a four-container structure:

    [ GlobalTriggers, Start_Container, Targets_Container, End_Container ]

Each container is a dict with ``Name``, ``Conditions``, ``Triggers``,
``Items``, and ``Status``. The ``Targets_Container`` holds one sub-container
per target, and each target sub-container has:

    PREPARE_TARGET  → slew & center, switch filter, autofocus, start guiding
    OSC IMAGING     → smart exposures per filter (conditions + triggers + items)
    TARGET END      → stop guiding
    GET FLATS       → flat panel loop (optional, controlled by ``flats`` flag)

The builder is deterministic — no LLM, no network calls. It takes a
``PlannerDecision`` (from ``plan_all()``) plus a ``SequenceBuildConfig`` and
returns a JSON-serialisable list of dicts.

The output is written to a file and loaded into NINA via the Advanced API's
``sequence/load?sequenceName=…`` endpoint. NINA expects the file in its
sequences directory (typically ``%LOCALAPPDATA%\\NINA\\Sequences``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .planner import PlannerAction, PlannerDecision, TargetPlan


# Container / item type constants — match NINA's serialised names.
_CONTAINER = "SequenceItem.Container"
_INSTRUCTION = "SequenceItem.Instruction"
_TRIGGER = "SequenceItem.Trigger"
_CONDITION = "SequenceItem.Condition"

# Status values NINA uses for fresh sequences.
_STATUS_CREATED = "CREATED"


@dataclass
class SequenceBuildConfig:
    """Parameters that shape the generated sequence.

    Defaults mirror astro5's reference sequence. Override per-session.
    """

    # Target imaging parameters
    exposure_count: int = 24
    """Frames per filter per target (NINA Smart Exposure count)."""

    dither_every: int = 5
    """Dither every N frames (0 = no dithering)."""

    # Safety / altitude gates
    min_altitude_deg: float = 40.0
    """Wait-for-altitude gate before each target's slew."""

    # Autofocus triggers
    af_after_temp_change_deg: float = 5.0
    """Re-run autofocus after this much temperature drift."""

    af_after_hfr_delta: float = 10.0
    """Re-run autofocus after HFR increases by this percentage."""

    af_after_hfr_sample_size: int = 10
    """Sample size for the HFR-delta trigger."""

    # Drift / re-centering
    center_after_drift_px: float = 10.0
    """Re-center target after this much drift (pixels)."""

    # Guiding
    force_calibration_on_first_target: bool = True
    """First target's Start Guiding forces calibration; subsequent ones reuse."""

    # Flats
    include_flats: bool = True
    """Add a GET FLATS block after each target's imaging block."""

    # Start container
    include_start_container: bool = True
    """Emit the Start_Container (equipment connect + polar alignment)."""

    # End container
    include_end_container: bool = True
    """Emit the End_Container (park + warm + close)."""

    # Camera cooling
    cooler_temp_c: float = -10.0
    """Cooler setpoint for the Start_Container cool-camera step."""

    # Output
    sequences_dir: Optional[str] = None
    """Where to write the JSON file. Defaults to ``%LOCALAPPDATA%\\NINA\\Sequences``."""


def _empty_container(name: str) -> dict[str, Any]:
    return {
        "Name": name,
        "Type": _CONTAINER,
        "Conditions": [],
        "Triggers": [],
        "Items": [],
        "Status": _STATUS_CREATED,
    }


def _instruction(name: str, **props: Any) -> dict[str, Any]:
    """A leaf sequence item (slew, cool, capture, etc.)."""
    item: dict[str, Any] = {
        "Name": name,
        "Type": _INSTRUCTION,
        "Status": _STATUS_CREATED,
    }
    item.update(props)
    return item


def _trigger(name: str, **props: Any) -> dict[str, Any]:
    t: dict[str, Any] = {
        "Name": name,
        "Type": _TRIGGER,
        "Status": _STATUS_CREATED,
    }
    t.update(props)
    return t


def _condition(name: str, **props: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "Name": name,
        "Type": _CONDITION,
        "Status": _STATUS_CREATED,
    }
    c.update(props)
    return c


# ---- Start_Container -----------------------------------------------------

def _build_start_container(cfg: SequenceBuildConfig) -> dict[str, Any]:
    """Equipment connect + polar alignment, run once at session start."""
    c = _empty_container("Start_Container")

    c["Items"] = [
        _instruction(
            "Wait for Time",
            description="Wait until astronomical darkness",
        ),
        _instruction(
            "Connect Equipment",
            description="Connect all configured devices",
        ),
        _instruction(
            "Cool Camera",
            temperature=cfg.cooler_temp_c,
            description=f"Cool to {cfg.cooler_temp_c}°C",
        ),
        _instruction(
            "Polar Alignment",
            description="Three-point polar alignment",
        ),
    ]
    return c


# ---- Per-target sub-container --------------------------------------------

def _build_prepare_target(tp: TargetPlan, cfg: SequenceBuildConfig) -> dict[str, Any]:
    """PREPARE_TARGET: slew, filter switch, autofocus, start guiding."""
    c = _empty_container("PREPARE_TARGET")

    items: list[dict[str, Any]] = [
        _instruction("Unpark Mount"),
        _instruction(
            "Slew & Center",
            ra=tp.target["ra"],
            dec=tp.target["dec"],
            epoch="J2000",
            targetName=tp.target["name"],
        ),
        _instruction(
            "Wait for Altitude",
            minimumAltitude=cfg.min_altitude_deg,
            targetName=tp.target["name"],
        ),
    ]

    # Switch to the first filter we'll image with
    if tp.plans:
        items.append(
            _instruction("Switch Filter", filter=tp.plans[0]["filter_name"])
        )

    items.append(_instruction("Autofocus"))
    items.append(
        _instruction(
            "Start Guiding",
            forceCalibration=cfg.force_calibration_on_first_target,
        )
    )

    c["Items"] = items
    return c


def _build_imaging_block(tp: TargetPlan, cfg: SequenceBuildConfig) -> dict[str, Any]:
    """OSC IMAGING: smart exposures per filter with AF/drift triggers."""
    c = _empty_container("OSC IMAGING")

    # Conditions — loop until time / altitude
    c["Conditions"] = [
        _condition("Loop Until Time"),
        _condition(
            "Loop until Altitude Below",
            minimumAltitude=cfg.min_altitude_deg,
        ),
    ]

    # Triggers — AF after temp/HFR drift, center after drift, restore guiding
    c["Triggers"] = [
        _trigger(
            "AF After Temp Change",
            temperatureDelta=cfg.af_after_temp_change_deg,
        ),
        _trigger(
            "AF After HFR Increase",
            deltaHFR=cfg.af_after_hfr_delta,
            sampleSize=cfg.af_after_hfr_sample_size,
        ),
        _trigger(
            "Center After Drift",
            targetDrift=cfg.center_after_drift_px,
        ),
        _trigger("Restore Guiding"),
    ]

    # Items — one Smart Exposure per filter
    items: list[dict[str, Any]] = []
    for plan in tp.plans:
        items.append(
            _instruction(
                "Smart Exposure",
                filter=plan["filter_name"],
                exposureTime=plan["exposure"],
                count=cfg.exposure_count,
                ditherEvery=cfg.dither_every if cfg.dither_every > 0 else None,
                gain=plan.get("gain"),
                offset=plan.get("offset"),
                binning=plan.get("bin"),
                imageType="LIGHT",
            )
        )
    c["Items"] = items
    return c


def _build_target_end() -> dict[str, Any]:
    """TARGET END: stop guiding."""
    c = _empty_container("TARGET END")
    c["Items"] = [_instruction("Stop Guiding")]
    return c


def _build_flats_block(tp: TargetPlan) -> dict[str, Any]:
    """GET FLATS: close flat panel, auto-brightness flat loop, open panel."""
    c = _empty_container("GET FLATS OSC IMAGING")
    c["Items"] = [
        _instruction("Close Flat Panel"),
        _instruction("Auto Brightness Flat"),
        _instruction("Open Flat Panel"),
    ]
    return c


def _build_target_container(
    tp: TargetPlan,
    cfg: SequenceBuildConfig,
    is_first: bool,
) -> dict[str, Any]:
    """One self-contained target block inside Targets_Container.

    Mirrors astro5's per-target sub-containers: PREPARE → IMAGING → END → FLATS.
    """
    c = _empty_container(f"Target: {tp.target['name']}")

    # Triggers on the target container itself (meridian flip, failure alert)
    c["Triggers"] = [
        _trigger("Meridian Flip"),
    ]

    # Force calibration only on the first target
    local_cfg = cfg
    if not is_first:
        # Shallow-copy and disable force-cal for subsequent targets
        local_cfg = SequenceBuildConfig(
            exposure_count=cfg.exposure_count,
            dither_every=cfg.dither_every,
            min_altitude_deg=cfg.min_altitude_deg,
            af_after_temp_change_deg=cfg.af_after_temp_change_deg,
            af_after_hfr_delta=cfg.af_after_hfr_delta,
            af_after_hfr_sample_size=cfg.af_after_hfr_sample_size,
            center_after_drift_px=cfg.center_after_drift_px,
            force_calibration_on_first_target=False,
            include_flats=cfg.include_flats,
            include_start_container=cfg.include_start_container,
            include_end_container=cfg.include_end_container,
            cooler_temp_c=cfg.cooler_temp_c,
            sequences_dir=cfg.sequences_dir,
        )

    items: list[dict[str, Any]] = [
        _build_prepare_target(tp, local_cfg),
        _build_imaging_block(tp, local_cfg),
        _build_target_end(),
    ]
    if cfg.include_flats:
        items.append(_build_flats_block(tp))

    c["Items"] = items
    return c


# ---- Targets_Container ---------------------------------------------------

def _build_targets_container(
    targets: list[TargetPlan],
    cfg: SequenceBuildConfig,
) -> dict[str, Any]:
    """Wraps one sub-container per actionable target."""
    c = _empty_container("Targets_Container")
    c["Items"] = [
        _build_target_container(tp, cfg, is_first=(i == 0))
        for i, tp in enumerate(targets)
    ]
    return c


# ---- End_Container -------------------------------------------------------

def _build_end_container() -> dict[str, Any]:
    """Park mount, warm camera, close dome — run once at session end."""
    c = _empty_container("End_Container")
    c["Items"] = [
        _instruction("Stop Guiding"),
        _instruction("Park Mount"),
        _instruction("Warm Camera"),
        _instruction("Close Dome Shutter"),
        _instruction("Disconnect Equipment"),
    ]
    return c


# ---- Public API ----------------------------------------------------------

def build_sequence(
    decision: PlannerDecision,
    cfg: Optional[SequenceBuildConfig] = None,
) -> list[dict[str, Any]]:
    """Build a NINA Advanced Sequence JSON from a multi-target planner decision.

    Returns a JSON-serialisable list of containers:
    ``[GlobalTriggers, Start_Container, Targets_Container, End_Container]``.

    Raises ``ValueError`` if the decision is NO_WORK.
    """
    if decision.action is PlannerAction.NO_WORK:
        raise ValueError("Cannot build a sequence from a NO_WORK decision")
    if not decision.targets:
        raise ValueError("Decision has no targets — cannot build sequence")

    cfg = cfg or SequenceBuildConfig()

    sequence: list[dict[str, Any]] = [
        [],  # GlobalTriggers — empty array (NINA convention)
    ]

    if cfg.include_start_container:
        sequence.append(_build_start_container(cfg))

    sequence.append(_build_targets_container(decision.targets, cfg))

    if cfg.include_end_container:
        sequence.append(_build_end_container())

    return sequence


def _default_sequences_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise RuntimeError("LOCALAPPDATA is not set — cannot locate NINA Sequences dir")
    return Path(base) / "NINA" / "Sequences"


def write_sequence(
    decision: PlannerDecision,
    filename: str,
    cfg: Optional[SequenceBuildConfig] = None,
) -> Path:
    """Build the sequence JSON and write it to the NINA sequences directory.

    Returns the path to the written file. The file name (without ``.json``
    extension) is what NINA's ``sequence/load?sequenceName=…`` expects.
    """
    cfg = cfg or SequenceBuildConfig()
    seq = build_sequence(decision, cfg)

    out_dir = Path(cfg.sequences_dir) if cfg.sequences_dir else _default_sequences_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".json"):
        filename += ".json"

    out_path = out_dir / filename
    out_path.write_text(json.dumps(seq, indent=2), encoding="utf-8")
    return out_path