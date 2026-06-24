"""Tests for sequence_builder.py — NINA multi-target sequence JSON generator.

Validates the four-container structure (GlobalTriggers, Start, Targets, End)
and the per-target sub-container shape (PREPARE → IMAGING → END → FLATS)
against the astro5 reference design.
"""

import json

import pytest

from nina_autopilot.planner import (
    PlannerAction,
    PlannerDecision,
    TargetPlan,
)
from nina_autopilot.sequence_builder import (
    SequenceBuildConfig,
    build_sequence,
    write_sequence,
)


# ---- Fixtures ------------------------------------------------------------

def _make_target_plan(
    name: str = "M81 Galaxy",
    ra: float = 148.888,
    dec: float = 69.065,
    filter_name: str = "L",
    exposure: float = 300.0,
    remaining: int = 20,
) -> TargetPlan:
    return TargetPlan(
        project={"id": 1, "name": "M81", "priority": 100},
        target={"id": 10, "name": name, "ra": ra, "dec": dec},
        plans=[
            {
                "plan_id": 1000,
                "template_name": "L",
                "filter_name": filter_name,
                "exposure": exposure,
                "gain": 100,
                "offset": 10,
                "bin": 1,
                "desired": 30,
                "acquired": 10,
                "accepted": 8,
                "remaining": remaining,
                "enabled": True,
            }
        ],
        summary=f"target={name} | project=M81 | {filter_name}×{remaining} ({exposure:.0f}s)",
    )


def _make_decision(targets: list[TargetPlan]) -> PlannerDecision:
    first = targets[0]
    return PlannerDecision(
        action=PlannerAction.IMAGE,
        sequence_name="autopilot_multitarget.json",
        project=first.project,
        target=first.target,
        plans=first.plans,
        summary=f"{len(targets)} target(s): " + " → ".join(t.target["name"] for t in targets),
        targets=targets,
    )


@pytest.fixture
def single_target_decision():
    return _make_decision([_make_target_plan()])


@pytest.fixture
def multi_target_decision():
    return _make_decision([
        _make_target_plan(name="M81 Galaxy", ra=148.888, dec=69.065),
        _make_target_plan(name="NGC 7000", ra=314.750, dec=44.330, filter_name="HA", exposure=180.0),
    ])


# ---- Top-level structure -------------------------------------------------

class TestSequenceStructure:
    def test_four_containers_by_default(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        # [GlobalTriggers, Start_Container, Targets_Container, End_Container]
        assert len(seq) == 4
        assert seq[0] == []  # GlobalTriggers is an empty array
        assert seq[1]["Name"] == "Start_Container"
        assert seq[2]["Name"] == "Targets_Container"
        assert seq[3]["Name"] == "End_Container"

    def test_skip_start_and_end(self, single_target_decision):
        cfg = SequenceBuildConfig(include_start_container=False, include_end_container=False)
        seq = build_sequence(single_target_decision, cfg)
        assert len(seq) == 2
        assert seq[0] == []
        assert seq[1]["Name"] == "Targets_Container"

    def test_all_containers_have_created_status(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        for c in seq[1:]:
            assert c["Status"] == "CREATED"

    def test_no_work_raises(self):
        d = PlannerDecision(action=PlannerAction.NO_WORK, summary="nothing")
        with pytest.raises(ValueError, match="NO_WORK"):
            build_sequence(d)

    def test_empty_targets_raises(self):
        d = PlannerDecision(
            action=PlannerAction.IMAGE,
            sequence_name="x.json",
            target={"id": 1, "name": "X", "ra": 0, "dec": 0},
            summary="x",
        )
        with pytest.raises(ValueError, match="no targets"):
            build_sequence(d)


# ---- Start_Container -----------------------------------------------------

class TestStartContainer:
    def test_has_equipment_connect_and_cool(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        start = seq[1]
        names = [item["Name"] for item in start["Items"]]
        assert "Connect Equipment" in names
        assert "Cool Camera" in names
        assert "Polar Alignment" in names

    def test_cool_temp_from_config(self, single_target_decision):
        cfg = SequenceBuildConfig(cooler_temp_c=-15.0)
        seq = build_sequence(single_target_decision, cfg)
        start = seq[1]
        cool = next(i for i in start["Items"] if i["Name"] == "Cool Camera")
        assert cool["temperature"] == -15.0


# ---- Targets_Container ---------------------------------------------------

class TestTargetsContainer:
    def test_one_sub_container_per_target(self, multi_target_decision):
        seq = build_sequence(multi_target_decision)
        targets_c = seq[2]
        assert len(targets_c["Items"]) == 2
        assert targets_c["Items"][0]["Name"] == "Target: M81 Galaxy"
        assert targets_c["Items"][1]["Name"] == "Target: NGC 7000"

    def test_target_container_has_meridian_flip_trigger(self, multi_target_decision):
        seq = build_sequence(multi_target_decision)
        tc = seq[2]["Items"][0]
        trigger_names = [t["Name"] for t in tc["Triggers"]]
        assert "Meridian Flip" in trigger_names

    def test_target_blocks_in_planner_order(self, multi_target_decision):
        seq = build_sequence(multi_target_decision)
        names = [c["Name"] for c in seq[2]["Items"]]
        assert names == ["Target: M81 Galaxy", "Target: NGC 7000"]


# ---- Per-target sub-container shape --------------------------------------

class TestTargetSubContainer:
    def test_prepare_imaging_end_flats_blocks(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        tc = seq[2]["Items"][0]
        block_names = [i["Name"] for i in tc["Items"]]
        assert block_names == [
            "PREPARE_TARGET",
            "OSC IMAGING",
            "TARGET END",
            "GET FLATS OSC IMAGING",
        ]

    def test_flats_omitted_when_disabled(self, single_target_decision):
        cfg = SequenceBuildConfig(include_flats=False)
        seq = build_sequence(single_target_decision, cfg)
        tc = seq[2]["Items"][0]
        block_names = [i["Name"] for i in tc["Items"]]
        assert "GET FLATS OSC IMAGING" not in block_names

    def test_prepare_slews_to_target_coordinates(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        tc = seq[2]["Items"][0]
        prepare = tc["Items"][0]
        slew = next(i for i in prepare["Items"] if i["Name"] == "Slew & Center")
        assert slew["ra"] == 148.888
        assert slew["dec"] == 69.065
        assert slew["targetName"] == "M81 Galaxy"
        assert slew["epoch"] == "J2000"

    def test_prepare_switches_to_first_filter(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        tc = seq[2]["Items"][0]
        prepare = tc["Items"][0]
        switch = next(i for i in prepare["Items"] if i["Name"] == "Switch Filter")
        assert switch["filter"] == "L"

    def test_imaging_has_smart_exposure_per_plan(self, multi_target_decision):
        seq = build_sequence(multi_target_decision)
        # Second target uses HA filter
        tc2 = seq[2]["Items"][1]
        imaging = tc2["Items"][1]
        assert len(imaging["Items"]) == 1
        exp = imaging["Items"][0]
        assert exp["Name"] == "Smart Exposure"
        assert exp["filter"] == "HA"
        assert exp["exposureTime"] == 180.0
        assert exp["imageType"] == "LIGHT"

    def test_imaging_has_af_and_drift_triggers(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        tc = seq[2]["Items"][0]
        imaging = tc["Items"][1]
        trigger_names = [t["Name"] for t in imaging["Triggers"]]
        assert "AF After Temp Change" in trigger_names
        assert "AF After HFR Increase" in trigger_names
        assert "Center After Drift" in trigger_names
        assert "Restore Guiding" in trigger_names

    def test_imaging_has_loop_conditions(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        tc = seq[2]["Items"][0]
        imaging = tc["Items"][1]
        cond_names = [c["Name"] for c in imaging["Conditions"]]
        assert "Loop Until Time" in cond_names
        assert "Loop until Altitude Below" in cond_names

    def test_target_end_stops_guiding(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        tc = seq[2]["Items"][0]
        end_block = tc["Items"][2]
        assert end_block["Name"] == "TARGET END"
        assert end_block["Items"][0]["Name"] == "Stop Guiding"

    def test_exposure_count_from_config(self, single_target_decision):
        cfg = SequenceBuildConfig(exposure_count=36)
        seq = build_sequence(single_target_decision, cfg)
        tc = seq[2]["Items"][0]
        imaging = tc["Items"][1]
        assert imaging["Items"][0]["count"] == 36

    def test_dither_every_from_config(self, single_target_decision):
        cfg = SequenceBuildConfig(dither_every=3)
        seq = build_sequence(single_target_decision, cfg)
        tc = seq[2]["Items"][0]
        imaging = tc["Items"][1]
        assert imaging["Items"][0]["ditherEvery"] == 3

    def test_dither_zero_means_none(self, single_target_decision):
        cfg = SequenceBuildConfig(dither_every=0)
        seq = build_sequence(single_target_decision, cfg)
        tc = seq[2]["Items"][0]
        imaging = tc["Items"][1]
        assert imaging["Items"][0]["ditherEvery"] is None


# ---- Force calibration only on first target ------------------------------

class TestGuidingCalibration:
    def test_first_target_forces_calibration(self, multi_target_decision):
        seq = build_sequence(multi_target_decision)
        first = seq[2]["Items"][0]
        prepare = first["Items"][0]
        guide = next(i for i in prepare["Items"] if i["Name"] == "Start Guiding")
        assert guide["forceCalibration"] is True

    def test_second_target_does_not_force_calibration(self, multi_target_decision):
        seq = build_sequence(multi_target_decision)
        second = seq[2]["Items"][1]
        prepare = second["Items"][0]
        guide = next(i for i in prepare["Items"] if i["Name"] == "Start Guiding")
        assert guide["forceCalibration"] is False


# ---- End_Container -------------------------------------------------------

class TestEndContainer:
    def test_park_warm_close_disconnect(self, single_target_decision):
        seq = build_sequence(single_target_decision)
        end = seq[3]
        names = [i["Name"] for i in end["Items"]]
        assert "Park Mount" in names
        assert "Warm Camera" in names
        assert "Close Dome Shutter" in names
        assert "Disconnect Equipment" in names


# ---- write_sequence ------------------------------------------------------

class TestWriteSequence:
    def test_writes_valid_json_to_sequences_dir(self, single_target_decision, tmp_path):
        cfg = SequenceBuildConfig(sequences_dir=str(tmp_path))
        path = write_sequence(single_target_decision, "test_seq.json", cfg)
        assert path.exists()
        assert path.name == "test_seq.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 4

    def test_appends_json_extension(self, single_target_decision, tmp_path):
        cfg = SequenceBuildConfig(sequences_dir=str(tmp_path))
        path = write_sequence(single_target_decision, "no_ext", cfg)
        assert path.name == "no_ext.json"

    def test_creates_dir_if_missing(self, single_target_decision, tmp_path):
        out = tmp_path / "deeply" / "nested" / "seqs"
        cfg = SequenceBuildConfig(sequences_dir=str(out))
        path = write_sequence(single_target_decision, "x.json", cfg)
        assert path.exists()