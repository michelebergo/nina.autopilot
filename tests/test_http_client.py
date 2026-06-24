"""Tests for http_client.py — specifically the sequence/state response parser.

The bug this covers: NINA's /v2/api/sequence/state returns Response as a
LIST of nested containers, each with its own Status. The old code treated
Response as a dict with a "State" key (which doesn't exist), so the
Conductor never saw "Finished" and got stuck in IMAGING forever.
"""

import pytest

from nina_autopilot.http_client import HttpNinaClient


def _collect(container_list):
    return HttpNinaClient._collect_sequence_statuses(container_list)


class TestCollectSequenceStatuses:
    def test_empty_list_returns_empty(self):
        assert _collect([]) == []

    def test_flat_containers(self):
        assert _collect([
            {"Name": "Start", "Status": "FINISHED"},
            {"Name": "End", "Status": "CREATED"},
        ]) == ["FINISHED", "CREATED"]

    def test_skips_containers_without_status(self):
        """Wrapper entries like {"GlobalTriggers": []} have no Status — ignore them."""
        assert _collect([
            {"GlobalTriggers": []},
            {"Name": "X", "Status": "FINISHED"},
        ]) == ["FINISHED"]

    def test_walks_nested_items(self):
        """The real NINA response nests containers via Items."""
        tree = [
            {"Name": "Targets_Container", "Status": "RUNNING", "Items": [
                {"Name": "autopilot_ts_sequence_Container", "Status": "RUNNING", "Items": [
                    {"Name": "Annotation", "Status": "FINISHED"},
                    {"Name": "Take Exposure", "Status": "RUNNING"},
                ]},
            ]},
        ]
        statuses = _collect(tree)
        assert "RUNNING" in statuses
        assert statuses.count("RUNNING") == 3
        assert statuses.count("FINISHED") == 1

    def test_handles_non_dict_entries(self):
        """Defensive — random strings / None in the list should not crash."""
        assert _collect([None, "garbage", 42, {"Status": "FINISHED"}]) == ["FINISHED"]


class TestGetSequenceStateParsing:
    """Drive HttpNinaClient.get_sequence_state with a monkeypatched _get to
    cover every state-derivation branch without touching HTTP."""

    async def _with_response(self, response_obj):
        client = HttpNinaClient("http://test")

        async def fake_get(path: str):
            return {"Response": response_obj, "Success": True, "Error": "", "StatusCode": 200}

        client._get = fake_get
        return await client.get_sequence_state()

    async def test_all_finished_maps_to_finished(self):
        """The original bug — NINA reports all containers FINISHED but the
        old code returned {State: 'Unknown'} and the Conductor never exited."""
        result = await self._with_response([
            {"Name": "Start_Container", "Status": "FINISHED"},
            {"Name": "Targets_Container", "Status": "FINISHED", "Items": [
                {"Name": "autopilot_ts_sequence_Container", "Status": "FINISHED", "Items": [
                    {"Name": "Take Exposure", "Status": "FINISHED"},
                ]},
            ]},
            {"Name": "End_Container", "Status": "FINISHED"},
        ])
        assert result["State"] == "Finished"

    async def test_any_running_maps_to_running(self):
        result = await self._with_response([
            {"Name": "Start_Container", "Status": "FINISHED"},
            {"Name": "Targets_Container", "Status": "RUNNING", "Items": [
                {"Name": "Take Exposure", "Status": "RUNNING"},
            ]},
        ])
        assert result["State"] == "Running"

    async def test_all_created_maps_to_idle(self):
        """Sequence loaded but never started."""
        result = await self._with_response([
            {"Name": "Start_Container", "Status": "CREATED"},
            {"Name": "End_Container", "Status": "CREATED"},
        ])
        assert result["State"] == "Idle"

    async def test_finished_and_skipped_mix_is_finished(self):
        result = await self._with_response([
            {"Name": "A", "Status": "FINISHED"},
            {"Name": "B", "Status": "SKIPPED"},
        ])
        assert result["State"] == "Finished"

    async def test_empty_response_list_is_idle(self):
        result = await self._with_response([])
        assert result["State"] == "Idle"

    async def test_global_triggers_wrapper_doesnt_force_unknown(self):
        """Real NINA response starts with [{GlobalTriggers: []}, ...] — that
        wrapper has no Status and must not pollute the state derivation."""
        result = await self._with_response([
            {"GlobalTriggers": []},
            {"Name": "Start_Container", "Status": "FINISHED"},
            {"Name": "End_Container", "Status": "FINISHED"},
        ])
        assert result["State"] == "Finished"
