"""Tests for state.py — minimal SQLite session store for the Conductor."""

import pytest

from nina_autopilot.state import SessionStore, open_store


@pytest.fixture
def store(tmp_path):
    return open_store(tmp_path / "session.sqlite")


class TestOpenStore:
    def test_open_creates_schema(self, tmp_path):
        s = open_store(tmp_path / "new.sqlite")
        assert isinstance(s, SessionStore)
        # Should be safe to call again on the same file (idempotent migration).
        s2 = open_store(tmp_path / "new.sqlite")
        assert isinstance(s2, SessionStore)


class TestSessions:
    def test_start_session_returns_id(self, store):
        sid = store.start_session(sequence_file="tonight.json")
        assert isinstance(sid, int)
        assert sid >= 1

    def test_start_session_records_metadata(self, store):
        sid = store.start_session(sequence_file="tonight.json")
        sess = store.get_session(sid)
        assert sess["sequence_file"] == "tonight.json"
        assert sess["started_at"] is not None
        assert sess["ended_at"] is None
        assert sess["phase"] == "BOOT"

    def test_set_phase_updates(self, store):
        sid = store.start_session("x.json")
        store.set_phase(sid, "IMAGING")
        assert store.get_session(sid)["phase"] == "IMAGING"

    def test_end_session_marks_finished(self, store):
        sid = store.start_session("x.json")
        store.end_session(sid, reason="sequence_complete")
        sess = store.get_session(sid)
        assert sess["ended_at"] is not None
        assert sess["end_reason"] == "sequence_complete"

    def test_current_session_picks_unfinished(self, store):
        """current_session returns the most recent session that hasn't ended."""
        assert store.current_session() is None
        sid1 = store.start_session("a.json")
        store.end_session(sid1, reason="done")
        # Finished — should not be picked up
        assert store.current_session() is None
        sid2 = store.start_session("b.json")
        cur = store.current_session()
        assert cur is not None
        assert cur["id"] == sid2


class TestEvents:
    def test_record_and_fetch_events(self, store):
        sid = store.start_session("x.json")
        store.record_event(sid, kind="PHASE_CHANGE", payload={"from": "BOOT", "to": "IMAGING"})
        store.record_event(sid, kind="SAFETY_WARN", payload={"reasons": ["humidity"]})
        events = store.list_events(sid)
        assert len(events) == 2
        kinds = [e["kind"] for e in events]
        assert kinds == ["PHASE_CHANGE", "SAFETY_WARN"]

    def test_events_preserve_payload(self, store):
        sid = store.start_session("x.json")
        store.record_event(sid, kind="X", payload={"foo": "bar", "n": 42})
        events = store.list_events(sid)
        assert events[0]["payload"] == {"foo": "bar", "n": 42}

    def test_events_scoped_per_session(self, store):
        s1 = store.start_session("a.json")
        s2 = store.start_session("b.json")
        store.record_event(s1, kind="A", payload={})
        store.record_event(s2, kind="B", payload={})
        assert [e["kind"] for e in store.list_events(s1)] == ["A"]
        assert [e["kind"] for e in store.list_events(s2)] == ["B"]
