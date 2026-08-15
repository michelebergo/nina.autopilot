"""Tests for wiki_ingest.py — the wiki consolidation agent.

A temp wiki with SCHEMA/index/raw files stands in for %LOCALAPPDATA%/NINA/llmwiki;
the fake LLM client returns canned JSON for the two ingest passes. The tests pin
the safety properties: only wiki/ paths are writable, ingest is incremental over
append-only raw files, and malformed model output degrades to a no-op.
"""

import json

import pytest

from nina_autopilot.llm import LLMClient
from nina_autopilot.wiki_ingest import run_ingest, run_lint

from tests.test_llm import FakeAnthropic


PAGE = """---
title: Test Camera
type: entity
created: 2026-08-15
updated: 2026-08-15
tags: [camera]
---

# Test Camera

> A camera.
"""


def make_wiki(tmp_path):
    root = tmp_path / "llmwiki"
    (root / "raw").mkdir(parents=True)
    (root / "wiki" / "entities").mkdir(parents=True)
    (root / "SCHEMA.md").write_text("# SCHEMA\nrules here\n", encoding="utf-8")
    (root / "index.md").write_text("# Index\n\n- wiki/entities/test-camera.md\n", encoding="utf-8")
    (root / "log.md").write_text("# Log\n", encoding="utf-8")
    (root / "wiki" / "entities" / "test-camera.md").write_text(PAGE, encoding="utf-8")
    (root / "raw" / "assistant-2026-08-15.md").write_text(
        "# assistant\n\n- 13:40 — Test Camera shows amp glow above gain 56\n", encoding="utf-8"
    )
    return root


def make_llm(*responses: str) -> LLMClient:
    fake = FakeAnthropic()
    for r in responses:
        fake.messages.queue(r)
    return LLMClient(client=fake)


class TestIngest:
    async def test_consolidates_notes_into_wiki_page(self, tmp_path):
        root = make_wiki(tmp_path)
        new_page = PAGE.replace("> A camera.", "> A camera.\n\nAmp glow above gain 56 (raw/assistant-2026-08-15.md).")
        llm = make_llm(
            json.dumps({"read": ["wiki/entities/test-camera.md"]}),
            json.dumps({
                "edits": [{"path": "wiki/entities/test-camera.md", "content": new_page}],
                "index": None,
                "log_line": "amp glow noted on Test Camera",
            }),
        )

        report = await run_ingest(llm, root=root)

        assert report.pages_written == ["wiki/entities/test-camera.md"]
        assert "amp glow" in (root / "wiki" / "entities" / "test-camera.md").read_text(encoding="utf-8").lower()
        assert "[ingest] amp glow noted on Test Camera" in (root / "log.md").read_text(encoding="utf-8")

    async def test_ingest_is_incremental_over_append_only_raw(self, tmp_path):
        root = make_wiki(tmp_path)
        llm = make_llm(
            json.dumps({"read": []}),
            json.dumps({"edits": [], "index": None, "log_line": "nothing durable"}),
        )
        await run_ingest(llm, root=root)

        # No new raw lines -> second run skips without an LLM call.
        report = await run_ingest(make_llm(), root=root)
        assert report.skipped_reason == "no new raw observations"

        # Appending one line makes exactly that line pending.
        with open(root / "raw" / "assistant-2026-08-15.md", "a", encoding="utf-8") as fh:
            fh.write("- 14:00 — second note\n")
        llm2 = make_llm(
            json.dumps({"read": []}),
            json.dumps({"edits": [], "index": None, "log_line": "x"}),
        )
        report = await run_ingest(llm2, root=root)
        assert report.new_lines == 1

    async def test_rejects_edits_outside_wiki(self, tmp_path):
        root = make_wiki(tmp_path)
        llm = make_llm(
            json.dumps({"read": []}),
            json.dumps({
                "edits": [
                    {"path": "raw/assistant-2026-08-15.md", "content": "---\nhacked\n"},
                    {"path": "SCHEMA.md", "content": "---\nhacked\n"},
                    {"path": "../outside.md", "content": "---\nhacked\n"},
                    {"path": "wiki/entities/ok.md", "content": "---\ntitle: Ok\n---\n\n# Ok\n"},
                ],
                "index": None,
                "log_line": "x",
            }),
        )

        report = await run_ingest(llm, root=root)

        assert report.pages_written == ["wiki/entities/ok.md"]
        assert "hacked" not in (root / "raw" / "assistant-2026-08-15.md").read_text(encoding="utf-8")
        assert "hacked" not in (root / "SCHEMA.md").read_text(encoding="utf-8")
        assert not (tmp_path / "outside.md").exists()

    async def test_unparseable_edit_output_is_a_noop(self, tmp_path):
        root = make_wiki(tmp_path)
        llm = make_llm(json.dumps({"read": []}), "sorry, I ate the JSON")

        report = await run_ingest(llm, root=root)

        assert report.pages_written == []
        assert report.skipped_reason is not None
        # State untouched -> the notes stay pending for the next run.
        llm2 = make_llm(
            json.dumps({"read": []}),
            json.dumps({"edits": [], "index": None, "log_line": "x"}),
        )
        report2 = await run_ingest(llm2, root=root)
        assert report2.skipped_reason is None

    async def test_dry_run_writes_nothing(self, tmp_path):
        root = make_wiki(tmp_path)
        llm = make_llm(
            json.dumps({"read": []}),
            json.dumps({
                "edits": [{"path": "wiki/entities/new.md", "content": "---\ntitle: New\n---\n\n# New\n"}],
                "index": None,
                "log_line": "x",
            }),
        )

        report = await run_ingest(llm, root=root, dry_run=True)

        assert report.pages_written == ["wiki/entities/new.md"]
        assert not (root / "wiki" / "entities" / "new.md").exists()
        # Pending state not consumed either.
        assert (await run_ingest(make_llm(json.dumps({"read": []}), "{}"), root=root)).new_lines > 0


class TestLint:
    async def test_mechanical_checks_and_report(self, tmp_path):
        root = make_wiki(tmp_path)
        (root / "wiki" / "entities" / "broken.md").write_text(
            "# No frontmatter\n\nSee [[does-not-exist]].\n", encoding="utf-8"
        )
        llm = make_llm(json.dumps({"issues": [{"page": "wiki/entities/test-camera.md", "issue": "uncited claim"}]}))

        report = await run_lint(llm, root=root)

        assert any("does-not-exist" in b for b in report.broken_links)
        assert "wiki/entities/broken.md" in report.missing_frontmatter
        assert report.llm_issues and report.llm_issues[0]["issue"] == "uncited claim"
        report_text = (root / "wiki" / "syntheses" / "lint-report.md").read_text(encoding="utf-8")
        assert "does-not-exist" in report_text
        assert "[lint]" in (root / "log.md").read_text(encoding="utf-8")
