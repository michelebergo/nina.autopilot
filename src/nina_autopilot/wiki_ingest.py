"""Wiki ingest agent — consolidates raw/ observations into the NINA LLM wiki.

The wiki (default %LOCALAPPDATA%/NINA/llmwiki) is the shared second brain of the
NINA AI plugins: they append immutable observations to raw/ (daily weather
digests, consented chat notes); THIS agent, run nightly or on demand with a
capable model, turns them into consolidated wiki/ pages following the wiki's
own SCHEMA.md. A separate lint mode checks the wiki's health weekly.

Design mirrors the Doctor: single-turn LLM calls returning strict JSON, budget
enforced by LLMClient, dependency-injected client for tests.

Ingest is incremental: raw files are append-only, so the state file records how
many lines of each raw file have been processed and only the delta is sent.

Two passes per ingest run:
  1. SELECT — given the index and the pending notes, the model names the wiki
     pages it needs to read (bounded).
  2. EDIT — given those pages in full plus the notes, the model returns full
     new contents for the pages it wants to create/update, an updated index,
     and a one-line log entry.

The agent only ever writes under wiki/ (plus index.md and a log.md line):
raw/ and SCHEMA.md are untouchable by construction.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from .llm import LLMClient, TokenUsage


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_PAGES_TO_READ = 8
MAX_EDITS = 12
STATE_FILE = ".ingest-state.json"


def default_wiki_root() -> Path:
    env = os.getenv("LLMWIKI_ROOT")
    if env:
        return Path(env)
    local = os.getenv("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "NINA" / "llmwiki"


@dataclass
class IngestReport:
    raw_files_processed: int = 0
    new_lines: int = 0
    pages_written: list[str] = field(default_factory=list)
    log_line: Optional[str] = None
    skipped_reason: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)


# ── state ────────────────────────────────────────────────────────────────────

def _load_state(root: Path) -> dict[str, int]:
    path = root / STATE_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(root: Path, state: dict[str, int]) -> None:
    (root / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _collect_pending(root: Path, state: dict[str, int]) -> list[tuple[str, list[str]]]:
    """Returns (relpath, new_lines) for every raw file with unprocessed lines."""
    pending: list[tuple[str, list[str]]] = []
    raw_dir = root / "raw"
    if not raw_dir.is_dir():
        return pending
    for file in sorted(raw_dir.glob("*.md")):
        if file.name.lower() == "readme.md":
            continue
        rel = f"raw/{file.name}"
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        done = state.get(rel, 0)
        if len(lines) > done:
            new = [l for l in lines[done:] if l.strip()]
            if new:
                pending.append((rel, new))
    return pending


def _page_catalog(root: Path) -> str:
    entries = []
    for file in sorted((root / "wiki").rglob("*.md")):
        rel = file.relative_to(root).as_posix()
        title = ""
        for line in file.read_text(encoding="utf-8", errors="replace").splitlines()[:12]:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        entries.append(f"- {rel}" + (f" — {title}" if title else ""))
    return "\n".join(entries) if entries else "(no wiki pages yet)"


def _extract_json(text: str) -> dict:
    """Parses the first JSON object in the model output, tolerating fences."""
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


# ── ingest ───────────────────────────────────────────────────────────────────

_SELECT_SYSTEM = """You are the maintainer of a personal astrophotography knowledge wiki.
You will receive: the wiki index, a catalog of existing pages, and NEW raw observations
(daily weather digests, user-confirmed notes) that must be consolidated into the wiki.

Decide which existing pages you need to read in full before editing. Pick only pages
plausibly affected by the new observations (the entity pages of mentioned hardware or
sites, concept pages of mentioned problems).

OUTPUT (strict): a single JSON object, no prose:
  {"read": ["wiki/entities/foo.md", ...]}
Empty list is valid if only new pages will be created."""


_EDIT_SYSTEM_TEMPLATE = """You are the maintainer of a personal astrophotography knowledge wiki.
You consolidate NEW raw observations into wiki pages, following the wiki's schema below.

THE SCHEMA (from the wiki's SCHEMA.md):
{schema}

HARD RULES:
- You may only create/update pages under wiki/ (entities, concepts, syntheses) and
  replace index.md. Never touch raw/ or SCHEMA.md.
- Every page you write must keep the frontmatter format (title/type/created/updated/tags),
  a one-line TL;DR blockquote, and cite raw file names for claims taken from observations.
- Update, do not duplicate: if an entity page for the hardware/site exists, extend it.
- Preserve existing correct content when updating a page: you receive the current
  content, return the full new content.
- Contradictions: keep both claims with citations and add the tag #to-resolve.
- Only consolidate what the observations actually support. It is fine to write nothing:
  return an empty edits list if the notes carry no durable knowledge.

OUTPUT (strict): a single JSON object, no prose:
{{
  "edits": [{{"path": "wiki/entities/<slug>.md", "content": "<full file content>"}}],
  "index": "<full new index.md content, or null if unchanged>",
  "log_line": "<one line describing what was consolidated>"
}}"""


async def run_ingest(
    llm: LLMClient,
    *,
    root: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> IngestReport:
    root = root or default_wiki_root()
    report = IngestReport()

    schema_path = root / "SCHEMA.md"
    index_path = root / "index.md"
    if not schema_path.is_file() or not index_path.is_file():
        report.skipped_reason = f"wiki not initialized at {root} (SCHEMA.md/index.md missing)"
        return report

    state = _load_state(root)
    pending = _collect_pending(root, state)
    if not pending:
        report.skipped_reason = "no new raw observations"
        return report

    report.raw_files_processed = len(pending)
    report.new_lines = sum(len(lines) for _, lines in pending)

    notes_block = "\n\n".join(
        f"### {rel} (new entries)\n" + "\n".join(lines) for rel, lines in pending
    )
    schema = schema_path.read_text(encoding="utf-8", errors="replace")
    index = index_path.read_text(encoding="utf-8", errors="replace")
    catalog = _page_catalog(root)

    # Pass 1 — page selection
    select = await llm.complete(
        model=model,
        system=_SELECT_SYSTEM,
        user=f"INDEX:\n{index}\n\nPAGE CATALOG:\n{catalog}\n\nNEW OBSERVATIONS:\n{notes_block}",
        max_tokens=1024,
        cache_system=False,
    )
    report.usage.add(select.usage)
    try:
        to_read = _extract_json(select.text).get("read", [])[:MAX_PAGES_TO_READ]
    except (ValueError, AttributeError):
        logger.warning("wiki-ingest: page-selection output unparseable; proceeding with no pages")
        to_read = []

    pages_block = ""
    for rel in to_read:
        file = (root / rel).resolve()
        if root.resolve() not in file.parents or not rel.startswith("wiki/") or not file.is_file():
            continue
        content = file.read_text(encoding="utf-8", errors="replace")
        pages_block += f"\n\n### CURRENT CONTENT OF {rel}\n{content}"

    # Pass 2 — edits
    edit = await llm.complete(
        model=model,
        system=_EDIT_SYSTEM_TEMPLATE.format(schema=schema),
        user=f"TODAY: {date.today().isoformat()}\n\nINDEX:\n{index}\n\nPAGE CATALOG:\n{catalog}\n\nNEW OBSERVATIONS:\n{notes_block}{pages_block}",
        max_tokens=16000,
    )
    report.usage.add(edit.usage)

    try:
        result = _extract_json(edit.text)
    except (ValueError, AttributeError) as ex:
        report.skipped_reason = f"edit output unparseable: {ex}"
        return report

    edits = result.get("edits") or []
    if len(edits) > MAX_EDITS:
        logger.warning("wiki-ingest: %d edits returned, capping at %d", len(edits), MAX_EDITS)
        edits = edits[:MAX_EDITS]

    for item in edits:
        rel = str(item.get("path", ""))
        content = item.get("content")
        target = (root / rel).resolve()
        valid = (
            rel.startswith("wiki/")
            and rel.endswith(".md")
            and root.resolve() in target.parents
            and isinstance(content, str)
            and content.lstrip().startswith("---")
        )
        if not valid:
            logger.warning("wiki-ingest: rejected edit to %r", rel)
            continue
        if dry_run:
            # Save the proposal where the human can read it before a real run.
            preview = root / ".ingest-preview" / rel
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_text(content, encoding="utf-8")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        report.pages_written.append(rel)

    new_index = result.get("index")
    if isinstance(new_index, str) and new_index.strip().startswith("#"):
        if dry_run:
            preview = root / ".ingest-preview" / "index.md"
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_text(new_index, encoding="utf-8")
        else:
            index_path.write_text(new_index, encoding="utf-8")
        report.pages_written.append("index.md")

    log_line = result.get("log_line")
    if isinstance(log_line, str) and log_line.strip():
        report.log_line = log_line.strip()
        if not dry_run:
            with open(root / "log.md", "a", encoding="utf-8") as fh:
                fh.write(f"- {date.today().isoformat()} — [ingest] {report.log_line}\n")

    if not dry_run:
        for rel, _ in pending:
            full = root / rel
            state[rel] = len(full.read_text(encoding="utf-8", errors="replace").splitlines())
        _save_state(root, state)

    return report


# ── lint ─────────────────────────────────────────────────────────────────────

_LINT_SYSTEM = """You review a personal astrophotography knowledge wiki for internal
consistency. You receive every wiki page. Report ONLY real problems:
  - contradictions between pages (or inside one page) about the same fact
  - claims about hardware behavior with no citation to a raw/ observation
  - pages whose content does not match their frontmatter type

OUTPUT (strict): a single JSON object, no prose:
  {"issues": [{"page": "wiki/...", "issue": "<one line>"}]}
Empty list means the wiki is healthy."""


@dataclass
class LintReport:
    broken_links: list[str] = field(default_factory=list)
    missing_frontmatter: list[str] = field(default_factory=list)
    llm_issues: list[dict] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


async def run_lint(
    llm: Optional[LLMClient],
    *,
    root: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
) -> LintReport:
    root = root or default_wiki_root()
    report = LintReport()

    pages = sorted((root / "wiki").rglob("*.md")) if (root / "wiki").is_dir() else []
    slugs = {p.stem.lower() for p in pages}

    corpus = ""
    for page in pages:
        rel = page.relative_to(root).as_posix()
        content = page.read_text(encoding="utf-8", errors="replace")
        if not content.lstrip().startswith("---"):
            report.missing_frontmatter.append(rel)
        for link in re.findall(r"\[\[([^\]|#]+)", content):
            slug = link.strip().split("/")[-1].lower().removesuffix(".md")
            if slug and slug not in slugs and slug not in {"schema", "log", "index"}:
                report.broken_links.append(f"{rel} -> [[{link.strip()}]]")
        corpus += f"\n\n### {rel}\n{content}"

    if llm is not None and corpus:
        result = await llm.complete(
            model=model,
            system=_LINT_SYSTEM,
            user=corpus[:180_000],
            max_tokens=4096,
            cache_system=False,
        )
        report.usage.add(result.usage)
        try:
            report.llm_issues = _extract_json(result.text).get("issues", [])
        except (ValueError, AttributeError):
            logger.warning("wiki-lint: LLM output unparseable; mechanical checks only")

    # Persist the report as a synthesis page (overwritten each run).
    lines = [
        "---",
        "title: Wiki lint report",
        "type: synthesis",
        f"created: {date.today().isoformat()}",
        f"updated: {date.today().isoformat()}",
        "tags: [lint, maintenance]",
        "---",
        "",
        "# Wiki lint report",
        "",
        f"> Automated health check of {date.today().isoformat()}.",
        "",
    ]
    if not (report.broken_links or report.missing_frontmatter or report.llm_issues):
        lines.append("No issues found.")
    else:
        if report.broken_links:
            lines.append("## Broken wiki-links")
            lines += [f"- {b}" for b in report.broken_links]
        if report.missing_frontmatter:
            lines.append("\n## Pages missing frontmatter")
            lines += [f"- {p}" for p in report.missing_frontmatter]
        if report.llm_issues:
            lines.append("\n## Consistency issues")
            lines += [f"- {i.get('page', '?')}: {i.get('issue', '?')}" for i in report.llm_issues]

    target = root / "wiki" / "syntheses" / "lint-report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(root / "log.md", "a", encoding="utf-8") as fh:
        issues = len(report.broken_links) + len(report.missing_frontmatter) + len(report.llm_issues)
        fh.write(f"- {date.today().isoformat()} — [lint] {issues} issue(s), report in wiki/syntheses/lint-report.md\n")

    return report
