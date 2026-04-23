"""
database.py
Connection management, schema initialization, and CRUD helpers.
Returns dataclass instances. No SQL logic outside this module.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from core.models import (
    Project, SourceText, Render, Region,
    Attempt, Decision, Tag, Splice
)

import os as _os
DB_PATH = Path(_os.environ.get("TTS_TOOL_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))) / "data" / "tts_tool.db"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    voice_name   TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS source_texts (
    source_text_id  INTEGER PRIMARY KEY,
    project_id      INTEGER NOT NULL REFERENCES projects(project_id),
    text_path       TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS renders (
    render_id    INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL REFERENCES projects(project_id),
    audio_path   TEXT NOT NULL,
    duration_ms  INTEGER,
    sample_rate  INTEGER,
    word_boundaries_json TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS regions (
    region_id         INTEGER PRIMARY KEY,
    render_id         INTEGER NOT NULL REFERENCES renders(render_id),
    start_ms          INTEGER NOT NULL,
    end_ms            INTEGER NOT NULL,
    context_before_ms INTEGER NOT NULL DEFAULT 200,
    context_after_ms  INTEGER NOT NULL DEFAULT 200,
    text_fragment     TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id           INTEGER PRIMARY KEY,
    region_id            INTEGER NOT NULL REFERENCES regions(region_id),
    strategy_name        TEXT NOT NULL,
    ssml_params_json     TEXT,
    dsp_params_json      TEXT,
    candidate_audio_path TEXT,
    features_json        TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id   INTEGER PRIMARY KEY,
    attempt_id    INTEGER NOT NULL REFERENCES attempts(attempt_id),
    accepted_bool INTEGER NOT NULL DEFAULT 0,
    notes         TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tags (
    tag_id  INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS decision_tags (
    decision_id INTEGER NOT NULL REFERENCES decisions(decision_id),
    tag_id      INTEGER NOT NULL REFERENCES tags(tag_id),
    PRIMARY KEY (decision_id, tag_id)
);

CREATE TABLE IF NOT EXISTS splices (
    splice_id         INTEGER PRIMARY KEY,
    region_id         INTEGER NOT NULL REFERENCES regions(region_id),
    attempt_id        INTEGER NOT NULL REFERENCES attempts(attempt_id),
    result_audio_path TEXT NOT NULL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_regions_render     ON regions(render_id);
CREATE INDEX IF NOT EXISTS idx_attempts_region    ON attempts(region_id);
CREATE INDEX IF NOT EXISTS idx_decisions_attempt  ON decisions(attempt_id);
CREATE INDEX IF NOT EXISTS idx_decision_tags_dec  ON decision_tags(decision_id);
CREATE INDEX IF NOT EXISTS idx_splices_region     ON splices(region_id);
"""


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Transaction helper
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def create_project(name: str, voice_name: str) -> Project:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO projects (name, voice_name) VALUES (?, ?)",
            (name, voice_name)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM projects WHERE project_id = ?", (cur.lastrowid,)
        ).fetchone()
    return Project(**_row_to_dict(row))


def get_project(project_id: int) -> Optional[Project]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone()
    return Project(**_row_to_dict(row)) if row else None


def list_projects() -> list[Project]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()
    return [Project(**_row_to_dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Source texts
# ---------------------------------------------------------------------------

def add_source_text(project_id: int, text_path: str) -> SourceText:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO source_texts (project_id, text_path) VALUES (?, ?)",
            (project_id, text_path)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM source_texts WHERE source_text_id = ?", (cur.lastrowid,)
        ).fetchone()
    return SourceText(**_row_to_dict(row))


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------

def add_render(
    project_id: int,
    audio_path: str,
    duration_ms: Optional[int] = None,
    sample_rate: Optional[int] = None,
    word_boundaries: Optional[list] = None,
) -> Render:
    wb_json = json.dumps(word_boundaries) if word_boundaries else None
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO renders
               (project_id, audio_path, duration_ms, sample_rate, word_boundaries_json)
               VALUES (?, ?, ?, ?, ?)""",
            (project_id, audio_path, duration_ms, sample_rate, wb_json)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM renders WHERE render_id = ?", (cur.lastrowid,)
        ).fetchone()
    return Render(**_row_to_dict(row))


def get_renders_for_project(project_id: int) -> list[Render]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM renders WHERE project_id = ? ORDER BY created_at",
            (project_id,)
        ).fetchall()
    return [Render(**_row_to_dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

def create_region(
    render_id: int,
    start_ms: int,
    end_ms: int,
    context_before_ms: int = 200,
    context_after_ms: int = 200,
    text_fragment: Optional[str] = None,
) -> Region:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO regions
               (render_id, start_ms, end_ms, context_before_ms, context_after_ms, text_fragment)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (render_id, start_ms, end_ms, context_before_ms, context_after_ms, text_fragment)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM regions WHERE region_id = ?", (cur.lastrowid,)
        ).fetchone()
    return Region(**_row_to_dict(row))


def get_regions_for_render(render_id: int) -> list[Region]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM regions WHERE render_id = ? ORDER BY start_ms",
            (render_id,)
        ).fetchall()
    return [Region(**_row_to_dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------

def add_attempt(
    region_id: int,
    strategy_name: str,
    candidate_audio_path: Optional[str] = None,
    ssml_params: Optional[dict] = None,
    dsp_params: Optional[dict] = None,
    features: Optional[dict] = None,
) -> Attempt:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO attempts
               (region_id, strategy_name, ssml_params_json, dsp_params_json,
                candidate_audio_path, features_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                region_id,
                strategy_name,
                json.dumps(ssml_params) if ssml_params else None,
                json.dumps(dsp_params) if dsp_params else None,
                candidate_audio_path,
                json.dumps(features) if features else None,
            )
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (cur.lastrowid,)
        ).fetchone()
    return Attempt(**_row_to_dict(row))


def get_attempts_for_region(region_id: int) -> list[Attempt]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM attempts WHERE region_id = ? ORDER BY created_at",
            (region_id,)
        ).fetchall()
    return [Attempt(**_row_to_dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def record_decision(
    attempt_id: int,
    accepted: bool,
    notes: Optional[str] = None,
) -> Decision:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO decisions (attempt_id, accepted_bool, notes) VALUES (?, ?, ?)",
            (attempt_id, int(accepted), notes)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM decisions WHERE decision_id = ?", (cur.lastrowid,)
        ).fetchone()
    return Decision(**_row_to_dict(row))


def get_decision_for_attempt(attempt_id: int) -> Optional[Decision]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM decisions WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
    return Decision(**_row_to_dict(row)) if row else None


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def get_or_create_tag(name: str) -> Tag:
    name = name.strip().lower()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM tags WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return Tag(**_row_to_dict(row))
        cur = conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
        conn.commit()
        row = conn.execute(
            "SELECT * FROM tags WHERE tag_id = ?", (cur.lastrowid,)
        ).fetchone()
    return Tag(**_row_to_dict(row))


def tag_decision(decision_id: int, tag_name: str) -> None:
    tag = get_or_create_tag(tag_name)
    with get_connection() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO decision_tags (decision_id, tag_id)
               VALUES (?, ?)""",
            (decision_id, tag.tag_id)
        )
        conn.commit()


def get_tags_for_decision(decision_id: int) -> list[Tag]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT t.* FROM tags t
               JOIN decision_tags dt ON dt.tag_id = t.tag_id
               WHERE dt.decision_id = ?""",
            (decision_id,)
        ).fetchall()
    return [Tag(**_row_to_dict(r)) for r in rows]


def list_all_tags() -> list[Tag]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM tags ORDER BY name").fetchall()
    return [Tag(**_row_to_dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Splices
# ---------------------------------------------------------------------------

def record_splice(
    region_id: int,
    attempt_id: int,
    result_audio_path: str,
) -> Splice:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO splices (region_id, attempt_id, result_audio_path)
               VALUES (?, ?, ?)""",
            (region_id, attempt_id, result_audio_path)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM splices WHERE splice_id = ?", (cur.lastrowid,)
        ).fetchone()
    return Splice(**_row_to_dict(row))


def get_splices_for_region(region_id: int) -> list[Splice]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM splices WHERE region_id = ? ORDER BY created_at",
            (region_id,)
        ).fetchall()
    return [Splice(**_row_to_dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------

def get_accepted_pairs() -> list[dict]:
    """
    Returns all accepted (region, attempt, decision, tags) tuples.
    This is the training dataset view.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT
                 r.region_id,
                 r.render_id,
                 r.start_ms,
                 r.end_ms,
                 r.context_before_ms,
                 r.context_after_ms,
                 r.text_fragment,
                 a.attempt_id,
                 a.strategy_name,
                 a.ssml_params_json,
                 a.dsp_params_json,
                 a.candidate_audio_path,
                 a.features_json,
                 d.decision_id,
                 d.notes
               FROM decisions d
               JOIN attempts a  ON a.attempt_id  = d.attempt_id
               JOIN regions r   ON r.region_id   = a.region_id
               WHERE d.accepted_bool = 1
               ORDER BY r.render_id, r.start_ms"""
        ).fetchall()

    if not rows:
        return []

    decision_ids = [dict(row)["decision_id"] for row in rows]
    placeholders = ",".join("?" * len(decision_ids))
    with get_connection() as conn:
        tag_rows = conn.execute(
            f"""SELECT dt.decision_id, t.name FROM tags t
               JOIN decision_tags dt ON dt.tag_id = t.tag_id
               WHERE dt.decision_id IN ({placeholders})""",
            decision_ids,
        ).fetchall()

    tags_by_decision: dict[int, list[str]] = {did: [] for did in decision_ids}
    for tr in tag_rows:
        tags_by_decision[tr["decision_id"]].append(tr["name"])

    results = []
    for row in rows:
        entry = dict(row)
        entry["ssml_params"] = json.loads(entry.pop("ssml_params_json") or "{}")
        entry["dsp_params"]  = json.loads(entry.pop("dsp_params_json")  or "{}")
        entry["features"]    = json.loads(entry.pop("features_json")    or "{}")
        entry["tags"]        = tags_by_decision.get(entry["decision_id"], [])
        results.append(entry)

    return results
