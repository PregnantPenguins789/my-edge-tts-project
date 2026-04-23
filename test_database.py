"""
test_database.py
Smoke test — runs against a temp DB, prints results, leaves no side effects.
Run from project root: python test_database.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Point DB at a temp file for this test
import core.database as db

tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
db.DB_PATH = Path(tmp.name)

db.init_db()
print("✓ Schema initialized")

# Project
project = db.create_project("test_project", "en-US-AriaNeural")
assert project.project_id is not None
print(f"✓ Project created: {project}")

# Source text
src = db.add_source_text(project.project_id, "data/texts/test.txt")
print(f"✓ Source text added: {src}")

# Render
render = db.add_render(
    project.project_id,
    audio_path="data/audio/render_001.wav",
    duration_ms=4200,
    sample_rate=22050,
    word_boundaries=[{"word": "hello", "start_ms": 0, "end_ms": 410}]
)
assert render.render_id is not None
assert render.word_boundaries()[0]["word"] == "hello"
print(f"✓ Render added: {render}")

# Region
region = db.create_region(
    render.render_id,
    start_ms=800,
    end_ms=1400,
    context_before_ms=200,
    context_after_ms=200,
    text_fragment="trails off here"
)
assert region.duration_ms == 600
assert region.context_start_ms == 600
print(f"✓ Region created: {region}")

# Attempt
attempt = db.add_attempt(
    region.region_id,
    strategy_name="ssml_nudge",
    candidate_audio_path="data/audio/attempt_001.wav",
    ssml_params={"pitch": "+10%", "rate": "-5%"},
    features={"mean_f0": 185.3, "f0_std": 12.1, "duration_ms": 600}
)
assert attempt.ssml_params()["pitch"] == "+10%"
print(f"✓ Attempt added: {attempt}")

# Decision — reject
rejection = db.record_decision(attempt.attempt_id, accepted=False, notes="still flat")
assert not rejection.accepted
print(f"✓ Rejection recorded: {rejection}")

# Second attempt
attempt2 = db.add_attempt(
    region.region_id,
    strategy_name="ssml_nudge",
    candidate_audio_path="data/audio/attempt_002.wav",
    ssml_params={"pitch": "+20%", "rate": "-5%"},
    features={"mean_f0": 205.0, "f0_std": 18.4, "duration_ms": 580}
)

# Decision — accept
acceptance = db.record_decision(attempt2.attempt_id, accepted=True, notes="much better")
assert acceptance.accepted
print(f"✓ Acceptance recorded: {acceptance}")

# Tags
db.tag_decision(acceptance.decision_id, "trails off")
db.tag_decision(acceptance.decision_id, "pitch low")
tags = db.get_tags_for_decision(acceptance.decision_id)
assert len(tags) == 2
assert {t.name for t in tags} == {"trails off", "pitch low"}
print(f"✓ Tags applied: {[t.name for t in tags]}")

# Tag idempotency
db.tag_decision(acceptance.decision_id, "trails off")
tags2 = db.get_tags_for_decision(acceptance.decision_id)
assert len(tags2) == 2
print("✓ Tag idempotency confirmed")

# Splice
splice = db.record_splice(region.region_id, attempt2.attempt_id, "data/audio/spliced_001.wav")
print(f"✓ Splice recorded: {splice}")

# Dataset extraction
pairs = db.get_accepted_pairs()
assert len(pairs) == 1
pair = pairs[0]
assert pair["tags"] == ["pitch low", "trails off"] or set(pair["tags"]) == {"trails off", "pitch low"}
assert pair["features"]["mean_f0"] == 205.0
print(f"✓ Dataset extraction: {len(pairs)} accepted pair(s)")
print(f"  tags: {pair['tags']}")
print(f"  features: {pair['features']}")

# List all tags in vocabulary
all_tags = db.list_all_tags()
print(f"✓ Tag vocabulary: {[t.name for t in all_tags]}")

# Cleanup
os.unlink(tmp.name)
print("\n✓ All checks passed. DB layer is clean.")
