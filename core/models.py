"""
models.py
Dataclass representations of database rows.
No SQL logic here. No ORM. Just typed containers.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class Project:
    project_id: int
    name: str
    voice_name: str
    created_at: Optional[str] = None


@dataclass
class SourceText:
    source_text_id: int
    project_id: int
    text_path: str
    created_at: Optional[str] = None


@dataclass
class Render:
    render_id: int
    project_id: int
    audio_path: str
    duration_ms: Optional[int] = None
    sample_rate: Optional[int] = None
    word_boundaries_json: Optional[str] = None
    created_at: Optional[str] = None

    def word_boundaries(self) -> list:
        if self.word_boundaries_json:
            return json.loads(self.word_boundaries_json)
        return []


@dataclass
class Region:
    region_id: int
    render_id: int
    start_ms: int
    end_ms: int
    context_before_ms: int = 200
    context_after_ms: int = 200
    text_fragment: Optional[str] = None
    created_at: Optional[str] = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def context_start_ms(self) -> int:
        return max(0, self.start_ms - self.context_before_ms)

    @property
    def context_end_ms(self) -> int:
        return self.end_ms + self.context_after_ms


@dataclass
class Attempt:
    attempt_id: int
    region_id: int
    strategy_name: str
    ssml_params_json: Optional[str] = None
    dsp_params_json: Optional[str] = None
    candidate_audio_path: Optional[str] = None
    features_json: Optional[str] = None
    created_at: Optional[str] = None

    def ssml_params(self) -> dict:
        return json.loads(self.ssml_params_json) if self.ssml_params_json else {}

    def dsp_params(self) -> dict:
        return json.loads(self.dsp_params_json) if self.dsp_params_json else {}

    def features(self) -> dict:
        return json.loads(self.features_json) if self.features_json else {}


@dataclass
class Decision:
    decision_id: int
    attempt_id: int
    accepted_bool: int
    notes: Optional[str] = None
    created_at: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return bool(self.accepted_bool)


@dataclass
class Tag:
    tag_id: int
    name: str


@dataclass
class DecisionTag:
    decision_id: int
    tag_id: int


@dataclass
class Splice:
    splice_id: int
    region_id: int
    attempt_id: int
    result_audio_path: str
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Typed container for dataset extraction output
# ---------------------------------------------------------------------------

@dataclass
class AcceptedPair:
    """
    One supervised training example.
    Input: original region audio (sourced from render via start/end_ms)
    Output: candidate_audio_path (the accepted replacement)
    """
    region_id: int
    render_id: int
    start_ms: int
    end_ms: int
    context_before_ms: int
    context_after_ms: int
    text_fragment: Optional[str]
    attempt_id: int
    strategy_name: str
    ssml_params: dict
    dsp_params: dict
    candidate_audio_path: Optional[str]
    features: dict
    decision_id: int
    notes: Optional[str]
    tags: list[str] = field(default_factory=list)
