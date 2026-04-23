"""
strategies/ssml_strategy.py

SSML prosody nudging strategy.
Generates correction candidates by adjusting edge-tts prosody parameters.

The strategy operates in a parameter space:
    rate:   speaking rate  (+/-% or named: x-slow slow medium fast x-fast)
    pitch:  pitch shift    (+/-Hz or +/-%)
    volume: loudness       (+/-%)

Each call to generate_candidate() returns a CandidateResult.
The strategy also supports:
    - parameter grid search (try multiple param combinations)
    - history-aware nudging (avoid params already tried)
    - seeding from prior accepted attempts for similar text
"""

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator

from tts.edge_wrapper import EdgeTTSWrapper, RenderResult


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

# Default nudge steps for iterative correction
RATE_STEPS   = ["-10%", "-5%", "0%", "+5%", "+10%", "+15%", "+20%"]
PITCH_STEPS  = ["-20Hz", "-10Hz", "-5Hz", "0Hz", "+5Hz", "+10Hz", "+20Hz", "+30Hz"]
VOLUME_STEPS = ["-10%", "0%", "+10%"]

# Minimal starting point — no change
DEFAULT_PARAMS = {"rate": "0%", "pitch": "0Hz", "volume": "0%"}


@dataclass
class CandidateResult:
    audio_path:   str
    ssml_params:  dict
    strategy:     str = "ssml_nudge"
    render_result: Optional[RenderResult] = None


# ---------------------------------------------------------------------------
# SSML Strategy
# ---------------------------------------------------------------------------

class SSMLStrategy:
    """
    Iterative SSML prosody correction strategy.

    Usage:
        strategy = SSMLStrategy(wrapper, text_fragment)
        candidate = await strategy.next_candidate()
        # human evaluates
        strategy.record_rejection(candidate.ssml_params)
        candidate2 = await strategy.next_candidate(hint={"pitch": "+10Hz"})
    """

    name = "ssml_nudge"

    def __init__(
        self,
        wrapper: EdgeTTSWrapper,
        text_fragment: str,
        seed_params: Optional[dict] = None,
    ):
        self.wrapper       = wrapper
        self.text_fragment = text_fragment
        self.seed_params   = seed_params or dict(DEFAULT_PARAMS)
        self._tried:  list[dict] = []
        self._attempt_n = 0

    async def next_candidate(
        self,
        hint: Optional[dict] = None,
    ) -> CandidateResult:
        """
        Generate the next candidate audio.

        hint: partial param dict to try, e.g. {"pitch": "+10Hz"}
              merged with seed_params.
        """
        self._attempt_n += 1
        params = dict(self.seed_params)
        if hint:
            params.update(hint)

        # Skip params already tried
        if params in self._tried:
            params = self._mutate(params)

        self._tried.append(dict(params))

        result = await self.wrapper.render_region(
            self.text_fragment,
            ssml_params=_clean_params(params),
            attempt_number=self._attempt_n,
        )

        return CandidateResult(
            audio_path=result.audio_path,
            ssml_params=params,
            strategy=self.name,
            render_result=result,
        )

    def record_rejection(self, params: dict) -> None:
        """Mark params as rejected. Nudges will avoid this region."""
        if params not in self._tried:
            self._tried.append(params)

    def record_acceptance(self, params: dict) -> None:
        """Store accepted params as new seed for future nudges."""
        self.seed_params = dict(params)

    async def grid_search(
        self,
        rate_options:   Optional[list] = None,
        pitch_options:  Optional[list] = None,
        volume_options: Optional[list] = None,
    ) -> AsyncIterator[CandidateResult]:
        """
        Async generator yielding candidates for every param combination.
        Useful for batch pre-generation before human review.

        Usage:
            async for candidate in strategy.grid_search(pitch_options=["-10Hz","0Hz","+10Hz"]):
                store(candidate)
        """
        rates   = rate_options   or [self.seed_params.get("rate",   "0%")]
        pitches = pitch_options  or [self.seed_params.get("pitch",  "0Hz")]
        volumes = volume_options or [self.seed_params.get("volume", "0%")]

        for rate, pitch, volume in itertools.product(rates, pitches, volumes):
            params = {"rate": rate, "pitch": pitch, "volume": volume}
            if params in self._tried:
                continue
            self._tried.append(dict(params))
            self._attempt_n += 1

            result = await self.wrapper.render_region(
                self.text_fragment,
                ssml_params=_clean_params(params),
                attempt_number=self._attempt_n,
            )
            yield CandidateResult(
                audio_path=result.audio_path,
                ssml_params=params,
                strategy=self.name,
                render_result=result,
            )

    def _mutate(self, params: dict) -> dict:
        """
        Small deterministic nudge when requested params were already tried.
        Increments pitch by one step.
        """
        current_pitch = params.get("pitch", "0Hz")
        idx = PITCH_STEPS.index(current_pitch) if current_pitch in PITCH_STEPS else 3
        next_idx = (idx + 1) % len(PITCH_STEPS)
        return dict(params, pitch=PITCH_STEPS[next_idx])

    @property
    def attempts_made(self) -> int:
        return self._attempt_n

    @property
    def tried_params(self) -> list[dict]:
        return list(self._tried)


# ---------------------------------------------------------------------------
# DSP strategy stub
# ---------------------------------------------------------------------------

class DSPStrategy:
    """
    Placeholder for DSP-based correction (pitch shift, time stretch).
    Satisfies the plugin interface but raises NotImplementedError until implemented.
    """
    name = "dsp_repair"

    def __init__(self, wrapper: EdgeTTSWrapper, text_fragment: str,
                 seed_params: Optional[dict] = None):
        self.wrapper       = wrapper
        self.text_fragment = text_fragment
        self.seed_params   = seed_params or {}

    async def next_candidate(self, hint: Optional[dict] = None) -> CandidateResult:
        raise NotImplementedError(
            "DSP strategy is not yet implemented. Use SSMLStrategy."
        )


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    SSMLStrategy.name: SSMLStrategy,
    DSPStrategy.name:  DSPStrategy,
}


def register_strategy(name: str, cls: type) -> None:
    """Register a custom correction strategy class."""
    _REGISTRY[name] = cls


def get_strategy(
    name: str,
    wrapper: EdgeTTSWrapper,
    text_fragment: str,
    seed_params: Optional[dict] = None,
) -> SSMLStrategy:
    """
    Instantiate a strategy by name.
    Raises KeyError if name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](wrapper, text_fragment, seed_params)


def list_strategies() -> list[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _clean_params(params: dict) -> dict:
    """
    Remove no-op values before passing to edge-tts.
    e.g. pitch="0Hz" and rate="0%" are no-ops — omitting them
    avoids unnecessary SSML wrapping.
    """
    out = {}
    if params.get("rate") not in (None, "0%", "medium"):
        out["rate"] = params["rate"]
    if params.get("pitch") not in (None, "0Hz", "0%"):
        out["pitch"] = params["pitch"]
    if params.get("volume") not in (None, "0%"):
        out["volume"] = params["volume"]
    return out


def params_summary(params: dict) -> str:
    """Human-readable one-line summary of SSML params for TUI display."""
    parts = []
    for k in ("pitch", "rate", "volume"):
        v = params.get(k)
        if v and v not in ("0Hz", "0%", "medium"):
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "no change"
