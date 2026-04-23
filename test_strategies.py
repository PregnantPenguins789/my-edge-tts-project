"""
test_strategies.py
Tests for strategies/ssml_strategy.py using MockWrapper.
Run from project root: python test_strategies.py
"""

import sys
import math
import wave
import struct
import asyncio
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tts.edge_wrapper import EdgeTTSWrapper, RenderResult, WordBoundary
from strategies.ssml_strategy import (
    SSMLStrategy, DSPStrategy,
    get_strategy, list_strategies, register_strategy,
    _clean_params, params_summary,
    DEFAULT_PARAMS,
)


# ---------------------------------------------------------------------------
# MockWrapper
# ---------------------------------------------------------------------------

def _write_silent_wav(path, duration_ms=300, sample_rate=22050):
    n = int(sample_rate * duration_ms / 1000)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


class MockWrapper(EdgeTTSWrapper):
    def __init__(self, tmpdir):
        self.voice      = "en-US-AriaNeural"
        self.output_dir = Path(tmpdir)
        self._call_log: list[dict] = []

    async def render_region(self, text, ssml_params=None, attempt_number=1):
        params = ssml_params or {}
        from tts.edge_wrapper import _deterministic_filename
        fname = _deterministic_filename(text, params, prefix=f"candidate_a{attempt_number:03d}")
        path  = self.output_dir / fname
        _write_silent_wav(str(path))
        self._call_log.append({"text": text, "params": params, "attempt": attempt_number})
        return RenderResult(
            audio_path=str(path),
            mp3_path=str(path).replace('.wav', '.mp3'),
            duration_ms=300,
            sample_rate=22050,
            word_boundaries=[],
            voice=self.voice,
            text=text,
            ssml_params=params,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_basic_candidate():
    with tempfile.TemporaryDirectory() as d:
        wrapper  = MockWrapper(d)
        strategy = SSMLStrategy(wrapper, "trails off here")
        c = await strategy.next_candidate()
        assert c.audio_path.endswith(".wav")
        assert os.path.exists(c.audio_path)
        assert c.strategy == "ssml_nudge"
        assert strategy.attempts_made == 1
    print(f"✓ basic candidate: path={os.path.basename(c.audio_path)}, params={c.ssml_params}")


async def test_hint_applied():
    with tempfile.TemporaryDirectory() as d:
        wrapper  = MockWrapper(d)
        strategy = SSMLStrategy(wrapper, "rises here")
        c = await strategy.next_candidate(hint={"pitch": "+10Hz"})
        assert c.ssml_params.get("pitch") == "+10Hz"
    print(f"✓ hint applied: {c.ssml_params}")


async def test_no_duplicate_params():
    """Same params tried twice should trigger a mutation on second call."""
    with tempfile.TemporaryDirectory() as d:
        wrapper  = MockWrapper(d)
        strategy = SSMLStrategy(wrapper, "test text", seed_params={"pitch": "0Hz", "rate": "0%", "volume": "0%"})
        c1 = await strategy.next_candidate()
        c2 = await strategy.next_candidate()
        # Second attempt must differ from first
        assert c1.ssml_params != c2.ssml_params, \
            f"Expected different params, got same twice: {c1.ssml_params}"
    print(f"✓ no duplicates: {c1.ssml_params} → {c2.ssml_params}")


async def test_record_rejection_avoids_params():
    with tempfile.TemporaryDirectory() as d:
        wrapper  = MockWrapper(d)
        strategy = SSMLStrategy(wrapper, "test")
        rejected = {"rate": "+10%", "pitch": "0Hz", "volume": "0%"}
        strategy.record_rejection(rejected)
        assert rejected in strategy.tried_params
    print("✓ record_rejection registers params as tried")


async def test_record_acceptance_updates_seed():
    with tempfile.TemporaryDirectory() as d:
        wrapper  = MockWrapper(d)
        strategy = SSMLStrategy(wrapper, "test")
        accepted = {"rate": "-5%", "pitch": "+10Hz", "volume": "0%"}
        strategy.record_acceptance(accepted)
        assert strategy.seed_params == accepted
    print("✓ record_acceptance updates seed params")


async def test_grid_search():
    with tempfile.TemporaryDirectory() as d:
        wrapper  = MockWrapper(d)
        strategy = SSMLStrategy(wrapper, "grid test")
        pitches  = ["-10Hz", "0Hz", "+10Hz"]
        results  = []
        async for candidate in strategy.grid_search(pitch_options=pitches):
            results.append(candidate)
        assert len(results) == 3
        found_pitches = {c.ssml_params.get("pitch") for c in results}
        # clean_params removes "0Hz" so check the other two are present
        assert "-10Hz" in found_pitches or "+10Hz" in found_pitches
    print(f"✓ grid_search yielded {len(results)} candidates")


async def test_grid_search_no_duplicates():
    """Grid search skips params already tried."""
    with tempfile.TemporaryDirectory() as d:
        wrapper  = MockWrapper(d)
        strategy = SSMLStrategy(wrapper, "test")
        strategy.record_rejection({"rate": "0%", "pitch": "-10Hz", "volume": "0%"})
        results = []
        async for c in strategy.grid_search(pitch_options=["-10Hz", "+10Hz"]):
            results.append(c)
        # -10Hz was pre-rejected so only +10Hz should come through
        assert len(results) == 1
    print(f"✓ grid_search skips pre-rejected params: {len(results)} candidate")


async def test_dsp_strategy_raises():
    with tempfile.TemporaryDirectory() as d:
        wrapper  = MockWrapper(d)
        strategy = DSPStrategy(wrapper, "test")
        try:
            await strategy.next_candidate()
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError as e:
            print(f"✓ DSPStrategy raises NotImplementedError: {e}")


def test_registry():
    strategies = list_strategies()
    assert "ssml_nudge" in strategies
    assert "dsp_repair" in strategies
    print(f"✓ registry: {strategies}")


def test_get_strategy():
    with tempfile.TemporaryDirectory() as d:
        wrapper = MockWrapper(d)
        s = get_strategy("ssml_nudge", wrapper, "test fragment")
        assert isinstance(s, SSMLStrategy)
    print("✓ get_strategy returns correct class")


def test_get_strategy_unknown():
    with tempfile.TemporaryDirectory() as d:
        wrapper = MockWrapper(d)
        try:
            get_strategy("nonexistent", wrapper, "test")
            assert False
        except KeyError as e:
            print(f"✓ get_strategy unknown raises KeyError: {e}")


def test_register_custom_strategy():
    class MyStrategy:
        name = "my_custom"
        def __init__(self, wrapper, text, seed_params=None): pass

    register_strategy("my_custom", MyStrategy)
    assert "my_custom" in list_strategies()
    print("✓ register_strategy adds to registry")


def test_clean_params_removes_noops():
    params = {"rate": "0%", "pitch": "0Hz", "volume": "0%"}
    cleaned = _clean_params(params)
    assert cleaned == {}
    print(f"✓ _clean_params removes all no-ops: {cleaned}")


def test_clean_params_keeps_real_values():
    params = {"rate": "+10%", "pitch": "0Hz", "volume": "-5%"}
    cleaned = _clean_params(params)
    assert cleaned.get("rate") == "+10%"
    assert "pitch" not in cleaned
    assert cleaned.get("volume") == "-5%"
    print(f"✓ _clean_params keeps real values: {cleaned}")


def test_params_summary():
    assert params_summary({"pitch": "+10Hz", "rate": "0%"}) == "pitch=+10Hz"
    assert params_summary({"pitch": "0Hz", "rate": "0%", "volume": "0%"}) == "no change"
    assert "rate" in params_summary({"rate": "+15%", "pitch": "+5Hz"})
    print("✓ params_summary formatting correct")


async def run_all():
    await test_basic_candidate()
    await test_hint_applied()
    await test_no_duplicate_params()
    await test_record_rejection_avoids_params()
    await test_record_acceptance_updates_seed()
    await test_grid_search()
    await test_grid_search_no_duplicates()
    await test_dsp_strategy_raises()
    test_registry()
    test_get_strategy()
    test_get_strategy_unknown()
    test_register_custom_strategy()
    test_clean_params_removes_noops()
    test_clean_params_keeps_real_values()
    test_params_summary()
    print("\n✓ All strategy checks passed.")


if __name__ == "__main__":
    asyncio.run(run_all())
