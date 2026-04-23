"""
test_edge_wrapper.py

Tests the edge_wrapper interface using a mock — no live edge-tts call needed.
Run from project root: python test_edge_wrapper.py
"""

import sys
import asyncio
import tempfile
import wave
import struct
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


# ---------------------------------------------------------------------------
# Minimal WAV writer for mock audio output
# ---------------------------------------------------------------------------

def _write_silent_wav(path: str, duration_ms: int = 500, sample_rate: int = 22050):
    n_frames = int(sample_rate * duration_ms / 1000)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))


# ---------------------------------------------------------------------------
# Mock edge_tts module
# ---------------------------------------------------------------------------

MOCK_BOUNDARIES = [
    {"type": "WordBoundary", "text": "Hello",   "offset": 0,          "duration": 4_000_000},
    {"type": "WordBoundary", "text": "world",   "offset": 4_500_000,  "duration": 4_000_000},
    {"type": "WordBoundary", "text": "this",    "offset": 9_500_000,  "duration": 3_000_000},
    {"type": "WordBoundary", "text": "is",      "offset": 13_000_000, "duration": 2_000_000},
    {"type": "WordBoundary", "text": "a",       "offset": 15_500_000, "duration": 1_500_000},
    {"type": "WordBoundary", "text": "test",    "offset": 17_500_000, "duration": 4_000_000},
]

MOCK_AUDIO_CHUNK = b"\x00" * 100  # placeholder bytes


async def mock_stream():
    """Yields mock audio + word boundary chunks as edge-tts would."""
    yield {"type": "audio", "data": MOCK_AUDIO_CHUNK}
    for b in MOCK_BOUNDARIES:
        yield b
    yield {"type": "audio", "data": MOCK_AUDIO_CHUNK}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ticks_to_ms():
    from tts.edge_wrapper import _ticks_to_ms
    assert _ticks_to_ms(0) == 0
    assert _ticks_to_ms(10_000) == 1
    assert _ticks_to_ms(4_000_000) == 400
    print("✓ _ticks_to_ms conversion correct")


def test_parse_boundaries():
    from tts.edge_wrapper import _parse_boundaries
    wbs = _parse_boundaries(MOCK_BOUNDARIES)
    assert len(wbs) == 6
    assert wbs[0].word == "Hello"
    assert wbs[0].start_ms == 0
    assert wbs[0].end_ms == 400       # 4_000_000 ticks = 400ms
    assert wbs[1].word == "world"
    assert wbs[1].start_ms == 450     # 4_500_000 ticks = 450ms
    print(f"✓ Boundaries parsed: {[(w.word, w.start_ms, w.end_ms) for w in wbs]}")


def test_words_in_range():
    from tts.edge_wrapper import _parse_boundaries, RenderResult
    wbs = _parse_boundaries(MOCK_BOUNDARIES)
    result = RenderResult(
        audio_path="x.wav", mp3_path="x.mp3", duration_ms=2100, sample_rate=22050,
        word_boundaries=wbs, voice="en-US-AriaNeural", text="Hello world this is a test"
    )
    in_range = result.words_in_range(400, 1000)
    words = [w.word for w in in_range]
    assert "world" in words
    print(f"✓ words_in_range(400, 1000): {words}")


def test_deterministic_filename():
    from tts.edge_wrapper import _deterministic_filename
    f1 = _deterministic_filename("hello", {"pitch": "+10%"})
    f2 = _deterministic_filename("hello", {"pitch": "+10%"})
    f3 = _deterministic_filename("hello", {"pitch": "+20%"})
    assert f1 == f2, "Same inputs must produce same filename"
    assert f1 != f3, "Different params must produce different filename"
    assert f1.endswith(".wav")
    print(f"✓ Deterministic filenames: {f1} / {f3}")


def test_build_ssml():
    from tts.edge_wrapper import build_ssml
    ssml = build_ssml("Hello", "en-US-AriaNeural", rate="+10%", pitch="+5Hz")
    assert 'rate="+10%"' in ssml
    assert 'pitch="+5Hz"' in ssml
    assert "Hello" in ssml
    assert "en-US-AriaNeural" in ssml
    print(f"✓ SSML built correctly")

    # No params — no prosody wrapper
    ssml_bare = build_ssml("Hello", "en-US-AriaNeural")
    assert "<prosody" not in ssml_bare
    print("✓ SSML with no params omits prosody tag")


async def test_render_mock():
    """Full render test using a subclassed wrapper — no live edge-tts needed."""
    with tempfile.TemporaryDirectory() as tmpdir:

        import tts.edge_wrapper as ew

        class MockWrapper(ew.EdgeTTSWrapper):
            async def render(self, text, ssml_params=None, filename=None):
                params = ssml_params or {}
                fname = filename or ew._deterministic_filename(text, params)
                audio_path = Path(self.output_dir) / fname
                _write_silent_wav(str(audio_path), duration_ms=500)

                boundaries = ew._parse_boundaries(MOCK_BOUNDARIES)

                return ew.RenderResult(
                    audio_path=str(audio_path),
                    mp3_path=str(audio_path).replace('.wav', '.mp3'),
                    duration_ms=ew._get_audio_duration_ms(str(audio_path)),
                    sample_rate=ew._get_audio_sample_rate(str(audio_path)),
                    word_boundaries=boundaries,
                    voice=self.voice,
                    text=text,
                    ssml_params=params,
                )

        wrapper = MockWrapper(voice="en-US-AriaNeural", output_dir=tmpdir)
        result = await wrapper.render(
            "Hello world this is a test",
            ssml_params={"pitch": "+10%"}
        )

        assert result.voice == "en-US-AriaNeural"
        assert result.ssml_params == {"pitch": "+10%"}
        assert result.sample_rate == 22050
        assert result.duration_ms == 500
        assert len(result.word_boundaries) == 6
        assert isinstance(result.word_boundaries_as_dicts(), list)
        assert result.word_boundaries_as_dicts()[0]["word"] == "Hello"
        print(f"✓ render() returns RenderResult with correct fields")
        print(f"  duration_ms={result.duration_ms}, sample_rate={result.sample_rate}")
        print(f"  word_boundaries: {[(w.word, w.start_ms) for w in result.word_boundaries]}")


def run_all():
    test_ticks_to_ms()
    test_parse_boundaries()
    test_words_in_range()
    test_deterministic_filename()
    test_build_ssml()
    asyncio.run(test_render_mock())
    print("\n✓ All edge_wrapper checks passed.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run_all()
