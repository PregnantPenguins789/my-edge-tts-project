"""
test_splice.py
Tests for audio/splice.py using only stdlib-generated WAV data.
Run from project root: python test_splice.py
"""

import sys
import wave
import array
import struct
import math
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audio.splice import (
    AudioBuffer, load, save,
    cut_region, replace_region, crossfade,
    stretch_to_duration, resample,
    normalize_for_splice, downsample_for_display,
    _to_mono, _convert_channels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tone(freq_hz: float, duration_ms: int, sample_rate: int = 22050) -> AudioBuffer:
    """Generate a sine wave at freq_hz for duration_ms."""
    n = int(sample_rate * duration_ms / 1000)
    samples = [
        int(16000 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        for i in range(n)
    ]
    return AudioBuffer(samples=samples, sample_rate=sample_rate, channels=1)


def _make_wav_file(path: str, duration_ms: int = 500, freq_hz: float = 440.0,
                   sample_rate: int = 22050) -> str:
    buf = _make_tone(freq_hz, duration_ms, sample_rate)
    return save(buf, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_save():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.wav")
        buf = _make_tone(440, 1000)
        save(buf, path)
        loaded = load(path)
        assert loaded.sample_rate == 22050
        assert loaded.channels == 1
        assert abs(loaded.duration_ms - 1000) <= 1
        assert len(loaded.samples) == len(buf.samples)
    print("✓ load/save round-trip correct")


def test_duration_ms():
    buf = _make_tone(440, 500)
    assert abs(buf.duration_ms - 500) <= 1
    print(f"✓ duration_ms: {buf.duration_ms}ms")


def test_slice_ms():
    buf = _make_tone(440, 1000)
    sliced = buf.slice_ms(200, 600)
    assert abs(sliced.duration_ms - 400) <= 2
    print(f"✓ slice_ms(200, 600): {sliced.duration_ms}ms")


def test_cut_region():
    buf = _make_tone(440, 1000)
    original_dur = buf.duration_ms
    cut = cut_region(buf, 200, 500)
    expected = original_dur - 300
    assert abs(cut.duration_ms - expected) <= 2, \
        f"Expected ~{expected}ms, got {cut.duration_ms}ms"
    print(f"✓ cut_region(200, 500): {buf.duration_ms}ms → {cut.duration_ms}ms")


def test_replace_region_duration():
    original = _make_tone(440, 1000)
    replacement = _make_tone(880, 200)  # shorter replacement
    result = replace_region(original, 300, 600, replacement, fade_ms=0)
    # before(300) + replacement(200) + after(400) = 900
    expected = 300 + 200 + 400
    assert abs(result.duration_ms - expected) <= 3, \
        f"Expected ~{expected}ms, got {result.duration_ms}ms"
    print(f"✓ replace_region duration: {result.duration_ms}ms (expected ~{expected}ms)")


def test_replace_region_with_crossfade():
    original    = _make_tone(440, 1000)
    replacement = _make_tone(880, 300)
    result = replace_region(original, 300, 600, replacement, fade_ms=20)
    # Duration changes only by fade at boundaries — should be in reasonable range
    assert 500 < result.duration_ms < 1100
    print(f"✓ replace_region with crossfade: {result.duration_ms}ms")


def test_replace_preserves_endpoints():
    """Samples before start_ms and after end_ms should be unchanged."""
    original    = _make_tone(440, 1000)
    replacement = _make_tone(880, 300)
    result = replace_region(original, 400, 600, replacement, fade_ms=0)

    # Check first 100ms of samples unchanged
    n_check = original._ms_to_sample(100)
    assert result.samples[:n_check] == original.samples[:n_check]
    print("✓ replace_region preserves pre-region samples")


def test_crossfade_concat():
    a = _make_tone(440, 500)
    b = _make_tone(880, 500)
    result = crossfade(a, b, fade_ms=20)
    # Total duration = 500 + 500 = 1000ms (crossfade doesn't overlap, just shapes)
    assert abs(result.duration_ms - 1000) <= 5
    print(f"✓ crossfade concat: {result.duration_ms}ms")


def test_crossfade_incompatible_rates():
    a = _make_tone(440, 500, sample_rate=22050)
    b = _make_tone(880, 500, sample_rate=44100)
    try:
        crossfade(a, b)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ crossfade rejects incompatible rates: {e}")


def test_resample():
    buf = _make_tone(440, 500, sample_rate=22050)
    resampled = resample(buf, 44100)
    assert resampled.sample_rate == 44100
    assert abs(resampled.duration_ms - 500) <= 2
    print(f"✓ resample 22050→44100: {resampled.duration_ms}ms, rate={resampled.sample_rate}")


def test_normalize_for_splice():
    source = _make_tone(440, 300, sample_rate=44100)
    target = _make_tone(880, 500, sample_rate=22050)
    normalized = normalize_for_splice(source, target)
    assert normalized.sample_rate == 22050
    assert normalized.channels == target.channels
    print(f"✓ normalize_for_splice: rate={normalized.sample_rate}, ch={normalized.channels}")


def test_stretch_to_duration():
    buf = _make_tone(440, 500)
    stretched = stretch_to_duration(buf, 700)
    assert abs(stretched.duration_ms - 700) <= 5
    print(f"✓ stretch_to_duration(700): {stretched.duration_ms}ms")


def test_downsample_for_display():
    buf = _make_tone(440, 2000)
    bins = downsample_for_display(buf, bins=1500)
    assert len(bins) == 1500
    assert all(0.0 <= v <= 1.0 for v in bins)
    assert max(bins) > 0.0  # not silent
    print(f"✓ downsample_for_display: {len(bins)} bins, max={max(bins):.3f}")


def test_channel_conversion():
    mono = _make_tone(440, 200)
    assert mono.channels == 1
    stereo = _convert_channels(mono, 2)
    assert stereo.channels == 2
    assert len(stereo.samples) == len(mono.samples) * 2
    back = _convert_channels(stereo, 1)
    assert back.channels == 1
    print("✓ channel conversion mono↔stereo")


def test_to_mono():
    stereo_samples = [100, 200, 300, 400]  # two frames of stereo
    mono = _to_mono(stereo_samples, channels=2)
    assert mono == [150, 350]
    print("✓ _to_mono averages channels correctly")


def test_full_workflow():
    """Simulate the real workflow: render → mark region → replace → save."""
    with tempfile.TemporaryDirectory() as d:
        master_path  = os.path.join(d, "master.wav")
        patch_path   = os.path.join(d, "candidate.wav")
        result_path  = os.path.join(d, "spliced.wav")

        master    = _make_tone(440, 2000)   # 2 seconds original
        candidate = _make_tone(660, 400)    # 400ms replacement
        save(master, master_path)
        save(candidate, patch_path)

        loaded_master    = load(master_path)
        loaded_candidate = load(patch_path)
        compatible       = normalize_for_splice(loaded_candidate, loaded_master)
        spliced          = replace_region(loaded_master, 600, 900, compatible, fade_ms=20)
        save(spliced, result_path)

        final = load(result_path)
        assert final.sample_rate == 22050
        assert final.channels == 1
        assert final.duration_ms > 0
        print(f"✓ full workflow: master={loaded_master.duration_ms}ms → spliced={final.duration_ms}ms")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_load_save()
    test_duration_ms()
    test_slice_ms()
    test_cut_region()
    test_replace_region_duration()
    test_replace_region_with_crossfade()
    test_replace_preserves_endpoints()
    test_crossfade_concat()
    test_crossfade_incompatible_rates()
    test_resample()
    test_normalize_for_splice()
    test_stretch_to_duration()
    test_downsample_for_display()
    test_channel_conversion()
    test_to_mono()
    test_full_workflow()
    print("\n✓ All splice checks passed.")
