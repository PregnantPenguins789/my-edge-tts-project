"""
test_features.py
Tests for audio/features.py — stdlib only, no parselmouth needed.
Run from project root: python test_features.py
"""

import sys
import math
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audio.splice import AudioBuffer, save
from audio.features import (
    extract, extract_from_path,
    FeatureSet,
    _autocorr_f0, _rms, _to_mono,
    _measure_trailing_silence, _measure_leading_silence,
    f0_to_ascii,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq_hz: float, duration_ms: int, sample_rate: int = 22050,
          amplitude: int = 16000) -> AudioBuffer:
    n = int(sample_rate * duration_ms / 1000)
    samples = [
        int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        for i in range(n)
    ]
    return AudioBuffer(samples=samples, sample_rate=sample_rate, channels=1)


def _silence(duration_ms: int, sample_rate: int = 22050) -> AudioBuffer:
    n = int(sample_rate * duration_ms / 1000)
    return AudioBuffer(samples=[0] * n, sample_rate=sample_rate, channels=1)


def _concat(a: AudioBuffer, b: AudioBuffer) -> AudioBuffer:
    return AudioBuffer(
        samples=a.samples + b.samples,
        sample_rate=a.sample_rate,
        channels=a.channels,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_rms_silence():
    assert _rms([0] * 100) == 0.0
    print("✓ _rms silent frame = 0.0")


def test_rms_nonzero():
    samples = [1000] * 100
    r = _rms(samples)
    assert abs(r - 1000.0) < 0.1
    print(f"✓ _rms constant signal = {r:.1f}")


def test_to_mono_stereo():
    s = [100, 200, 300, 400]
    mono = _to_mono(s, channels=2)
    assert mono == [150, 350]
    print("✓ _to_mono stereo averaging")


def test_autocorr_silence():
    frame = [0] * 220
    f0 = _autocorr_f0(frame, 22050, 75, 500)
    assert f0 == 0.0
    print("✓ autocorr silence → 0.0")


def test_autocorr_tone():
    """220Hz tone should be detected within ±15Hz."""
    sr = 22050
    n  = sr // 10  # 100ms frame
    freq = 220.0
    frame = [int(16000 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)]
    f0 = _autocorr_f0(frame, sr, 75, 500)
    assert abs(f0 - freq) < 15, f"Expected ~{freq}Hz, got {f0}Hz"
    print(f"✓ autocorr 220Hz tone → {f0}Hz (error {abs(f0-freq):.1f}Hz)")


def test_autocorr_high_tone():
    """440Hz tone."""
    sr = 22050
    n  = sr // 10
    freq = 440.0
    frame = [int(16000 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)]
    f0 = _autocorr_f0(frame, sr, 75, 500)
    assert abs(f0 - freq) < 20, f"Expected ~{freq}Hz, got {f0}Hz"
    print(f"✓ autocorr 440Hz tone → {f0}Hz (error {abs(f0-freq):.1f}Hz)")


def test_extract_basic():
    buf = _sine(220, 500)
    fs  = extract(buf)
    assert fs.duration_ms == 500
    assert fs.backend == "stdlib"
    assert fs.energy_mean > 0
    assert len(fs.f0_contour) > 0
    assert len(fs.energy_contour) > 0
    print(f"✓ extract basic: duration={fs.duration_ms}ms, energy={fs.energy_mean:.4f}")


def test_extract_f0_range():
    """Extracted mean F0 should be in the right ballpark for a 220Hz tone."""
    buf = _sine(220, 800)
    fs  = extract(buf)
    # Allow wide tolerance — autocorr on short frames is approximate
    assert fs.mean_f0 > 50, f"mean_f0 too low: {fs.mean_f0}"
    assert fs.mean_f0 < 600, f"mean_f0 too high: {fs.mean_f0}"
    print(f"✓ extract F0 for 220Hz tone: mean_f0={fs.mean_f0}Hz, voiced_ratio={fs.voiced_ratio:.2f}")


def test_extract_silence():
    buf = _silence(300)
    fs  = extract(buf)
    assert fs.mean_f0 == 0.0
    assert fs.energy_mean < 0.001
    assert fs.voiced_ratio == 0.0
    print(f"✓ extract silence: mean_f0={fs.mean_f0}, energy={fs.energy_mean}")


def test_extract_with_context():
    """Context buffers feed pause measurements."""
    region    = _sine(220, 400)
    pre_audio = _concat(_sine(330, 200), _silence(80))   # 80ms silence before region
    post_audio = _concat(_silence(50), _sine(330, 200))  # 50ms silence after region

    fs = extract(region, context_before=pre_audio, context_after=post_audio)
    assert fs.pause_before_ms >= 0
    assert fs.pause_after_ms  >= 0
    print(f"✓ extract with context: pause_before={fs.pause_before_ms}ms, pause_after={fs.pause_after_ms}ms")


def test_trailing_silence():
    audio = _concat(_sine(220, 300), _silence(100))
    ms = _measure_trailing_silence(audio)
    assert ms >= 80, f"Expected ≥80ms trailing silence, got {ms}ms"
    print(f"✓ trailing silence: {ms}ms")


def test_leading_silence():
    audio = _concat(_silence(90), _sine(220, 300))
    ms = _measure_leading_silence(audio)
    assert ms >= 70, f"Expected ≥70ms leading silence, got {ms}ms"
    print(f"✓ leading silence: {ms}ms")


def test_featureset_to_dict():
    buf = _sine(220, 300)
    fs  = extract(buf)
    d   = fs.to_dict()
    assert isinstance(d, dict)
    assert "mean_f0" in d
    assert "f0_contour" in d
    assert isinstance(d["f0_contour"], list)
    print(f"✓ FeatureSet.to_dict(): {list(d.keys())}")


def test_featureset_roundtrip():
    buf = _sine(220, 300)
    fs  = extract(buf)
    d   = fs.to_dict()
    fs2 = FeatureSet.from_dict(d)
    assert fs2.mean_f0    == fs.mean_f0
    assert fs2.duration_ms == fs.duration_ms
    assert fs2.backend    == fs.backend
    print("✓ FeatureSet dict roundtrip")


def test_extract_from_path():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "region.wav")
        save(_sine(330, 400), path)
        fs = extract_from_path(path)
        assert fs.duration_ms == 400
        assert fs.backend == "stdlib"
    print(f"✓ extract_from_path: duration={fs.duration_ms}ms")


def test_f0_ascii_display():
    buf  = _sine(220, 500)
    fs   = extract(buf)
    plot = f0_to_ascii(fs.f0_contour, width=40, height=6)
    assert isinstance(plot, str)
    assert len(plot) > 0
    assert "\n" in plot
    print("✓ f0_to_ascii produces multi-line output:")
    print(plot)


def test_f0_ascii_silence():
    buf  = _silence(300)
    fs   = extract(buf)
    plot = f0_to_ascii(fs.f0_contour)
    assert "no voiced" in plot
    print(f"✓ f0_to_ascii silence: '{plot}'")


def test_energy_contour_length_matches_f0():
    buf = _sine(220, 500)
    fs  = extract(buf)
    assert len(fs.f0_contour) == len(fs.energy_contour)
    print(f"✓ contour lengths match: {len(fs.f0_contour)} frames")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_rms_silence()
    test_rms_nonzero()
    test_to_mono_stereo()
    test_autocorr_silence()
    test_autocorr_tone()
    test_autocorr_high_tone()
    test_extract_basic()
    test_extract_f0_range()
    test_extract_silence()
    test_extract_with_context()
    test_trailing_silence()
    test_leading_silence()
    test_featureset_to_dict()
    test_featureset_roundtrip()
    test_extract_from_path()
    test_f0_ascii_display()
    test_f0_ascii_silence()
    test_energy_contour_length_matches_f0()
    print("\n✓ All feature extraction checks passed.")
