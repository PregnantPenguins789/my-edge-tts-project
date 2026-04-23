"""
audio/features.py

Acoustic feature extraction for TTS correction regions.

Extracts:
    mean_f0         float   Hz  (0.0 if unvoiced)
    f0_std          float   Hz
    f0_min          float   Hz
    f0_max          float   Hz
    f0_contour      list[float]  per-frame F0 values (Hz, 0=unvoiced)
    energy_mean     float   RMS amplitude 0.0–1.0
    energy_std      float
    energy_contour  list[float]  per-frame RMS
    duration_ms     int
    voiced_ratio    float   fraction of frames detected as voiced
    pause_before_ms int     silence before region (from context window)
    pause_after_ms  int     silence after region
    speaking_rate   float   voiced frames per second (proxy for rate)

Two backends:
    parselmouth (Praat)  — used automatically if installed, higher quality
    stdlib autocorrelation — always available, adequate for short regions

Output is a FeatureSet dataclass, also serializable to dict for JSON storage.

Dependencies:
    stdlib only (always)
    parselmouth (optional, pip install praat-parselmouth)
"""

import math
import array
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

from audio.splice import AudioBuffer, load


# ---------------------------------------------------------------------------
# FeatureSet
# ---------------------------------------------------------------------------

@dataclass
class FeatureSet:
    # Pitch
    mean_f0:        float = 0.0
    f0_std:         float = 0.0
    f0_min:         float = 0.0
    f0_max:         float = 0.0
    f0_contour:     list  = field(default_factory=list)

    # Energy
    energy_mean:    float = 0.0
    energy_std:     float = 0.0
    energy_contour: list  = field(default_factory=list)

    # Timing
    duration_ms:    int   = 0
    voiced_ratio:   float = 0.0
    speaking_rate:  float = 0.0   # voiced frames / second
    pause_before_ms: int  = 0
    pause_after_ms:  int  = 0

    # Metadata
    backend: str = "stdlib"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Truncate contours to 2dp for compact JSON storage
        d["f0_contour"]     = [round(v, 2) for v in self.f0_contour]
        d["energy_contour"] = [round(v, 4) for v in self.energy_contour]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSet":
        return cls(**d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(
    audio: AudioBuffer,
    context_before: Optional[AudioBuffer] = None,
    context_after:  Optional[AudioBuffer] = None,
    frame_ms: int = 10,
    f0_min_hz: float = 75.0,
    f0_max_hz: float = 500.0,
) -> FeatureSet:
    """
    Extract features from audio (the region being corrected).

    context_before / context_after: audio immediately surrounding the region.
    Used to measure pause length at boundaries.

    frame_ms: analysis frame size in ms (default 10ms)
    f0_min_hz / f0_max_hz: search range for pitch detection
    """
    if PARSELMOUTH_AVAILABLE:
        features = _extract_parselmouth(audio, frame_ms, f0_min_hz, f0_max_hz)
    else:
        features = _extract_stdlib(audio, frame_ms, f0_min_hz, f0_max_hz)

    features.duration_ms = audio.duration_ms

    if context_before is not None:
        features.pause_before_ms = _measure_trailing_silence(context_before)
    if context_after is not None:
        features.pause_after_ms = _measure_leading_silence(context_after)

    return features


def extract_from_path(
    path: str,
    context_before_path: Optional[str] = None,
    context_after_path:  Optional[str] = None,
    **kwargs,
) -> FeatureSet:
    audio   = load(path)
    before  = load(context_before_path) if context_before_path else None
    after   = load(context_after_path)  if context_after_path  else None
    return extract(audio, before, after, **kwargs)


# ---------------------------------------------------------------------------
# Parselmouth (Praat) backend
# ---------------------------------------------------------------------------

def _extract_parselmouth(
    audio: AudioBuffer,
    frame_ms: int,
    f0_min_hz: float,
    f0_max_hz: float,
) -> FeatureSet:
    import numpy as np
    import tempfile, os

    # Write to temp WAV — parselmouth reads files
    import wave as _wave, array as _array, tempfile as _tmp, os as _os

    tmp = _tmp.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    raw = _array.array("h", audio.samples).tobytes()
    with _wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(audio.channels)
        wf.setsampwidth(2)
        wf.setframerate(audio.sample_rate)
        wf.writeframes(raw)

    try:
        snd = parselmouth.Sound(tmp.name)

        # Pitch
        pitch_obj = snd.to_pitch(
            time_step=frame_ms / 1000,
            pitch_floor=f0_min_hz,
            pitch_ceiling=f0_max_hz,
        )
        f0_values = [
            pitch_obj.get_value_at_time(t) or 0.0
            for t in pitch_obj.xs()
        ]
        f0_voiced = [v for v in f0_values if v > 0]

        # Intensity
        intensity_obj = snd.to_intensity(time_step=frame_ms / 1000)
        energy_values = [
            max(0.0, intensity_obj.get_value(t) or 0.0) / 96.0  # dB→0–1 approx
            for t in intensity_obj.xs()
        ]

    finally:
        _os.unlink(tmp.name)

    return _build_featureset(f0_values, f0_voiced, energy_values, backend="parselmouth")


# ---------------------------------------------------------------------------
# Stdlib autocorrelation backend
# ---------------------------------------------------------------------------

def _extract_stdlib(
    audio: AudioBuffer,
    frame_ms: int,
    f0_min_hz: float,
    f0_max_hz: float,
) -> FeatureSet:
    mono     = _to_mono(audio.samples, audio.channels)
    sr       = audio.sample_rate
    frame_n  = int(sr * frame_ms / 1000)

    f0_values:     list[float] = []
    energy_values: list[float] = []

    for start in range(0, len(mono) - frame_n, frame_n):
        frame = mono[start:start + frame_n]
        energy_values.append(_rms(frame) / 32768.0)
        f0 = _autocorr_f0(frame, sr, f0_min_hz, f0_max_hz)
        f0_values.append(f0)

    f0_voiced = [v for v in f0_values if v > 0]
    return _build_featureset(f0_values, f0_voiced, energy_values, backend="stdlib")


# ---------------------------------------------------------------------------
# Shared builder
# ---------------------------------------------------------------------------

def _build_featureset(
    f0_values:     list[float],
    f0_voiced:     list[float],
    energy_values: list[float],
    backend: str,
) -> FeatureSet:
    n_frames = len(f0_values)

    mean_f0  = statistics.mean(f0_voiced)  if f0_voiced     else 0.0
    f0_std   = statistics.stdev(f0_voiced) if len(f0_voiced) > 1 else 0.0
    f0_min   = min(f0_voiced)              if f0_voiced     else 0.0
    f0_max   = max(f0_voiced)              if f0_voiced     else 0.0

    e_mean   = statistics.mean(energy_values)              if energy_values else 0.0
    e_std    = statistics.stdev(energy_values) if len(energy_values) > 1 else 0.0

    voiced_ratio   = len(f0_voiced) / n_frames if n_frames > 0 else 0.0
    speaking_rate  = len(f0_voiced)            # voiced frames; caller can divide by duration

    return FeatureSet(
        mean_f0        = round(mean_f0, 2),
        f0_std         = round(f0_std,  2),
        f0_min         = round(f0_min,  2),
        f0_max         = round(f0_max,  2),
        f0_contour     = f0_values,
        energy_mean    = round(e_mean, 4),
        energy_std     = round(e_std,  4),
        energy_contour = energy_values,
        voiced_ratio   = round(voiced_ratio, 3),
        speaking_rate  = float(speaking_rate),
        backend        = backend,
    )


# ---------------------------------------------------------------------------
# Autocorrelation F0 (stdlib)
# ---------------------------------------------------------------------------

def _autocorr_f0(
    frame: list,
    sample_rate: int,
    f0_min: float,
    f0_max: float,
) -> float:
    """
    Estimate F0 of a mono PCM frame via autocorrelation.
    Returns 0.0 if the frame appears unvoiced.

    Period search range:
        lag_min = sample_rate / f0_max
        lag_max = sample_rate / f0_min
    """
    n = len(frame)
    if n < 2:
        return 0.0

    # Energy gate — skip near-silent frames
    rms = _rms(frame)
    if rms < 200:  # ~0.6% of 16-bit max
        return 0.0

    lag_min = max(1, int(sample_rate / f0_max))
    lag_max = min(n - 1, int(sample_rate / f0_min))

    if lag_min >= lag_max:
        return 0.0

    # Compute autocorrelation at lag 0 (normalizer)
    r0 = sum(s * s for s in frame)
    if r0 == 0:
        return 0.0

    best_lag   = 0
    best_corr  = -1.0

    for lag in range(lag_min, lag_max + 1):
        r = sum(frame[i] * frame[i + lag] for i in range(n - lag))
        normalized = r / r0
        if normalized > best_corr:
            best_corr = normalized
            best_lag  = lag

    # Voicing threshold — reject weak correlations
    if best_corr < 0.3:
        return 0.0

    return round(sample_rate / best_lag, 2)


# ---------------------------------------------------------------------------
# Silence / pause detection
# ---------------------------------------------------------------------------

_SILENCE_THRESHOLD = 0.01   # RMS fraction of max (1.0)
_SILENCE_FRAME_MS  = 10


def _measure_trailing_silence(audio: AudioBuffer) -> int:
    """
    Measure milliseconds of silence at the END of audio.
    Used for pause_before_ms (silence trailing into the region).
    """
    mono    = _to_mono(audio.samples, audio.channels)
    sr      = audio.sample_rate
    frame_n = int(sr * _SILENCE_FRAME_MS / 1000)
    frames  = [mono[i:i + frame_n] for i in range(0, len(mono) - frame_n, frame_n)]

    silent_frames = 0
    for frame in reversed(frames):
        rms = _rms(frame) / 32768.0
        if rms < _SILENCE_THRESHOLD:
            silent_frames += 1
        else:
            break

    return silent_frames * _SILENCE_FRAME_MS


def _measure_leading_silence(audio: AudioBuffer) -> int:
    """
    Measure milliseconds of silence at the START of audio.
    Used for pause_after_ms.
    """
    mono    = _to_mono(audio.samples, audio.channels)
    sr      = audio.sample_rate
    frame_n = int(sr * _SILENCE_FRAME_MS / 1000)
    frames  = [mono[i:i + frame_n] for i in range(0, len(mono) - frame_n, frame_n)]

    silent_frames = 0
    for frame in frames:
        rms = _rms(frame) / 32768.0
        if rms < _SILENCE_THRESHOLD:
            silent_frames += 1
        else:
            break

    return silent_frames * _SILENCE_FRAME_MS


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _rms(samples: list) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _to_mono(samples: list, channels: int) -> list:
    if channels == 1:
        return samples
    mono = []
    for i in range(0, len(samples), channels):
        mono.append(int(sum(samples[i:i + channels]) / channels))
    return mono


# ---------------------------------------------------------------------------
# ASCII contour display (for TUI)
# ---------------------------------------------------------------------------

def f0_to_ascii(
    f0_contour: list[float],
    width: int = 60,
    height: int = 8,
    label: bool = True,
) -> str:
    """
    Render an F0 contour as a small ASCII plot.
    Unvoiced frames (0.0) are shown as dots at baseline.

    Returns a multi-line string ready for display in a Textual widget.
    """
    voiced = [v for v in f0_contour if v > 0]
    if not voiced:
        return "(no voiced frames detected)"

    f_min = min(voiced)
    f_max = max(voiced)
    f_range = f_max - f_min or 1.0

    # Downsample contour to display width
    step = max(1, len(f0_contour) // width)
    cols = []
    for i in range(0, len(f0_contour), step):
        chunk = f0_contour[i:i + step]
        voiced_chunk = [v for v in chunk if v > 0]
        if voiced_chunk:
            cols.append(sum(voiced_chunk) / len(voiced_chunk))
        else:
            cols.append(0.0)
    cols = cols[:width]

    # Build grid
    grid = [[" "] * len(cols) for _ in range(height)]
    for x, val in enumerate(cols):
        if val == 0.0:
            grid[height - 1][x] = "·"
        else:
            row = int((1.0 - (val - f_min) / f_range) * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][x] = "█"

    lines = ["".join(row) for row in grid]
    if label:
        lines.insert(0, f"F0  {f_max:.0f}Hz ┐")
        lines.append(   f"    {f_min:.0f}Hz ┘")
    return "\n".join(lines)
