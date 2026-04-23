"""
audio/splice.py

Deterministic, non-destructive audio splice engine.
All operations return new audio objects — originals are never modified.

Primary operations:
    cut_region(audio, start_ms, end_ms)         → AudioBuffer
    replace_region(audio, start_ms, end_ms, replacement) → AudioBuffer
    crossfade(a, b, fade_ms)                    → AudioBuffer
    save(audio, path)                           → str

AudioBuffer is a lightweight container:
    samples     list[int]   (16-bit signed PCM)
    sample_rate int
    channels    int

All ms timecodes map to sample indices via:
    sample_index = ms * sample_rate // 1000

Dependencies: stdlib only (wave, array, struct)
Optional:     pydub (used automatically if available for MP3/OGG input)
"""

import wave
import array
import struct
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# AudioBuffer
# ---------------------------------------------------------------------------

@dataclass
class AudioBuffer:
    samples: list        # list[int], 16-bit signed PCM, interleaved if stereo
    sample_rate: int
    channels: int = 1

    @property
    def duration_ms(self) -> int:
        n_frames = len(self.samples) // self.channels
        return int(n_frames / self.sample_rate * 1000)

    @property
    def n_frames(self) -> int:
        return len(self.samples) // self.channels

    def _ms_to_frame(self, ms: int) -> int:
        return int(ms * self.sample_rate / 1000)

    def _ms_to_sample(self, ms: int) -> int:
        return self._ms_to_frame(ms) * self.channels

    def slice_ms(self, start_ms: int, end_ms: int) -> "AudioBuffer":
        """Return a new AudioBuffer containing [start_ms, end_ms)."""
        s = self._ms_to_sample(max(0, start_ms))
        e = self._ms_to_sample(min(self.duration_ms, end_ms))
        return AudioBuffer(
            samples=self.samples[s:e],
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def copy(self) -> "AudioBuffer":
        return AudioBuffer(
            samples=list(self.samples),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load(path: str) -> AudioBuffer:
    """
    Load a WAV file into an AudioBuffer.
    Raises ValueError if format is unsupported (not 16-bit PCM).
    """
    path = str(path)
    with wave.open(path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sampwidth   = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames    = wf.getnframes()
        raw         = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(
            f"Only 16-bit PCM WAV is supported. Got {sampwidth * 8}-bit: {path}"
        )

    samples = list(array.array("h", raw))
    return AudioBuffer(samples=samples, sample_rate=sample_rate, channels=n_channels)


def save(audio: AudioBuffer, path: str) -> str:
    """Write AudioBuffer to a WAV file. Returns the path written."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw = array.array("h", audio.samples).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(audio.channels)
        wf.setsampwidth(2)
        wf.setframerate(audio.sample_rate)
        wf.writeframes(raw)
    return path


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def cut_region(audio: AudioBuffer, start_ms: int, end_ms: int) -> AudioBuffer:
    """
    Remove [start_ms, end_ms) from audio.
    Returns a new AudioBuffer with that region excised.
    """
    s = audio._ms_to_sample(max(0, start_ms))
    e = audio._ms_to_sample(min(audio.duration_ms, end_ms))
    samples = audio.samples[:s] + audio.samples[e:]
    return AudioBuffer(samples=samples, sample_rate=audio.sample_rate, channels=audio.channels)


def replace_region(
    audio: AudioBuffer,
    start_ms: int,
    end_ms: int,
    replacement: AudioBuffer,
    fade_ms: int = 20,
) -> AudioBuffer:
    """
    Replace [start_ms, end_ms) in audio with replacement.
    Applies a crossfade at both boundaries to hide seams.

    Requirements:
    - replacement.sample_rate must match audio.sample_rate
    - replacement.channels must match audio.channels

    fade_ms: crossfade length at each boundary (default 20ms)
              set to 0 to hard-cut
    """
    _assert_compatible(audio, replacement)

    s_sample = audio._ms_to_sample(max(0, start_ms))
    e_sample = audio._ms_to_sample(min(audio.duration_ms, end_ms))

    before = audio.samples[:s_sample]
    after  = audio.samples[e_sample:]
    rep    = replacement.samples

    if fade_ms <= 0:
        combined = before + rep + after
    else:
        # Crossfade at the join between before↔replacement
        before_xf, rep_start = _crossfade_join(
            before, rep, fade_ms, audio.sample_rate, audio.channels
        )
        # Crossfade at the join between replacement↔after
        rep_end, after_xf = _crossfade_join(
            rep_start, after, fade_ms, audio.sample_rate, audio.channels
        )
        combined = before_xf + rep_end + after_xf

    return AudioBuffer(samples=combined, sample_rate=audio.sample_rate, channels=audio.channels)


def crossfade(a: AudioBuffer, b: AudioBuffer, fade_ms: int = 20) -> AudioBuffer:
    """
    Concatenate a and b with a crossfade at the join point.
    The tail of a fades out while the head of b fades in.
    """
    _assert_compatible(a, b)
    tail, head = _crossfade_join(a.samples, b.samples, fade_ms, a.sample_rate, a.channels)
    return AudioBuffer(samples=tail + head, sample_rate=a.sample_rate, channels=a.channels)


# ---------------------------------------------------------------------------
# Duration matching
# ---------------------------------------------------------------------------

def stretch_to_duration(audio: AudioBuffer, target_ms: int) -> AudioBuffer:
    """
    Naive linear time-stretch to match target_ms.
    Acceptable for small corrections (±20%). For larger changes,
    a phase vocoder (pydub/librosa) is preferable.

    This is a placeholder implementation — it resamples by interpolation.
    """
    if audio.duration_ms == 0:
        return audio.copy()

    ratio = target_ms / audio.duration_ms
    src = audio.samples
    n_src = len(src) // audio.channels
    n_dst = int(n_src * ratio)

    out = []
    for i in range(n_dst):
        src_pos = i / ratio
        i0 = int(src_pos)
        i1 = min(i0 + 1, n_src - 1)
        frac = src_pos - i0
        for c in range(audio.channels):
            s0 = src[i0 * audio.channels + c]
            s1 = src[i1 * audio.channels + c]
            out.append(int(s0 + (s1 - s0) * frac))

    return AudioBuffer(samples=out, sample_rate=audio.sample_rate, channels=audio.channels)


def resample(audio: AudioBuffer, target_rate: int) -> AudioBuffer:
    """
    Resample audio to target_rate using linear interpolation.
    Used to normalize sample rates before splicing.
    """
    if audio.sample_rate == target_rate:
        return audio.copy()

    ratio = target_rate / audio.sample_rate
    src = audio.samples
    n_src = len(src) // audio.channels
    n_dst = int(n_src * ratio)

    out = []
    for i in range(n_dst):
        src_pos = i / ratio
        i0 = int(src_pos)
        i1 = min(i0 + 1, n_src - 1)
        frac = src_pos - i0
        for c in range(audio.channels):
            s0 = src[i0 * audio.channels + c]
            s1 = src[i1 * audio.channels + c]
            out.append(int(s0 + (s1 - s0) * frac))

    return AudioBuffer(samples=out, sample_rate=target_rate, channels=audio.channels)


def normalize_for_splice(source: AudioBuffer, target: AudioBuffer) -> AudioBuffer:
    """
    Ensure source matches target's sample_rate and channels.
    Returns a compatible copy of source ready for splicing into target.
    """
    result = source.copy()
    if result.sample_rate != target.sample_rate:
        result = resample(result, target.sample_rate)
    if result.channels != target.channels:
        result = _convert_channels(result, target.channels)
    return result


# ---------------------------------------------------------------------------
# Waveform downsampling for TUI visualization
# ---------------------------------------------------------------------------

def downsample_for_display(audio: AudioBuffer, bins: int = 1500) -> list[float]:
    """
    Reduce audio to `bins` RMS amplitude values in [0.0, 1.0].
    Used by the TUI waveform display. Fast enough for interactive use.
    """
    mono = _to_mono(audio.samples, audio.channels)
    n = len(mono)
    if n == 0:
        return [0.0] * bins

    chunk_size = max(1, n // bins)
    result = []
    for i in range(bins):
        start = i * chunk_size
        end   = min(start + chunk_size, n)
        chunk = mono[start:end]
        if chunk:
            rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
            result.append(min(1.0, rms / 32768.0))
        else:
            result.append(0.0)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_compatible(a: AudioBuffer, b: AudioBuffer) -> None:
    if a.sample_rate != b.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: {a.sample_rate} vs {b.sample_rate}. "
            "Call normalize_for_splice() first."
        )
    if a.channels != b.channels:
        raise ValueError(
            f"Channel count mismatch: {a.channels} vs {b.channels}. "
            "Call normalize_for_splice() first."
        )


def _crossfade_join(
    a: list,
    b: list,
    fade_ms: int,
    sample_rate: int,
    channels: int,
) -> tuple[list, list]:
    """
    Apply a linear crossfade between the tail of a and the head of b.
    Returns (a_faded, b_remaining) — does NOT overlap them (no duration change).
    The fade region is applied in-place at the boundary:
        last fade_samples of a fade to zero
        first fade_samples of b fade from zero
    """
    fade_frames  = int(fade_ms * sample_rate / 1000)
    fade_samples = fade_frames * channels

    a = list(a)
    b = list(b)

    # Fade out tail of a
    if len(a) >= fade_samples > 0:
        for i in range(fade_samples):
            gain = 1.0 - (i // channels) / fade_frames
            idx  = len(a) - fade_samples + i
            a[idx] = int(a[idx] * gain)

    # Fade in head of b
    if len(b) >= fade_samples > 0:
        for i in range(min(fade_samples, len(b))):
            gain = (i // channels) / fade_frames
            b[i] = int(b[i] * gain)

    return a, b


def _to_mono(samples: list, channels: int) -> list:
    if channels == 1:
        return samples
    mono = []
    for i in range(0, len(samples), channels):
        mono.append(int(sum(samples[i:i + channels]) / channels))
    return mono


def _convert_channels(audio: AudioBuffer, target_channels: int) -> AudioBuffer:
    if audio.channels == target_channels:
        return audio.copy()
    if audio.channels == 2 and target_channels == 1:
        mono = _to_mono(audio.samples, 2)
        return AudioBuffer(samples=mono, sample_rate=audio.sample_rate, channels=1)
    if audio.channels == 1 and target_channels == 2:
        stereo = []
        for s in audio.samples:
            stereo.extend([s, s])
        return AudioBuffer(samples=stereo, sample_rate=audio.sample_rate, channels=2)
    raise ValueError(f"Unsupported channel conversion: {audio.channels} → {target_channels}")
