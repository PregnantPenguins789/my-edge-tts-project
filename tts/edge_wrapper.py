"""
tts/edge_wrapper.py

Async wrapper around edge-tts.
edge-tts 6+ outputs MP3. This wrapper:
  - saves raw MP3 from edge-tts
  - converts to WAV (16-bit PCM) for the splice engine
  - reads duration/sample_rate from the WAV

Dependencies:
    pip install edge-tts
    ffmpeg must be on PATH  (sudo apt install ffmpeg)
    OR: pip install pydub   (uses ffmpeg internally)
"""

import asyncio
import hashlib
import json
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class WordBoundary:
    word: str
    start_ms: int
    end_ms: int


@dataclass
class RenderResult:
    audio_path: str          # WAV path — ready for splice engine
    mp3_path: str            # original MP3 from edge-tts
    duration_ms: int
    sample_rate: int
    word_boundaries: list
    voice: str
    text: str
    ssml_params: dict = field(default_factory=dict)

    def word_boundaries_as_dicts(self) -> list[dict]:
        return [
            {"word": wb.word, "start_ms": wb.start_ms, "end_ms": wb.end_ms}
            for wb in self.word_boundaries
        ]

    def words_in_range(self, start_ms: int, end_ms: int) -> list:
        return [
            wb for wb in self.word_boundaries
            if wb.start_ms < end_ms and wb.end_ms > start_ms
        ]


# ---------------------------------------------------------------------------
# SSML builder
# ---------------------------------------------------------------------------

def build_ssml(text, voice, rate=None, pitch=None, volume=None) -> str:
    attrs = []
    if rate:   attrs.append(f'rate="{rate}"')
    if pitch:  attrs.append(f'pitch="{pitch}"')
    if volume: attrs.append(f'volume="{volume}"')
    prosody_open  = f"<prosody {' '.join(attrs)}>" if attrs else ""
    prosody_close = "</prosody>" if attrs else ""
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">{prosody_open}{text}{prosody_close}</voice></speak>'
    )


def _deterministic_filename(text: str, ssml_params: dict, prefix: str = "render") -> str:
    payload = json.dumps({"text": text, "params": ssml_params}, sort_keys=True)
    digest  = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"   # no extension — added per format


def _get_wav_duration_ms(path: str) -> int:
    with wave.open(path, "rb") as wf:
        return int(wf.getnframes() / wf.getframerate() * 1000)


def _get_wav_sample_rate(path: str) -> int:
    with wave.open(path, "rb") as wf:
        return wf.getframerate()


# ---------------------------------------------------------------------------
# MP3 → WAV conversion
# ---------------------------------------------------------------------------

def _mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int = 22050) -> None:
    """
    Convert MP3 to 16-bit mono WAV using ffmpeg.
    ffmpeg must be installed: sudo apt install ffmpeg
    """
    cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-sample_fmt", "s16",
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed:\n{result.stderr.decode()}\n"
            "Install ffmpeg with: sudo apt install ffmpeg"
        )


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class EdgeTTSWrapper:

    def __init__(
        self,
        voice: str = "en-US-AriaNeural",
        output_dir: str = "data/audio",
        sample_rate: int = 22050,
    ):
        self.voice       = voice
        self.output_dir  = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def render(
        self,
        text: str,
        ssml_params: Optional[dict] = None,
        filename: Optional[str] = None,
    ) -> RenderResult:
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

        params   = ssml_params or {}
        basename = filename or _deterministic_filename(text, params)
        mp3_path = self.output_dir / f"{basename}.mp3"
        wav_path = self.output_dir / f"{basename}.wav"

        communicate = edge_tts.Communicate(text=text, voice=self.voice)

        boundaries: list[dict] = []
        audio_chunks: list[bytes] = []

        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
                elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                    boundaries.append(chunk)
        except Exception:
            if mp3_path.exists():
                mp3_path.unlink()
            raise

        # Write MP3
        with open(mp3_path, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)

        # Convert to WAV for splice engine
        try:
            _mp3_to_wav(str(mp3_path), str(wav_path), self.sample_rate)
        except Exception:
            if wav_path.exists():
                wav_path.unlink()
            raise

        word_boundaries = _parse_boundaries(boundaries)
        duration_ms     = _get_wav_duration_ms(str(wav_path))
        sample_rate     = _get_wav_sample_rate(str(wav_path))

        return RenderResult(
            audio_path=str(wav_path),
            mp3_path=str(mp3_path),
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            word_boundaries=word_boundaries,
            voice=self.voice,
            text=text,
            ssml_params=params,
        )

    async def render_region(
        self,
        text_fragment: str,
        ssml_params: Optional[dict] = None,
        attempt_number: int = 1,
    ) -> RenderResult:
        params   = ssml_params or {}
        basename = _deterministic_filename(
            text_fragment, params, prefix=f"candidate_a{attempt_number:03d}"
        )
        return await self.render(text_fragment, ssml_params=params, filename=basename)

    def render_sync(self, text: str, ssml_params: Optional[dict] = None) -> RenderResult:
        return asyncio.run(self.render(text, ssml_params=ssml_params))


# ---------------------------------------------------------------------------
# Boundary parsing
# ---------------------------------------------------------------------------

def _parse_boundaries(raw: list[dict]) -> list[WordBoundary]:
    result = []
    for i, b in enumerate(raw):
        start_ms = _ticks_to_ms(b.get("offset", 0))
        dur_ms   = _ticks_to_ms(b.get("duration", 0))
        end_ms   = start_ms + dur_ms
        if dur_ms == 0 and i + 1 < len(raw):
            end_ms = _ticks_to_ms(raw[i + 1].get("offset", 0))
        result.append(WordBoundary(
            word=b.get("text", ""),
            start_ms=start_ms,
            end_ms=end_ms,
        ))
    return result


def _ticks_to_ms(ticks: int) -> int:
    return ticks // 10_000


# ---------------------------------------------------------------------------
# Voice listing
# ---------------------------------------------------------------------------

async def list_voices(locale: Optional[str] = None) -> list[dict]:
    if not EDGE_TTS_AVAILABLE:
        raise RuntimeError("edge-tts not installed.")
    voices = await edge_tts.list_voices()
    if locale:
        voices = [v for v in voices if v.get("Locale", "").startswith(locale)]
    return voices
