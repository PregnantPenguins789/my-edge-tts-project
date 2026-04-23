# TTS Prosody Correction Tool

Iterative TTS inflection correction with a Textual TUI.
Generates edge-tts candidates, extracts acoustic features,
stores human decisions in SQLite, and splices accepted audio
back into the master render.

---

## Install

```bash
pip install edge-tts textual sounddevice numpy
# Optional — higher quality F0 extraction:
pip install praat-parselmouth
```

---

## Quickstart

### 1. Create a project and render audio

```bash
# Create project
python -m ui.cli list-projects   # check existing

python -m ui.app --new "My Project" --voice en-US-AriaNeural --text path/to/script.txt
# This creates the project in the DB and opens the TUI.
```

Or do it in two steps:

```bash
# Create project + generate initial audio separately
python - <<EOF
import asyncio, core.database as db
from tts.edge_wrapper import EdgeTTSWrapper

db.init_db()
p = db.create_project("My Project", "en-US-AriaNeural")
wrapper = EdgeTTSWrapper(voice="en-US-AriaNeural", output_dir="data/audio")
result = asyncio.run(wrapper.render("Hello, this is a test sentence."))
db.add_render(p.project_id, result.audio_path, result.duration_ms, result.sample_rate,
              result.word_boundaries_as_dicts())
print(f"Project {p.project_id}, render saved to {result.audio_path}")
EOF
```

### 2. Mark problem regions

```bash
python -m ui.cli add-region \
    --render-id 1 \
    --start 800 --end 1400 \
    --text "trails off here" \
    --context-before 200 --context-after 200
```

### 3. Open the TUI

```bash
python -m ui.app --project-id 1
```

---

## TUI Keybindings

| Key      | Action                                      |
|----------|---------------------------------------------|
| Space    | Play / Pause current audio                  |
| L        | Toggle loop                                 |
| G        | Generate next correction candidate          |
| A        | Accept current attempt                      |
| R        | Reject current attempt                      |
| N        | New region (shows CLI instruction)          |
| ↑ / ↓   | Navigate region list                        |
| Ctrl+S   | Splice accepted attempt into master audio   |
| Q        | Quit                                        |

---

## Workflow

```
1. Load project → waveform appears in timeline
2. Select a region from the region list
3. Press G to generate a correction candidate
4. Press Space to listen
5. Press A to accept or R to reject
   - Add tags (comma-separated) and notes in the bottom inputs
6. If rejected, press G again — strategy avoids already-tried params
7. Once accepted, press Ctrl+S to splice into master
8. Export dataset at any time:
```

```bash
python -m ui.cli export --project-id 1 --out data/dataset.json
```

---

## Project layout

```
tts_tool/
├── core/
│   ├── database.py       SQLite CRUD layer
│   └── models.py         Dataclass models
├── audio/
│   ├── splice.py         Cut / replace / crossfade
│   └── features.py       F0, energy, pause extraction
├── tts/
│   └── edge_wrapper.py   Async edge-tts wrapper
├── strategies/
│   └── ssml_strategy.py  SSML nudging + plugin registry
├── ui/
│   ├── app.py            Textual TUI
│   └── cli.py            CLI helper
├── data/
│   ├── tts_tool.db       SQLite database
│   └── audio/            All audio artifacts (never overwritten)
└── requirements.txt
```

---

## Dataset export format

Each accepted pair in `dataset.json`:

```json
{
  "region_id": 1,
  "render_id": 1,
  "start_ms": 800,
  "end_ms": 1400,
  "context_before_ms": 200,
  "context_after_ms": 200,
  "text_fragment": "trails off here",
  "attempt_id": 3,
  "strategy_name": "ssml_nudge",
  "ssml_params": {"pitch": "+10Hz", "rate": "-5%"},
  "candidate_audio_path": "data/audio/candidate_a003_xxxx.wav",
  "features": {
    "mean_f0": 205.0,
    "f0_std": 18.4,
    "duration_ms": 580,
    "voiced_ratio": 0.94,
    ...
  },
  "tags": ["trails off", "pitch low"],
  "notes": "much better ending"
}
```

Input audio (bad region): slice `start_ms`–`end_ms` from the render's `audio_path`.
Target audio (correction): `candidate_audio_path`.

---

## Adding a custom correction strategy

```python
from strategies.ssml_strategy import register_strategy
from tts.edge_wrapper import EdgeTTSWrapper

class MyStrategy:
    name = "my_strategy"

    def __init__(self, wrapper: EdgeTTSWrapper, text_fragment: str, seed_params=None):
        self.wrapper = wrapper
        self.text_fragment = text_fragment

    async def next_candidate(self, hint=None):
        # your logic here
        ...

register_strategy("my_strategy", MyStrategy)
```
