"""
ui/app.py

Main Textual TUI for the TTS prosody correction tool.

Layout:
    ┌─ Header ──────────────────────────────────────────────┐
    │ project · voice · render info          keybinds hint  │
    ├─ Timeline ────────────────────────────────────────────┤
    │ waveform · ruler · region markers                     │
    ├─ RegionList ──────────┬─ IterationPanel ──────────────┤
    │ regions with status   │ attempts + F0 display         │
    ├─ Transport ───────────┴─ Annotations ─────────────────┤
    │ playback controls     │ tags · notes                  │
    └───────────────────────────────────────────────────────┘

Run:
    python -m ui.app --project-id 1
    python -m ui.app --new "My Project" --voice en-US-AriaNeural --text path/to/text.txt
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import (
    Header, Footer, Static, Button, Input,
    Label, ListItem, ListView, RichLog,
)
from textual.widget import Widget
from textual.message import Message
from textual import work

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import core.database as db
from core.models import Project, Render, Region, Attempt, Decision
from audio.splice import load as load_audio, downsample_for_display, AudioBuffer
from audio.features import extract, f0_to_ascii, FeatureSet
from strategies.ssml_strategy import SSMLStrategy, params_summary
from tts.edge_wrapper import EdgeTTSWrapper

# Optional audio playback
try:
    import sounddevice as sd
    import numpy as np
    PLAYBACK_AVAILABLE = True
except ImportError:
    PLAYBACK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Waveform widget
# ---------------------------------------------------------------------------

class WaveformDisplay(Widget):
    """
    Renders a downsampled waveform as Unicode block characters.
    Highlights the currently selected region.
    """

    DEFAULT_CSS = """
    WaveformDisplay {
        height: 6;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
        padding: 0 1;
    }
    """

    bins:        reactive[list]  = reactive([], layout=True)
    region_start: reactive[float] = reactive(0.0)   # 0.0–1.0 fraction
    region_end:   reactive[float] = reactive(0.0)
    cursor_pos:   reactive[float] = reactive(0.0)
    duration_ms:  reactive[int]   = reactive(0)

    BLOCKS = " ▁▂▃▄▅▆▇█"

    def render(self) -> str:
        if not self.bins:
            return "[dim]── no audio loaded ──[/dim]"

        width = self.size.width - 2
        if width < 1:
            return ""

        # Downsample bins to display width
        step = max(1, len(self.bins) // width)
        cols = []
        for i in range(0, len(self.bins), step):
            chunk = self.bins[i:i + step]
            cols.append(max(chunk) if chunk else 0.0)
        cols = cols[:width]

        lines = []
        for row in range(4, 0, -1):
            line = ""
            threshold = row / 4.0
            for x, val in enumerate(cols):
                frac = x / max(len(cols) - 1, 1)
                in_region = self.region_start <= frac <= self.region_end
                at_cursor  = abs(frac - self.cursor_pos) < (1 / max(len(cols), 1))

                if at_cursor:
                    line += "[bold red]│[/bold red]"
                elif in_region:
                    char = self.BLOCKS[min(int(val * 8), 8)] if val >= threshold else " "
                    line += f"[on dark_orange3]{char}[/on dark_orange3]"
                else:
                    char = self.BLOCKS[min(int(val * 8), 8)] if val >= threshold else " "
                    line += f"[cyan]{char}[/cyan]"
            lines.append(line)

        # Ruler
        ruler = self._build_ruler(len(cols))
        lines.append(f"[dim]{ruler}[/dim]")
        return "\n".join(lines)

    def _build_ruler(self, width: int) -> str:
        if self.duration_ms == 0 or width < 10:
            return "─" * width
        ruler = list("─" * width)
        # Place tick marks every ~10% or at even seconds
        step_ms = max(500, (self.duration_ms // 10 // 500) * 500)
        for ms in range(0, self.duration_ms, step_ms):
            x = int(ms / self.duration_ms * width)
            if 0 <= x < width:
                label = f"{ms // 1000}s"
                for i, ch in enumerate(label):
                    if x + i < width:
                        ruler[x + i] = ch
        return "".join(ruler)

    def load(self, bins: list[float], duration_ms: int) -> None:
        self.bins        = bins
        self.duration_ms = duration_ms
        self.region_start = 0.0
        self.region_end   = 0.0
        self.cursor_pos   = 0.0

    def highlight_region(self, start_ms: int, end_ms: int) -> None:
        if self.duration_ms > 0:
            self.region_start = start_ms / self.duration_ms
            self.region_end   = end_ms   / self.duration_ms

    def set_cursor(self, ms: int) -> None:
        if self.duration_ms > 0:
            self.cursor_pos = ms / self.duration_ms


# ---------------------------------------------------------------------------
# Region list item
# ---------------------------------------------------------------------------

class RegionItem(ListItem):
    def __init__(self, region: Region, status: str = "pending") -> None:
        super().__init__()
        self._region = region
        self._status = status

    def compose(self) -> ComposeResult:
        color = {"pending": "yellow", "accepted": "green", "rejected": "red"}.get(
            self._status, "white"
        )
        text = self._region.text_fragment or "(no text)"
        if len(text) > 30:
            text = text[:28] + "…"
        yield Label(
            f"[{color}]●[/{color}] "
            f"[bold]{self._region.start_ms}–{self._region.end_ms}ms[/bold]  "
            f"[dim]{text}[/dim]"
        )


# ---------------------------------------------------------------------------
# Attempt item
# ---------------------------------------------------------------------------

class AttemptItem(ListItem):
    def __init__(self, attempt: Attempt, decision: Optional[Decision] = None) -> None:
        super().__init__()
        self.attempt  = attempt
        self._decision = decision

    def compose(self) -> ComposeResult:
        if self.decision is None:
            status = "[dim]unreviewed[/dim]"
        elif self._decision.accepted:
            status = "[green bold]✓ accepted[/green bold]"
        else:
            status = "[red]✗ rejected[/red]"

        summary = params_summary(self._attempt.ssml_params())
        yield Label(
            f"[bold]#{self._attempt.attempt_id}[/bold]  "
            f"[cyan]{summary}[/cyan]  {status}"
        )



# ---------------------------------------------------------------------------
# ListItem accessor helpers (avoids reserved property name conflicts)
# ---------------------------------------------------------------------------

def _get_region(item) -> "Region":
    return item._region

def _get_attempt(item) -> "Attempt":
    return item._attempt


# ---------------------------------------------------------------------------
# F0 display panel
# ---------------------------------------------------------------------------

class F0Panel(Static):
    DEFAULT_CSS = """
    F0Panel {
        height: 12;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
        padding: 1;
        color: $text-muted;
    }
    """

    def show(self, features: FeatureSet) -> None:
        plot = f0_to_ascii(features.f0_contour, width=50, height=6)
        stats = (
            f"F0 mean=[cyan]{features.mean_f0:.0f}Hz[/cyan]  "
            f"std=[cyan]{features.f0_std:.1f}[/cyan]  "
            f"voiced=[cyan]{features.voiced_ratio:.0%}[/cyan]  "
            f"energy=[cyan]{features.energy_mean:.3f}[/cyan]  "
            f"dur=[cyan]{features.duration_ms}ms[/cyan]"
        )
        self.update(f"{plot}\n{stats}")

    def clear(self) -> None:
        self.update("[dim]── select an attempt to see F0 contour ──[/dim]")


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class TTSCorrectionApp(App):

    CSS = """
    Screen {
        layout: vertical;
    }

    #timeline-panel {
        height: 8;
        border: solid $primary-darken-1;
        padding: 0;
    }

    #main-row {
        layout: horizontal;
        height: 1fr;
    }

    #left-col {
        width: 35%;
        border-right: solid $primary-darken-2;
        layout: vertical;
    }

    #right-col {
        width: 65%;
        layout: vertical;
    }

    #region-list-header {
        background: $primary-darken-3;
        color: $text;
        padding: 0 1;
        height: 1;
    }

    #iteration-header {
        background: $primary-darken-3;
        color: $text;
        padding: 0 1;
        height: 1;
    }

    #region-list {
        height: 1fr;
        border-bottom: solid $primary-darken-2;
    }

    #attempt-list {
        height: 8;
        border-bottom: solid $primary-darken-2;
    }

    #f0-panel {
        height: 14;
    }

    #bottom-row {
        height: 7;
        layout: horizontal;
        border-top: solid $primary-darken-1;
    }

    #transport {
        width: 40%;
        layout: horizontal;
        padding: 1;
        align: left middle;
        border-right: solid $primary-darken-2;
    }

    #annotations {
        width: 60%;
        layout: vertical;
        padding: 0 1;
    }

    #tag-input {
        height: 3;
    }

    #notes-input {
        height: 3;
    }

    Button {
        margin: 0 1;
        min-width: 8;
    }

    Button.accept {
        background: $success-darken-1;
    }

    Button.reject {
        background: $error-darken-1;
    }

    Button.generate {
        background: $primary-darken-1;
    }

    #status-bar {
        height: 1;
        background: $primary-darken-3;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("space",     "play_pause",       "Play/Pause",    show=True),
        Binding("l",         "toggle_loop",       "Loop",         show=True),
        Binding("a",         "accept_attempt",    "Accept",       show=True),
        Binding("r",         "reject_attempt",    "Reject",       show=True),
        Binding("g",         "generate_next",     "Generate",     show=True),
        Binding("n",         "new_region",        "New Region",   show=True),
        Binding("up",        "prev_region",       "Prev Region",  show=False),
        Binding("down",      "next_region",       "Next Region",  show=False),
        Binding("ctrl+s",    "save_splice",       "Splice",       show=True),
        Binding("q",         "quit",              "Quit",         show=True),
    ]

    # Reactive state
    current_project:  reactive[Optional[Project]] = reactive(None)
    current_render:   reactive[Optional[Render]]  = reactive(None)
    current_region:   reactive[Optional[Region]]  = reactive(None)
    current_attempt:  reactive[Optional[Attempt]] = reactive(None)
    is_playing:       reactive[bool]              = reactive(False)
    is_looping:       reactive[bool]              = reactive(False)
    status_message:   reactive[str]               = reactive("Ready.")

    def __init__(
        self,
        project_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._project_id   = project_id
        self._audio_buffer: Optional[AudioBuffer] = None
        self._strategy:     Optional[SSMLStrategy] = None
        self._wrapper:      Optional[EdgeTTSWrapper] = None
        self._playback_task = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Timeline
        with Vertical(id="timeline-panel"):
            yield Label("[bold] Timeline[/bold]", id="timeline-label")
            yield WaveformDisplay(id="waveform")

        # Main content row
        with Horizontal(id="main-row"):

            # Left — region list
            with Vertical(id="left-col"):
                yield Label(" Regions", id="region-list-header")
                yield ListView(id="region-list")

            # Right — iteration panel
            with Vertical(id="right-col"):
                yield Label(" Attempts", id="iteration-header")
                with ScrollableContainer():
                    yield ListView(id="attempt-list")
                yield F0Panel(id="f0-panel")
                with Horizontal(id="attempt-buttons"):
                    yield Button("Generate [G]", id="btn-generate", classes="generate")
                    yield Button("Accept [A]",   id="btn-accept",   classes="accept")
                    yield Button("Reject [R]",   id="btn-reject",   classes="reject")

        # Bottom row — transport + annotations
        with Horizontal(id="bottom-row"):
            with Horizontal(id="transport"):
                yield Button("▶ Play",  id="btn-play")
                yield Button("⟳ Loop",  id="btn-loop")
                yield Button("◀ Prev",  id="btn-prev")
                yield Button("▶ Next",  id="btn-next")
                yield Button("✂ Splice [^S]", id="btn-splice")

            with Vertical(id="annotations"):
                yield Input(
                    placeholder="Tags  (comma-separated: trails off, pitch low, …)",
                    id="tag-input",
                )
                yield Input(
                    placeholder="Notes (free text)",
                    id="notes-input",
                )

        yield Label("", id="status-bar")
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        db.init_db()
        if self._project_id:
            self._load_project(self._project_id)
        else:
            self.set_status("No project loaded. Use --project-id N or --new.")

    def _load_project(self, project_id: int) -> None:
        project = db.get_project(project_id)
        if not project:
            self.set_status(f"[red]Project {project_id} not found.[/red]")
            return
        self.current_project = project
        self._wrapper = EdgeTTSWrapper(
            voice=project.voice_name,
            output_dir="data/audio",
        )
        self.title = f"TTS Correction — {project.name}"
        renders = db.get_renders_for_project(project_id)
        if renders:
            self._load_render(renders[-1])
        self.set_status(f"Project: [bold]{project.name}[/bold]  voice: [cyan]{project.voice_name}[/cyan]")

    def _load_render(self, render: Render) -> None:
        self.current_render = render
        if os.path.exists(render.audio_path):
            try:
                self._audio_buffer = load_audio(render.audio_path)
                bins = downsample_for_display(self._audio_buffer, bins=1500)
                wf = self.query_one("#waveform", WaveformDisplay)
                wf.load(bins, self._audio_buffer.duration_ms)
            except Exception as e:
                self.set_status(f"[yellow]Waveform load failed: {e}[/yellow]")
        self._refresh_region_list()

    def _refresh_region_list(self) -> None:
        if not self.current_render:
            return
        lv = self.query_one("#region-list", ListView)
        lv.clear()
        regions = db.get_regions_for_render(self.current_render.render_id)
        for region in regions:
            attempts  = db.get_attempts_for_region(region.region_id)
            accepted  = any(
                db.get_decision_for_attempt(a.attempt_id) and
                db.get_decision_for_attempt(a.attempt_id).accepted
                for a in attempts
            )
            rejected_all = attempts and all(
                db.get_decision_for_attempt(a.attempt_id) and
                not db.get_decision_for_attempt(a.attempt_id).accepted
                for a in attempts
            )
            status = "accepted" if accepted else ("rejected" if rejected_all else "pending")
            lv.append(RegionItem(region, status))

    def _refresh_attempt_list(self, region: Region) -> None:
        lv = self.query_one("#attempt-list", ListView)
        lv.clear()
        attempts = db.get_attempts_for_region(region.region_id)
        for attempt in attempts:
            decision = db.get_decision_for_attempt(attempt.attempt_id)
            lv.append(AttemptItem(attempt, decision))

        f0_panel = self.query_one("#f0-panel", F0Panel)
        f0_panel.clear()

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, RegionItem):
            self._select_region(_get_region(event.item))
        elif isinstance(event.item, AttemptItem):
            self._select_attempt(_get_attempt(event.item))

    def _select_region(self, region: Region) -> None:
        self.current_region  = region
        self.current_attempt = None

        wf = self.query_one("#waveform", WaveformDisplay)
        wf.highlight_region(region.start_ms, region.end_ms)

        self._refresh_attempt_list(region)

        fragment = region.text_fragment or ""
        if self.current_project and self._wrapper:
            self._strategy = SSMLStrategy(
                self._wrapper,
                fragment,
            )

        self.set_status(
            f"Region [{region.region_id}]  "
            f"{region.start_ms}–{region.end_ms}ms  "
            f"[dim]{fragment[:40]}[/dim]"
        )

    def _select_attempt(self, attempt: Attempt) -> None:
        self.current_attempt = attempt
        f0_panel = self.query_one("#f0-panel", F0Panel)
        features_dict = attempt.features()
        if features_dict:
            from audio.features import FeatureSet
            fs = FeatureSet.from_dict(features_dict)
            f0_panel.show(fs)
        else:
            f0_panel.clear()

        decision = db.get_decision_for_attempt(attempt.attempt_id)
        tags = db.get_tags_for_decision(decision.decision_id) if decision else []
        tag_input = self.query_one("#tag-input", Input)
        tag_input.value = ", ".join(t.name for t in tags)
        notes_input = self.query_one("#notes-input", Input)
        notes_input.value = decision.notes or "" if decision else ""

        self.set_status(
            f"Attempt #{attempt.attempt_id}  "
            f"strategy=[cyan]{attempt.strategy_name}[/cyan]  "
            f"params=[cyan]{params_summary(attempt.ssml_params())}[/cyan]"
        )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def action_play_pause(self) -> None:
        if not PLAYBACK_AVAILABLE:
            self.set_status("[yellow]sounddevice not installed — playback unavailable.[/yellow]")
            return
        if self.is_playing:
            sd.stop()
            self.is_playing = False
            self.set_status("Stopped.")
        else:
            self._play_current()

    def _play_current(self) -> None:
        """Play current attempt audio, or region audio if no attempt selected."""
        path = None
        if self.current_attempt and self.current_attempt.candidate_audio_path:
            path = self.current_attempt.candidate_audio_path
        elif self._audio_buffer and self.current_region:
            self._play_region_from_master()
            return

        if not path or not os.path.exists(path):
            self.set_status("[yellow]No audio file to play.[/yellow]")
            return
        self._play_path(path)

    def _play_path(self, path: str) -> None:
        try:
            buf = load_audio(path)
            arr = np.array(buf.samples, dtype=np.int16)
            sd.play(arr, samplerate=buf.sample_rate)
            self.is_playing = True
            self.set_status(f"[green]Playing:[/green] {os.path.basename(path)}")
        except Exception as e:
            self.set_status(f"[red]Playback error: {e}[/red]")

    def _play_region_from_master(self) -> None:
        if not self._audio_buffer or not self.current_region:
            return
        region = self.current_region
        sliced = self._audio_buffer.slice_ms(region.start_ms, region.end_ms)
        try:
            arr = np.array(sliced.samples, dtype=np.int16)
            sd.play(arr, samplerate=sliced.sample_rate)
            self.is_playing = True
            self.set_status(f"[green]Playing region[/green] {region.start_ms}–{region.end_ms}ms")
        except Exception as e:
            self.set_status(f"[red]Playback error: {e}[/red]")

    def action_toggle_loop(self) -> None:
        self.is_looping = not self.is_looping
        self.set_status(f"Loop: {'[green]ON[/green]' if self.is_looping else '[dim]off[/dim]'}")

    # ------------------------------------------------------------------
    # Region navigation
    # ------------------------------------------------------------------

    def action_prev_region(self) -> None:
        lv = self.query_one("#region-list", ListView)
        lv.action_cursor_up()

    def action_next_region(self) -> None:
        lv = self.query_one("#region-list", ListView)
        lv.action_cursor_down()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def action_generate_next(self) -> None:
        if not self.current_region:
            self.set_status("[yellow]Select a region first.[/yellow]")
            return
        if not self._strategy:
            self.set_status("[yellow]No strategy initialised.[/yellow]")
            return
        self._run_generate()

    @work(exclusive=False, thread=False)
    async def _run_generate(self) -> None:
        region = self.current_region
        self.set_status("[cyan]Generating candidate…[/cyan]")
        try:
            candidate = await self._strategy.next_candidate()

            # Extract features
            audio = load_audio(candidate.audio_path)
            fs    = extract(audio)

            # Store attempt in DB
            attempt = db.add_attempt(
                region_id=region.region_id,
                strategy_name=candidate.strategy,
                candidate_audio_path=candidate.audio_path,
                ssml_params=candidate.ssml_params,
                features=fs.to_dict(),
            )

            self._refresh_attempt_list(region)
            self._refresh_region_list()
            self.current_attempt = attempt
            self.set_status(
                f"[green]Generated attempt #{attempt.attempt_id}[/green]  "
                f"params=[cyan]{params_summary(candidate.ssml_params)}[/cyan]"
            )

            # Auto-play new candidate
            if PLAYBACK_AVAILABLE:
                self._play_path(candidate.audio_path)

        except Exception as e:
            self.set_status(f"[red]Generation failed: {e}[/red]")

    # ------------------------------------------------------------------
    # Accept / Reject
    # ------------------------------------------------------------------

    def action_accept_attempt(self) -> None:
        if not self.current_attempt:
            self.set_status("[yellow]Select an attempt first.[/yellow]")
            return
        self._record_decision(accepted=True)

    def action_reject_attempt(self) -> None:
        if not self.current_attempt:
            self.set_status("[yellow]Select an attempt first.[/yellow]")
            return
        self._record_decision(accepted=False)

    def _record_decision(self, accepted: bool) -> None:
        attempt = self.current_attempt
        notes   = self.query_one("#notes-input", Input).value.strip() or None
        tags    = [
            t.strip()
            for t in self.query_one("#tag-input", Input).value.split(",")
            if t.strip()
        ]

        decision = db.record_decision(attempt.attempt_id, accepted=accepted, notes=notes)
        for tag in tags:
            db.tag_decision(decision.decision_id, tag)

        if accepted and self._strategy:
            self._strategy.record_acceptance(attempt.ssml_params())
        elif not accepted and self._strategy:
            self._strategy.record_rejection(attempt.ssml_params())

        self._refresh_attempt_list(self.current_region)
        self._refresh_region_list()

        verb = "[green]Accepted[/green]" if accepted else "[red]Rejected[/red]"
        self.set_status(f"{verb} attempt #{attempt.attempt_id}  tags: {tags}")

    # ------------------------------------------------------------------
    # Splice
    # ------------------------------------------------------------------

    def action_save_splice(self) -> None:
        if not self.current_attempt or not self.current_region:
            self.set_status("[yellow]Select an accepted attempt to splice.[/yellow]")
            return
        decision = db.get_decision_for_attempt(self.current_attempt.attempt_id)
        if not decision or not decision.accepted:
            self.set_status("[yellow]Attempt must be accepted before splicing.[/yellow]")
            return
        self._run_splice()

    @work(exclusive=False, thread=False)
    async def _run_splice(self) -> None:
        region  = self.current_region
        attempt = self.current_attempt
        render  = self.current_render

        self.set_status("[cyan]Splicing…[/cyan]")
        try:
            from audio.splice import load, replace_region, normalize_for_splice, save
            master      = load(render.audio_path)
            replacement = load(attempt.candidate_audio_path)
            compatible  = normalize_for_splice(replacement, master)
            spliced     = replace_region(
                master,
                region.start_ms,
                region.end_ms,
                compatible,
                fade_ms=20,
            )

            out_dir  = Path("data/audio/spliced")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / f"splice_r{region.region_id}_a{attempt.attempt_id}.wav")
            save(spliced, out_path)

            db.record_splice(region.region_id, attempt.attempt_id, out_path)

            self.set_status(
                f"[green]Spliced →[/green] {out_path}  "
                f"({spliced.duration_ms}ms)"
            )

        except Exception as e:
            self.set_status(f"[red]Splice failed: {e}[/red]")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "btn-play":     self.action_play_pause,
            "btn-loop":     self.action_toggle_loop,
            "btn-prev":     self.action_prev_region,
            "btn-next":     self.action_next_region,
            "btn-generate": self.action_generate_next,
            "btn-accept":   self.action_accept_attempt,
            "btn-reject":   self.action_reject_attempt,
            "btn-splice":   self.action_save_splice,
        }
        handler = actions.get(event.button.id)
        if handler:
            handler()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_status(self, message: str) -> None:
        self.status_message = message
        try:
            self.query_one("#status-bar", Label).update(message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# New region helper (modal-less, uses input bar)
# ---------------------------------------------------------------------------

    def action_new_region(self) -> None:
        """
        Prompt via status bar convention.
        Full modal region-creation dialog is a future enhancement.
        For now, instructs the user to use the CLI helper.
        """
        self.set_status(
            "[yellow]New region:[/yellow] use  "
            "[bold]python -m ui.cli add-region --render-id N --start MS --end MS[/bold]"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="TTS Prosody Correction TUI")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--project-id", type=int, help="Load existing project by ID")
    group.add_argument("--new",        type=str, help="Create new project with given name")
    parser.add_argument("--voice",     type=str, default="en-US-AriaNeural")
    parser.add_argument("--text",      type=str, help="Path to source text file (for --new)")
    args = parser.parse_args()

    db.init_db()

    project_id = args.project_id

    if args.new:
        project = db.create_project(args.new, args.voice)
        if args.text:
            db.add_source_text(project.project_id, args.text)
        project_id = project.project_id
        print(f"Created project {project_id}: {args.new}")

    app = TTSCorrectionApp(project_id=project_id)
    app.run()


if __name__ == "__main__":
    main()
