"""
ui/cli.py

Command-line helper for operations that happen outside the TUI:
    - listing projects
    - adding renders (after edge-tts produces audio)
    - adding regions (marking problem spans)
    - exporting the training dataset

Usage:
    python -m ui.cli list-projects
    python -m ui.cli add-render  --project-id 1 --audio data/audio/render.wav
    python -m ui.cli add-region  --render-id 1 --start 800 --end 1400 --text "trails off"
    python -m ui.cli export      --project-id 1 --out data/dataset.json
    python -m ui.cli show        --project-id 1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import core.database as db


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list_projects(args) -> None:
    db.init_db()
    projects = db.list_projects()
    if not projects:
        print("No projects found.")
        return
    print(f"{'ID':>4}  {'Name':<30}  {'Voice':<25}  Created")
    print("─" * 75)
    for p in projects:
        print(f"{p.project_id:>4}  {p.name:<30}  {p.voice_name:<25}  {p.created_at}")


def cmd_show(args) -> None:
    db.init_db()
    project = db.get_project(args.project_id)
    if not project:
        print(f"Project {args.project_id} not found.")
        sys.exit(1)

    print(f"\nProject {project.project_id}: {project.name}")
    print(f"  Voice:   {project.voice_name}")
    print(f"  Created: {project.created_at}")

    renders = db.get_renders_for_project(project.project_id)
    print(f"\n  Renders ({len(renders)}):")
    for r in renders:
        regions = db.get_regions_for_render(r.render_id)
        print(f"    [{r.render_id}] {r.audio_path}  {r.duration_ms}ms  {len(regions)} region(s)")
        for reg in regions:
            attempts = db.get_attempts_for_region(reg.region_id)
            accepted = [
                a for a in attempts
                if (d := db.get_decision_for_attempt(a.attempt_id)) and d.accepted
            ]
            print(
                f"      region {reg.region_id}: {reg.start_ms}–{reg.end_ms}ms  "
                f"{len(attempts)} attempt(s)  {len(accepted)} accepted  "
                f"[{reg.text_fragment or ''}]"
            )


def cmd_add_render(args) -> None:
    db.init_db()
    path = args.audio
    if not Path(path).exists():
        print(f"Audio file not found: {path}")
        sys.exit(1)

    duration_ms  = args.duration
    sample_rate  = args.sample_rate

    if not duration_ms or not sample_rate:
        import wave
        try:
            with wave.open(path, "rb") as wf:
                sample_rate  = wf.getframerate()
                duration_ms  = int(wf.getnframes() / wf.getframerate() * 1000)
        except Exception as e:
            print(f"Could not read WAV header: {e}")
            sys.exit(1)

    render = db.add_render(
        project_id=args.project_id,
        audio_path=path,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
    )
    print(f"Added render {render.render_id}:  {path}  ({duration_ms}ms @ {sample_rate}Hz)")


def cmd_add_region(args) -> None:
    db.init_db()
    region = db.create_region(
        render_id=args.render_id,
        start_ms=args.start,
        end_ms=args.end,
        context_before_ms=args.context_before,
        context_after_ms=args.context_after,
        text_fragment=args.text,
    )
    print(
        f"Added region {region.region_id}:  "
        f"{region.start_ms}–{region.end_ms}ms  "
        f"context ±{region.context_before_ms}/{region.context_after_ms}ms  "
        f"[{region.text_fragment or ''}]"
    )


def cmd_export(args) -> None:
    db.init_db()
    pairs = db.get_accepted_pairs()

    if args.project_id:
        renders = {r.render_id for r in db.get_renders_for_project(args.project_id)}
        pairs   = [p for p in pairs if p["render_id"] in renders]

    out = args.out or "data/dataset.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Exported {len(pairs)} accepted pair(s) → {out}")


def cmd_tags(args) -> None:
    db.init_db()
    tags = db.list_all_tags()
    if not tags:
        print("No tags in vocabulary yet.")
        return
    print("Tag vocabulary:")
    for t in tags:
        print(f"  [{t.tag_id:>3}] {t.name}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def main() -> None:
    db.init_db()

    parser = argparse.ArgumentParser(
        prog="python -m ui.cli",
        description="TTS Correction CLI helper",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list-projects
    sub.add_parser("list-projects", help="List all projects")

    # show
    p_show = sub.add_parser("show", help="Show project details")
    p_show.add_argument("--project-id", type=int, required=True)

    # add-render
    p_render = sub.add_parser("add-render", help="Register an audio render")
    p_render.add_argument("--project-id", type=int, required=True)
    p_render.add_argument("--audio",      type=str, required=True, help="Path to WAV file")
    p_render.add_argument("--duration",   type=int, help="Duration ms (auto-detected from WAV)")
    p_render.add_argument("--sample-rate",type=int, dest="sample_rate")

    # add-region
    p_region = sub.add_parser("add-region", help="Mark a problem region in a render")
    p_region.add_argument("--render-id",      type=int, required=True)
    p_region.add_argument("--start",          type=int, required=True, help="Start ms")
    p_region.add_argument("--end",            type=int, required=True, help="End ms")
    p_region.add_argument("--text",           type=str, default=None,  help="Text fragment")
    p_region.add_argument("--context-before", type=int, default=200,   dest="context_before")
    p_region.add_argument("--context-after",  type=int, default=200,   dest="context_after")

    # export
    p_export = sub.add_parser("export", help="Export accepted pairs as JSON dataset")
    p_export.add_argument("--project-id", type=int, default=None)
    p_export.add_argument("--out",        type=str, default=None, help="Output path")

    # tags
    sub.add_parser("tags", help="List tag vocabulary")

    args = parser.parse_args()

    dispatch = {
        "list-projects": cmd_list_projects,
        "show":          cmd_show,
        "add-render":    cmd_add_render,
        "add-region":    cmd_add_region,
        "export":        cmd_export,
        "tags":          cmd_tags,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
