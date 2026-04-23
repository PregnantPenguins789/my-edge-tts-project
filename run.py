#!/usr/bin/env python3
"""
run.py — launcher for the TTS correction tool.

    python run.py --new "My Project" --voice en-US-AriaNeural
    python run.py --project-id 1
    python run.py cli list-projects
    python run.py cli add-region --render-id 1 --start 800 --end 1400 --text "trails off"
    python run.py cli export --project-id 1
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Set DB path explicitly so all entry points use the same file
os.environ["TTS_TOOL_ROOT"] = HERE

if len(sys.argv) > 1 and sys.argv[1] == "cli":
    sys.argv.pop(1)
    from ui.cli import main
    main()
else:
    from ui.app import main
    main()
