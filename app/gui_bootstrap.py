"""Console-free GUI launcher with persistent startup-error logging."""
import os
import sys
import runpy
import traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
LOG = LOGS / "gui_startup.log"

# Unbuffered, line-oriented diagnostics without keeping a command prompt open.
with LOG.open("a", encoding="utf-8", buffering=1) as log:
    log.write(f"\n=== WhisperBox GUI launch {datetime.now().isoformat(timespec='seconds')} ===\n")
    log.flush()
    sys.stdout = log
    sys.stderr = log
    try:
        os.chdir(ROOT)
        runpy.run_path(str(ROOT / "app" / "app.py"), run_name="__main__")
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(file=log)
        raise
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
