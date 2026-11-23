import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _find_binary() -> str:
    env = os.environ.get("TESSERA_BIN")
    if env:
        return env
    for candidate in (ROOT / "build" / "tessera-run", ROOT / "build" / "Release" / "tessera-run.exe"):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("tessera-run")
    if found:
        return found
    pytest.skip("tessera-run not built; set TESSERA_BIN or build into ./build")


@pytest.fixture(scope="session")
def tessera():
    binary = _find_binary()

    def run(*args, check=True):
        proc = subprocess.run([binary, *map(str, args)], capture_output=True, text=True, timeout=300)
        if check and proc.returncode != 0:
            raise AssertionError(f"tessera-run {' '.join(map(str, args))} failed:\n{proc.stdout}\n{proc.stderr}")
        return json.loads(proc.stdout) if proc.returncode in (0, 2) else proc

    return run
