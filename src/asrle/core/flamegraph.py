from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class FlamegraphConfig:
    out_path: str = "runs/latest/flame.svg"
    rate: int = 100
    format: str = "svg"  # svg or speedscope (if supported)
    native: bool = False


def run_py_spy_record(args_for_asrle: list[str], cfg: FlamegraphConfig) -> str:
    """
    Spawns a new python process under py-spy to record a flamegraph.

    Example:
      asrle flamegraph audio.wav --backend hf-whisper --out runs/x
    This will run:
      py-spy record -o runs/latest/flame.svg -- python -m asrle.cli.app analyze ...
    """
    os.makedirs(os.path.dirname(cfg.out_path) or ".", exist_ok=True)
    cmd = [
        "py-spy",
        "record",
        "-o",
        cfg.out_path,
        "--rate",
        str(cfg.rate),
    ]
    if cfg.native:
        cmd.append("--native")

    cmd += ["--", sys.executable, "-m", "asrle.cli.app"] + args_for_asrle
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("py-spy not found. Install extras: pip install -e '.[profile]'") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"py-spy failed: {e}") from e

    return cfg.out_path


def pretty_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)
