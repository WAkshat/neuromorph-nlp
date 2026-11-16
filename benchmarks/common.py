"""Shared helpers for the benchmark scripts: system loading and dataset reading."""

import csv
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
# The upstream "Initial project upload" commit: the unmodified original implementation.
BASELINE_COMMIT = "d80de35"


def load_split(split: str) -> list[dict]:
    with open(DATA / f"{split}.tsv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def load_orchestrator(system: str, **config):
    """Return an orchestrator for `system` ('baseline' or 'current').

    The baseline is imported from a pristine `git archive` of BASELINE_COMMIT, so it
    is always the original code no matter what the working tree looks like. Only one
    system can be loaded per process (both use the top-level package name `agents`).
    """
    if system == "baseline":
        if config:
            raise ValueError("the baseline takes no configuration")
        tar_bytes = subprocess.run(
            ["git", "archive", "--format=tar", BASELINE_COMMIT, "backend"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout
        tmp = tempfile.mkdtemp(prefix="baseline_")
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            tar.extractall(tmp, filter="data")
        sys.path.insert(0, str(Path(tmp) / "backend"))
        from agents.orchestrator import MultiAgentOrchestrator
        return MultiAgentOrchestrator()

    sys.path.insert(0, str(ROOT / "backend"))
    from agents.orchestrator import MultiAgentOrchestrator, PipelineConfig
    return MultiAgentOrchestrator(PipelineConfig(**config))
