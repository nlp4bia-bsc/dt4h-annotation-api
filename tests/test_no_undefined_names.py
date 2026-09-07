"""Static guard against undefined names reaching runtime.

``FuzzyMatchPipeline.predict`` and ``BM25OkapiPipeline.predict`` called
``ner_inference``, a name the module never imported. Both raised ``NameError``
on any call, and nothing noticed — neither is reachable over HTTP, because
``/process_bulk`` hardcodes ``method = 'biencoder'``.

Unit tests would not have caught it: exercising those pipelines needs real NER
checkpoints, which this suite deliberately does not download. A static check
catches the whole class of bug in the unreachable code as cheaply as in the
hot path, which is exactly where it is needed.

Scoped to F821 (undefined name) on purpose. This is a correctness guard, not a
style gate; broadening it would make the suite fail on formatting opinions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["app", "scripts", "tests", "run_nerl.py", "test_init.py"]


def ruff_command() -> list[str] | None:
    """Prefer an installed ruff; fall back to uvx, else skip."""
    if shutil.which("ruff"):
        return ["ruff"]
    if shutil.which("uvx"):
        return ["uvx", "ruff"]
    return None


def test_no_undefined_names():
    command = ruff_command()
    if command is None:
        pytest.skip("ruff is not available (install it, or run via uv)")

    existing = [target for target in TARGETS if (REPO_ROOT / target).exists()]
    result = subprocess.run(
        [*command, "check", "--select", "F821", "--no-cache", *existing],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(
            "ruff found undefined name(s) — a NameError waiting to happen:\n\n"
            f"{result.stdout}\n{result.stderr}"
        )


def test_every_pipeline_predict_resolves_its_calls():
    """Import the pipelines module and confirm the repaired names exist.

    A direct check that does not depend on ruff being installed, so the
    specific regression stays covered even when the linter check skips.
    """
    from app.src import pipelines

    for name in ("encoder_inference", "fuzzymatch_inference", "bm25okapi_inference",
                 "lookup_inference", "biencoder_inference", "join_all_entities"):
        assert hasattr(pipelines, name), f"pipelines.py uses {name} but does not define it"

    assert not hasattr(pipelines, "ner_inference"), (
        "pipelines.py should call encoder_inference; ner_inference was the undefined name"
    )
