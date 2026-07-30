"""Run bookkeeping: seeding, manifests, result serialisation.

Every result file records the seed, the code version, and the config that
produced it. The design is large enough that an untracked run is a worthless
run -- with eight cells, three seeds and a battery of instruments, "which
version produced this number" stops being answerable from memory very quickly.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int):
    """Seed every source of randomness we touch, and return a torch Generator.

    The generator is returned rather than relied on globally so that analysis
    functions can be handed an explicit, reproducible stream.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def git_sha(default="unknown"):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return os.environ.get("CALIBRATION_GIT_SHA", default)


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


@dataclass
class RunManifest:
    """Everything needed to say what produced a number."""

    experiment: str
    config: dict
    seeds: list
    git_sha: str = field(default_factory=git_sha)
    started_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    finished_at: str | None = None
    model_id: str | None = None
    device: str | None = None
    torch_version: str = field(default_factory=lambda: torch.__version__)
    python_version: str = field(default_factory=platform.python_version)
    notes: str = ""

    @property
    def config_hash(self):
        return config_hash(self.config)

    @property
    def run_id(self):
        return f"{self.started_at.replace(':', '').replace('-', '')}_{self.config_hash}"

    def finish(self):
        self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self


def save_results(out_dir, manifest: RunManifest, results: dict):
    """Write manifest + results as one JSON, named by run id."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": asdict(manifest),
        "run_id": manifest.run_id,
        "config_hash": manifest.config_hash,
        "results": results,
    }
    path = out / f"{manifest.run_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    return path


def _jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


def summarise(values):
    """Mean and a seed-level bootstrap CI.

    Reported instead of a bare mean because the whole power argument for this
    design is that a number without a spread across seeds is not a measurement.
    With few seeds the CI is wide -- which is the honest signal, not a defect.
    """
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    mean = t.mean(0)
    if t.shape[0] < 2:
        nan = torch.full_like(mean, float("nan"))
        return mean, nan, nan
    g = torch.Generator().manual_seed(0)
    boots = torch.stack(
        [
            t[torch.randint(t.shape[0], (t.shape[0],), generator=g)].mean(0)
            for _ in range(2000)
        ]
    )
    lo = boots.quantile(0.025, dim=0)
    hi = boots.quantile(0.975, dim=0)
    return mean, lo, hi
