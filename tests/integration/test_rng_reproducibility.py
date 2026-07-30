"""Portable subprocess integration tests for the Issue #8 RNG contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
_ROOT = Path(__file__).resolve().parents[2]


def _run_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )


def test_repeated_subprocess_execution_is_identical() -> None:
    script = """
import json, random
import numpy as np
from expertforge.config.resolve import resolve_config
from expertforge.rng import RngManager
config = resolve_config('configs/smoke.yaml').config
manager = RngManager.from_config(config, component='integration')
manager.initialize()
print(json.dumps({
    'python': [random.random() for _ in range(5)],
    'legacy': np.random.random(5).tolist(),
    'owned': manager.generator.random(5).tolist(),
    'streams': [seed.digest for seed in manager.initialization.derived_seeds],
}, sort_keys=True, separators=(',', ':')))
"""

    first = _run_script(script).stdout
    second = _run_script(script).stdout

    assert first == second


def test_different_root_seed_changes_subprocess_output() -> None:
    script = """
import json, random, sys
import numpy as np
from expertforge.rng import RngManager, SeedContext
manager = RngManager(root_seed=int(sys.argv[1]), context=SeedContext(component='integration'))
manager.initialize()
print(json.dumps([
    random.random(),
    float(np.random.random()),
    float(manager.generator.random()),
], separators=(',', ':')))
"""

    def run(seed: int) -> str:
        return subprocess.run(
            [sys.executable, "-c", script, str(seed)],
            cwd=_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        ).stdout

    assert run(7) != run(8)


def test_checkpoint_style_json_state_restores_next_samples() -> None:
    script = """
import json, random
import numpy as np
from expertforge.rng import RngManager, RngStateBundle, SeedContext
manager = RngManager(root_seed=31, context=SeedContext(component='integration.resume'))
manager.initialize()
for _ in range(9):
    random.random(); np.random.random(); manager.generator.random()
payload = manager.capture_state().to_deterministic_json()
expected = [random.random(), float(np.random.random()), float(manager.generator.random())]
restored = RngManager(root_seed=31, context=SeedContext(component='integration.resume'))
restored.restore_state(RngStateBundle.from_json_bytes(payload))
actual = [random.random(), float(np.random.random()), float(restored.generator.random())]
print(json.dumps({'equal': expected == actual, 'payload': payload.decode('utf-8')}, sort_keys=True))
"""

    result = json.loads(_run_script(script).stdout)

    assert result["equal"] is True
    assert '"rng_state_schema_version": 1' in result["payload"].replace(":", ": ")


def test_importing_rng_package_does_not_mutate_random_state() -> None:
    script = """
import json, random
import numpy as np
random.seed(123)
np.random.seed(456)
python_before = random.getstate()
numpy_before = np.random.get_state()
import expertforge.rng
python_after = random.getstate()
numpy_after = np.random.get_state()
print(json.dumps({
    'python_equal': python_before == python_after,
    'numpy_equal': (
        numpy_before[0] == numpy_after[0]
        and np.array_equal(numpy_before[1], numpy_after[1])
        and numpy_before[2:] == numpy_after[2:]
    ),
    'torch_imported': 'torch' in __import__('sys').modules,
}, sort_keys=True))
"""

    result = json.loads(_run_script(script).stdout)

    assert result == {
        "numpy_equal": True,
        "python_equal": True,
        "torch_imported": False,
    }
