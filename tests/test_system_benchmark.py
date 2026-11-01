"""A total-backend ablation must compare distinct binaries and identical work."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))


def paired_records():
    return [dict(block=b, variant=v, native_extension_sha256=v,
                 capabilities={'muon_ns_tiled': v == 'new', 'matmul_strided': v == 'new'},
                 metrics={'training_adamw': dict(seconds=t, output_sha256='same',
                                                 initial_parameter_sha256='init')})
            for b in range(2) for v, t in (('old', .004), ('new', .002))]


def test_total_ablation_checks_distinct_native_versions_and_output_parity():
    from benchmark_system import validate_measurements

    rows = paired_records()
    assert validate_measurements(rows)['training_adamw']['speedup']['median'] == 2.
    for field in ('binary', 'capabilities', 'initial_state'):
        changed = copy.deepcopy(rows)
        if field == 'binary': changed[1]['native_extension_sha256'] = 'old'
        if field == 'capabilities': changed[1]['capabilities']['muon_ns_tiled'] = False
        if field == 'initial_state': changed[1]['metrics']['training_adamw']['initial_parameter_sha256'] = 'different'
        with pytest.raises(ValueError):
            validate_measurements(changed)
