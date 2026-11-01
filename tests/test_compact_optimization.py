"""Independent native Compact copy semantics and paired measurement checks."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from needle.backend_ndarray import NDArray, cpu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))


LAYOUTS = [
    ((), (), 3),
    ((0, 3), (3, 1), 2),
    ((3, 2), (1, 3), 0),
    ((4, 3), (0, 1), 2),
    ((2, 3), (-3, -1), 8),
    ((3, 4), (9, 2), 4),
    ((2, 1, 3, 1, 4, 2), (24, 0, 8, 0, 2, 1), 2),
    ((16, 16, 8, 8), (1024, 8, 128, 1), 0),
    ((16, 8, 16, 8), (1024, 8, 64, 1), 0),
]


def check_layout(shape, strides, offset):
    maximum = offset + sum(max(0, (n - 1) * s) for n, s in zip(shape, strides))
    bits = np.random.default_rng(13).integers(0, 2**32, max(maximum + 1, 16), dtype='uint32')
    bits[:6] = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x7fa05678]
    source = bits.view('float32')
    expected = np.ndarray(shape, dtype='float32', buffer=source,
                          offset=offset * 4, strides=tuple(s * 4 for s in strides))
    device = cpu()
    native = NDArray(source, device=device)
    output = NDArray.make(shape, device=device)
    device.compact(native._handle, output._handle, shape, strides, offset)
    actual_bits = np.ascontiguousarray(output.numpy()).reshape(-1).view('uint32')
    expected_bits = np.ascontiguousarray(expected).reshape(-1).view('uint32')
    np.testing.assert_array_equal(actual_bits, expected_bits)
    np.testing.assert_array_equal(native.numpy().view('uint32'), bits)


@pytest.mark.parametrize('shape,strides,offset', LAYOUTS)
def test_native_compact_preserves_bits_and_source(shape, strides, offset):
    check_layout(shape, strides, offset)


def test_random_valid_strided_layouts_match_numpy():
    rng = np.random.default_rng(29)
    for _ in range(100):
        rank = int(rng.integers(1, 7))
        shape = tuple(int(n) for n in rng.integers(1, 5, rank))
        strides = tuple(int(n) for n in rng.integers(-15, 16, rank))
        offset = 2 + sum(max(0, -(n - 1) * s) for n, s in zip(shape, strides))
        check_layout(shape, strides, offset)


def test_paired_speed_summary_uses_pair_ratios_and_checks_output_parity():
    from common import summarize_pairs

    rows = [dict(block=block, variant=variant,
                 metrics={'case': dict(seconds=seconds, output_sha256='same')})
            for block, values in enumerate(((.004, .001), (.006, .003)))
            for variant, seconds in zip(('old', 'new'), values)]
    summary = summarize_pairs(rows)
    assert summary['case']['speedup']['values'] == [4., 2.]
    assert summary['case']['speedup']['mean'] == 3.
    assert summary['case']['speedup']['sample_std'] == pytest.approx(math.sqrt(2))
    rows[-1]['metrics']['case']['output_sha256'] = 'changed'
    with pytest.raises(ValueError, match='output'):
        summarize_pairs(rows)


def test_paired_speed_summary_rejects_missing_and_duplicate_variants():
    from common import summarize_pairs

    row = dict(block=0, variant='old', metrics={'x': dict(seconds=.1, output_sha256='a')})
    for rows in ([row], [row, row]):
        with pytest.raises(ValueError, match='pair'):
            summarize_pairs(rows)
