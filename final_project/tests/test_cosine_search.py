import numpy as np
import pytest

import needle as ndl
from needle.quantization import FloatVectorStore, TurboQuantMSE, UniformQuantizer


@pytest.mark.parametrize('device_name', ['cpu', 'cpu_numpy'])
@pytest.mark.parametrize('kind', ['fp32', 'uniform', 'turbo'])
def test_cosine_matches_independent_decoded_reference(device_name, kind):
    rng = np.random.default_rng(57)
    gallery = rng.normal(size=(37, 9)).astype('float32')
    gallery[0] = 0
    queries = rng.normal(size=(4, 9)).astype('float32') * np.array([1e30, 1e-30, 1, 1])[:, None]
    queries[3] = 0
    queries = queries.astype('float32')
    device = getattr(ndl, device_name)()
    if kind == 'fp32':
        store = FloatVectorStore(gallery, device=device)
    else:
        quantizer = (UniformQuantizer(9, 4, device=device) if kind == 'uniform'
                     else TurboQuantMSE(9, 4, seed=8, device=device))
        store = quantizer.encode(gallery)
    decoded = store.decode_block(0, len(gallery)).numpy().astype('float64')
    q = queries.astype('float64')
    dnorm = np.linalg.norm(decoded, axis=1, keepdims=True)
    qnorm = np.linalg.norm(q, axis=1, keepdims=True)
    d = np.divide(decoded, dnorm, out=np.zeros_like(decoded), where=dnorm != 0)
    q = np.divide(q, qnorm, out=np.zeros_like(q), where=qnorm != 0)
    scores = q @ d.T
    expected = np.argsort(-scores, axis=1, kind='stable')[:, :5]
    bytes_before = store.nbytes
    first = store.search(queries, k=5, block_size=7, metric='cosine')
    second = store.search(queries, k=5, block_size=11, metric='cosine')
    np.testing.assert_array_equal(first.indices, expected)
    np.testing.assert_array_equal(second.indices, expected)
    np.testing.assert_allclose(first.scores, np.take_along_axis(scores, expected, axis=1), atol=3e-6)
    assert np.isfinite(first.scores).all()
    assert store.nbytes == bytes_before
    assert first.max_decoded_block_bytes <= 7 * 9 * 4


def test_cosine_handles_extreme_gallery_norms_and_zero_ties():
    gallery = np.array([[1e30, 1e30, 0], [1e-30, 1e-30, 0], [0, 0, 0], [0, 1e30, 1e30]], dtype='float32')
    queries = np.array([[1e30, 1e30, 0], [0, 0, 0]], dtype='float32')
    result = FloatVectorStore(gallery, device=ndl.cpu()).search(queries, k=4, block_size=2, metric='cosine')
    np.testing.assert_array_equal(result.indices, [[0, 1, 3, 2], [0, 1, 2, 3]])
    np.testing.assert_allclose(result.scores, [[1, 1, .5, 0], [0, 0, 0, 0]], atol=2e-6)


def test_explicit_inner_product_matches_default_and_invalid_metric_is_rejected():
    values = np.random.default_rng(29).normal(size=(19, 8)).astype('float32')
    store = TurboQuantMSE(8, 2, seed=4, device=ndl.cpu()).encode(values)
    legacy = store.search(values[:3], k=5)
    explicit = store.search(values[:3], k=5, metric='inner_product')
    np.testing.assert_array_equal(legacy.indices, explicit.indices)
    np.testing.assert_array_equal(legacy.scores, explicit.scores)
    with pytest.raises(ValueError, match='metric'):
        store.search(values[:3], metric='unknown')
