"""Numerical and storage contracts for Algorithm 1 of TurboQuant."""
import math

import numpy as np
import pytest

import needle as ndl


@pytest.mark.parametrize("bits", [2, 4])
def test_sphere_codebook_matches_uniform_distribution_in_three_dimensions(bits):
    from needle.quantization import sphere_codebook
    levels = 1 << bits
    expected = -1 + (2 * np.arange(levels) + 1) / levels
    np.testing.assert_allclose(sphere_codebook(3, bits), expected, atol=2e-7)


def test_codebook_centroids_match_independent_quadrature():
    from scipy.integrate import quad
    from needle.quantization import sphere_codebook
    dim = 64
    centers = sphere_codebook(dim, 2)
    edges = np.r_[-1., (centers[:-1] + centers[1:]) / 2, 1.]
    density = lambda x: (1 - x*x) ** ((dim - 3) / 2)
    for lo, hi, centroid in zip(edges[:-1], edges[1:], centers):
        mass = quad(density, lo, hi, epsabs=1e-12)[0]
        moment = quad(lambda x: x * density(x), lo, hi, epsabs=1e-12)[0]
        assert abs(centroid - moment / mass) < 2e-7


def test_rotation_is_orthogonal_and_seeded():
    from needle.quantization import TurboQuantMSE
    first = TurboQuantMSE(17, 4, seed=9)
    second = TurboQuantMSE(17, 4, seed=9)
    rotation = first.rotation.numpy()
    np.testing.assert_array_equal(rotation, second.rotation.numpy())
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(17), atol=2e-6)


@pytest.mark.parametrize("device_name", ["cpu_numpy", "cpu"])
@pytest.mark.parametrize("bits", [2, 4])
def test_encode_and_decode_match_numpy_reference(device_name, bits):
    from needle.quantization import TurboQuantMSE
    rng = np.random.default_rng(3)
    data = rng.normal(size=(13, 9)).astype('float32')
    data[0] = 0
    quantizer = TurboQuantMSE(9, bits, seed=4, device=getattr(ndl, device_name)())
    store = quantizer.encode(data)
    norms = np.linalg.norm(data.astype('float64'), axis=1).astype('float32')
    unit = np.divide(data, norms[:, None], out=np.zeros_like(data), where=norms[:, None] != 0)
    rotated = unit @ quantizer.rotation.numpy()
    distances = abs(rotated[:, :, None] - quantizer.codebook)
    indices = distances.argmin(axis=2)
    expected = (quantizer.codebook[indices] * norms[:, None]) @ quantizer.rotation.numpy().T
    np.testing.assert_allclose(store.decode_block(0, 13).numpy(), expected, atol=2e-6)
    np.testing.assert_array_equal(store.decode_block(0, 1).numpy(), np.zeros((1, 9)))
    assert store.packed.dtype == np.uint8
    assert store.packed.shape == (13, math.ceil(9 * bits / 8))
    assert store.storage_bytes == {
        'codes': store.packed.nbytes, 'norms': norms.nbytes,
        'rotation': 9 * 9 * 4, 'codebook': (1 << bits) * 4,
    }
    assert store.nbytes == sum(store.storage_bytes.values())
    assert not any(isinstance(x, np.ndarray) and x.shape == data.shape for x in vars(store).values())


def test_native_bit_packing_has_known_layout_and_zero_padding():
    from needle.backend_ndarray import ndarray_backend_cpu as native
    codes = np.array([[0, 1, 2, 3, 1], [3, 2, 1, 0, 2]], dtype='uint8')
    packed = native.pack_codes(codes, 2)
    np.testing.assert_array_equal(packed, np.array([[228, 1], [27, 2]], dtype='uint8'))
    with pytest.raises(ValueError):
        native.pack_codes(np.array([[4]], dtype='uint8'), 2)


@pytest.mark.parametrize("bits,limit", [(2, .14), (4, .015)])
def test_random_sphere_reconstruction_error_is_reasonable(bits, limit):
    from needle.quantization import TurboQuantMSE
    rng = np.random.default_rng(15)
    data = rng.normal(size=(2048, 64)).astype('float32')
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    store = TurboQuantMSE(64, bits, seed=8).encode(data)
    error = np.sum((data - store.decode_block(0, len(data)).numpy()) ** 2, axis=1).mean()
    assert error < limit, error


@pytest.mark.parametrize("kind", ["turbo", "uniform", "fp32"])
def test_blockwise_search_matches_search_on_reconstructed_vectors(kind):
    from needle.quantization import TurboQuantMSE, UniformQuantizer, FloatVectorStore
    rng = np.random.default_rng(40)
    data = rng.normal(size=(37, 16)).astype('float32')
    queries = rng.normal(size=(4, 16)).astype('float32')
    if kind == 'turbo':
        store = TurboQuantMSE(16, 4, seed=6).encode(data)
    elif kind == 'uniform':
        store = UniformQuantizer(16, 4).encode(data)
    else:
        store = FloatVectorStore(data)
    reconstructed = store.decode_block(0, len(data)).numpy()
    expected_scores = queries @ reconstructed.T
    expected_ids = np.argsort(-expected_scores, axis=1, kind='stable')[:, :5]
    blocks = []
    original = store.decode_block
    def checked(start, stop, **kwargs):
        blocks.append(stop-start)
        assert stop-start <= 7
        return original(start, stop, **kwargs)
    store.decode_block = checked
    result = store.search(queries, k=5, block_size=7)
    np.testing.assert_array_equal(result.indices, expected_ids)
    np.testing.assert_allclose(result.scores, np.take_along_axis(expected_scores, expected_ids, axis=1), atol=2e-5)
    assert len(blocks) == math.ceil(len(data) / 7)
    assert result.max_decoded_block_bytes <= 7 * 16 * 4
    assert result.max_score_block_bytes <= 4 * 7 * 4


def test_search_ties_and_uniform_constant_vectors():
    from needle.quantization import UniformQuantizer
    store = UniformQuantizer(5, 2).encode(np.ones((11, 5), dtype='float32'))
    np.testing.assert_array_equal(store.decode_block(0, 11).numpy(), np.ones((11, 5)))
    first = store.search(np.zeros((2, 5), dtype='float32'), k=4, block_size=3)
    second = store.search(np.zeros((2, 5), dtype='float32'), k=4, block_size=11)
    np.testing.assert_array_equal(first.indices, np.tile(np.arange(4), (2, 1)))
    np.testing.assert_array_equal(first.indices, second.indices)
    assert store.nbytes == 11 * math.ceil(5 * 2 / 8) + 11 * 8


def test_invalid_quantization_inputs_are_rejected():
    from needle.quantization import TurboQuantMSE
    with pytest.raises(ValueError):
        TurboQuantMSE(1, 2)
    with pytest.raises(ValueError):
        TurboQuantMSE(8, 3)
    q = TurboQuantMSE(8, 2)
    for values in [np.zeros((0, 8)), np.zeros((2, 7)), np.full((2, 8), np.nan)]:
        with pytest.raises(ValueError):
            q.encode(values)
    store = q.encode(np.zeros((10, 8)))
    with pytest.raises(ValueError):
        store.search(np.ones((1, 7)), k=2)
    with pytest.raises(ValueError):
        store.search(np.ones((1, 8)), k=11)
    with pytest.raises(ValueError):
        store.search(np.ones((1, 8)), block_size=0)


def test_packed_store_serialization_round_trip(tmp_path):
    from needle.quantization import TurboQuantMSE, PackedVectorStore
    data = np.random.default_rng(5).normal(size=(21, 13)).astype('float32')
    first = TurboQuantMSE(13, 4, seed=6).encode(data)
    path = tmp_path / 'gallery.npz'
    first.save(path)
    second = PackedVectorStore.load(path)
    np.testing.assert_array_equal(first.packed, second.packed)
    np.testing.assert_allclose(first.decode_block(0, 21).numpy(), second.decode_block(0, 21).numpy(), atol=1e-6)
    assert first.nbytes == second.nbytes


def test_loading_malformed_norm_shape_raises_value_error(tmp_path):
    from needle.quantization import PackedVectorStore, sphere_codebook
    path = tmp_path / 'bad.npz'
    np.savez(path, format_version=np.int32(1), bits=np.int32(2),
             packed=np.zeros((1, 1), dtype='uint8'), norms=np.float32(1),
             rotation=np.eye(3, dtype='float32'), codebook=sphere_codebook(3, 2))
    with pytest.raises(ValueError):
        PackedVectorStore.load(path)
