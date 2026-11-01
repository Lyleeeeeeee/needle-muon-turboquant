from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))


def test_diagnostic_rankers_separate_direction_from_length():
    from diagnose_quantization import rank_reconstructions

    queries = np.array([[1., 0.], [0., 0.]])
    decoded = np.array([[.9, 0.], [1., .3], [0., 1.]])
    ip = rank_reconstructions(queries, decoded, 2, 'inner_product')
    cosine = rank_reconstructions(queries, decoded, 2, 'cosine')
    distance = rank_reconstructions(queries, decoded, 2, 'squared_l2')
    np.testing.assert_array_equal(ip, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(cosine, [[0, 1], [0, 1]])
    np.testing.assert_array_equal(distance, [[0, 1], [0, 2]])


def test_diagnostic_rankers_agree_for_unit_vectors_and_stable_ties():
    from diagnose_quantization import rank_reconstructions

    gallery = np.array([[1., 0.], [0., 1.], [1., 0.], [-1., 0.]])
    queries = np.array([[1., 0.], [0., 1.]])
    for metric in ('inner_product', 'cosine', 'squared_l2'):
        result = rank_reconstructions(queries, gallery, 2, metric, batch_size=1)
        np.testing.assert_array_equal(result, [[0, 2], [1, 0]])


def test_independent_quadrature_matches_known_uniform_sphere_case():
    from diagnose_quantization import quadrature_codebook

    centers, distortion = quadrature_codebook(3, 2)
    np.testing.assert_allclose(centers, [-.75, -.25, .25, .75], atol=2e-8)
    assert abs(distortion - 1 / 16) < 1e-10
