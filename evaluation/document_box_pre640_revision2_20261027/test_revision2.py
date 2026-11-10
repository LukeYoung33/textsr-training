"""Independent tail, role and native-rendering mechanism checks."""
import math

import cv2
import numpy as np

from evaluation.document_box_pre640_revision2_20261027 import revision2 as candidate


def test_soft_tails_scalar_interior_and_monotonicity():
    """@luna-comment
    Input: Known40/200rails and dense/scalar Y samples.
    Output: Independent formula, identity interior, compression and slope assertions.
    Key Steps: Compare scalar math.exp reference and finite differences.
    Key Params: Slopes.1/.4,tau2,support24; minimum slope.1.
    State/Dependencies: math/NumPy only, no files or estimator feedback.
    """
    weight = 160**2/(160**2+24**2)
    for slope in [.1, .4]:
        for value in [0., 10., 40., 100., 200., 230., 255.]:
            if value < 40:
                distance = 40-value
                compressed = 40-(slope*distance+(1-slope)*2*(1-math.exp(-distance/2)))
            elif value > 200:
                distance = value-200
                compressed = 200+slope*distance+(1-slope)*2*(1-math.exp(-distance/2))
            else:
                compressed = value
            expected = value+weight*(compressed-value)
            assert abs(float(candidate.curve(value, 40, 200, slope))-expected) < 1e-10
        dense = np.linspace(0, 255, 255001)
        mapped = candidate.curve(dense, 40, 200, slope)
        assert np.min(np.diff(mapped)/.001) >= slope-1e-7
        np.testing.assert_array_equal(mapped[(dense >= 40) & (dense <= 200)], dense[(dense >= 40) & (dense <= 200)])
        assert 200 < float(candidate.curve(230, 40, 200, slope)) < 230
        assert 10 < float(candidate.curve(10, 40, 200, slope)) < 40


def test_background_cluster_preserves_foreground_core_and_polarity():
    """@luna-comment
    Input: Bimodal ink plus paper/halo histogram and its brightness complement.
    Output: Background selects main cluster while foreground keeps conservative core.
    Key Steps: Compare role weights/rails with expected ink5,paper180 and complements.
    Key Params: Paper720/halo180; dark core20/gray80 gives foreground lift trap.
    State/Dependencies: Original cluster_mode via candidate; no real-source tuning.
    """
    histogram = np.zeros(256, np.int64)
    histogram[[5, 30, 180, 220]] = [20, 80, 720, 180]
    model = candidate.rails(histogram)
    assert model['low_core'] == 5 and model['low_cluster'] > 20
    assert model['low'] == 5 and model['light_role'] == 1
    assert abs(model['high']-180) < 1
    reversed_model = candidate.rails(histogram[::-1])
    assert reversed_model['high'] == 250 and reversed_model['dark_role'] == 1
    assert abs(reversed_model['low']-75) < 1


def test_uniform_limit_and_continuous_roles():
    """@luna-comment
    Input: Uniform histogram/color and population values around.5/.75.
    Output: Exact constant-image identity and continuous role endpoints.
    Key Steps: Full estimator/render plus direct cubic limit checks.
    Key Params: Native91x137; both fixed arms.
    State/Dependencies: OpenCV/NumPy, no persistent output.
    """
    image = np.full((91, 137, 3), [15, 101, 242], np.uint8)
    context = candidate.estimate(image)
    for arm in ['LOW', 'STANDARD']:
        output, telemetry = candidate.render(image, context, arm)
        np.testing.assert_array_equal(output, image)
        assert telemetry['native_statistical_pixels'] == 0
    for value in [0, 1]:
        assert candidate.smoothstep(value) == value
    assert abs(candidate.smoothstep(1e-6)) < 1e-10
    assert abs(candidate.smoothstep(1-1e-6)-1) < 1e-10


def test_renderer_chunk_common_delta_and_identity_interior():
    """@luna-comment
    Input: Synthetic two-sided native halo fixture and fixed node tables.
    Output: Chunk identity, common BGR deltas and unchanged interior samples.
    Key Steps: Run inherited renderer with37/128rows and compare exact arrays.
    Key Params: L40/H200,s.1; colors stay in shared gamut.
    State/Dependencies: Mapping only; no estimation, no file writes.
    """
    gray = np.tile(np.array([10, 40, 80, 150, 200, 230], np.uint8), (173, 45))
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    table = candidate.curve(np.arange(256), 40, 200, .1).astype(np.float32)
    packed = dict(nodes=np.array([0, 639]), native_shape=list(gray.shape),
                  tables={'STANDARD': np.broadcast_to(table, (2, 2, 256))})
    output, _ = candidate.render(image, packed, chunk=37)
    other, _ = candidate.render(image, packed, chunk=128)
    np.testing.assert_array_equal(output, other)
    interior = (gray >= 40) & (gray <= 200)
    np.testing.assert_array_equal(output[interior], image[interior])
    assert np.all(output[:, ::6, 0] > 10)
    assert np.all(output[:, 5::6, 0] < 230)
    delta = output.astype(np.int16)-image.astype(np.int16)
    np.testing.assert_array_equal(delta[:, :, 0], delta[:, :, 1])
    np.testing.assert_array_equal(delta[:, :, 1], delta[:, :, 2])
