"""Independent numerical contracts for the isolated 640 document candidate."""
import cv2
import numpy as np

from evaluation.document_box_pre640_astra_20261027 import pre640 as candidate


def test_scalar_curve_and_faint_counterexample():
    """@luna-comment
    Input: Known40/173.2/220 rails and dense scalar samples.
    Output: Assertions for analytic values, exterior identity and adverse faint contrast.
    Key Steps: Independent scalar calculation and finite differences.
    Key Params: Alpha.75/gamma1.8; lower derivative bound.25, tolerance1e-7.
    State/Dependencies: NumPy; no files or image-dependent estimation.
    """
    y = np.linspace(0, 255, 255001)
    mapped = candidate.curve(y, 40., 173.2, 220., .75)
    assert np.min(np.diff(mapped)/.001) >= .25-1e-7
    np.testing.assert_array_equal(mapped[(y <= 40) | (y >= 220)], y[(y <= 40) | (y >= 220)])
    expected = .25*130+.75*(40+133.2*(90/133.2)**1.8)
    assert abs(candidate.curve(130., 40., 173.2, 220., .75)-expected) < 1e-10
    faint = float(candidate.curve(200., 40., 173.2, 220., .75))
    assert 0 < 220-faint < 20  # Explicit loss, never a safety claim.


def test_binary_population_midpoint_and_complement():
    """@luna-comment
    Input: Two exact binary histograms with opposite occupancy.
    Output: Assertions on class-core anchors and continuous T placement.
    Key Steps: Partition by Otsu, compare against count-derived scalar midpoint.
    Key Params: Counts100/900; low40/high220; T173.2 or86.8.
    State/Dependencies: NumPy only; does not use Otsu threshold as T.
    """
    for low_count, high_count in [(100, 900), (900, 100)]:
        hist = np.zeros(256, np.int64)
        hist[40], hist[220] = low_count, high_count
        low, mid, high, fraction, flat = candidate.otsu_anchors(hist)
        assert (low, high, flat) == (40, 220, False)
        assert abs(mid-(40+(.2+.6*high_count/1000)*180)) < 1e-10
        assert fraction == high_count/1000


def test_constant_and_identity_full_pipeline():
    """@luna-comment
    Input: Non-gray constant color and deterministic random native BGR.
    Output: Byte-exact constant and zero-strength identities.
    Key Steps: Full640 estimation then native rendering; verify counters/geometry.
    Key Params: Native97x151; all estimation remains640-square.
    State/Dependencies: OpenCV/NumPy candidate only, no persistent files.
    """
    constant = np.full((97, 151, 3), [15, 101, 242], np.uint8)
    context = candidate.estimate(constant)
    output, telemetry = candidate.render(constant, context)
    np.testing.assert_array_equal(output, constant)
    random = np.random.default_rng(7).integers(0, 256, (97, 151, 3), dtype=np.uint8)
    context = candidate.estimate(random)
    output, telemetry = candidate.render(random, context, strength=0)
    np.testing.assert_array_equal(output, random)
    assert context['analysis_shape'] == [640, 640]
    assert telemetry['native_statistical_pixels'] == 0


def test_native_scalar_interpolation_gamut_and_chunks():
    """@luna-comment
    Input: Synthetic two-axis LUT field and random native uint8 BGR.
    Output: Exact chunk agreement and scalar pixel correspondence within1code.
    Key Steps: Independent bilinear formula at selected native pixel centers.
    Key Params: End nodes0/639, chunks37/128; common-delta gamut clamp.
    State/Dependencies: No estimation, so geometry is tested independently of histogram code.
    """
    image = np.random.default_rng(12).integers(0, 256, (173, 257, 3), dtype=np.uint8)
    nodes = np.array([0, 639], np.int32)
    values = np.arange(256, dtype=np.float32)
    tables = np.stack([values+q for q in [-19., 7., 31., -3.]]).reshape(2, 2, 256)
    context = dict(nodes=nodes, tables=tables, native_shape=[173, 257])
    result, _ = candidate.render(image, context, chunk=37)
    other, _ = candidate.render(image, context, chunk=128)
    np.testing.assert_array_equal(result, other)
    difference = result.astype(np.int16)-image.astype(np.int16)
    np.testing.assert_array_equal(difference[:, :, 0], difference[:, :, 1])
    np.testing.assert_array_equal(difference[:, :, 1], difference[:, :, 2])
    for row, col in [(0, 0), (172, 256), (90, 127), (1, 200)]:
        fx = np.clip((col+.5)*640/257-.5, 0, 639)/639
        fy = np.clip((row+.5)*640/173-.5, 0, 639)/639
        delta = round((1-fy)*((1-fx)*-19+fx*7)+fy*((1-fx)*31+fx*-3))
        pixel = image[row, col].astype(int)
        delta = np.clip(delta, -pixel.min(), 255-pixel.max())
        np.testing.assert_array_equal(result[row, col], pixel+delta)


def test_haar_noise_scale_and_dense_lut_slope():
    """@luna-comment
    Input: IID Gaussian grayscale640 and estimated node tables.
    Output: Noise-proxy calibration and minimum slope bounds.
    Key Steps: Full estimate; compare interior median sigma and every LUT increment.
    Key Params: Input sigma5, tolerance15%; alpha<=.75.
    State/Dependencies: Deterministic RNG22; OpenCV Y, no native noise claim.
    """
    gray = np.rint(np.clip(np.random.default_rng(22).normal(128, 5, (640, 640)), 0, 255)).astype(np.uint8)
    context = candidate.estimate(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert abs(float(np.median(context['sigma'][5:-5, 5:-5]))-5) < .75
    assert np.min(np.diff(context['tables'], axis=2)) >= .25-1e-4
    assert np.max(context['alpha']) <= .75


def test_render_does_not_estimate_hidden_native_marks():
    """@luna-comment
    Input: Fixed constant640 context and native content changed after estimation.
    Output: Exact original hidden marks and unchanged context arrays.
    Key Steps: Reuse same LUTs; alter native content without recomputing analysis.
    Key Params: Onepixel dark/bright/color marks; constant LUTs are identity.
    State/Dependencies: Confirms renderer has no native statistics, not detection recall.
    """
    image = np.full((960, 1280, 3), 128, np.uint8)
    context = candidate.estimate(image)
    before = context['tables'].copy()
    image[137, 193] = [0, 0, 0]
    image[139, 195] = [255, 255, 255]
    image[141, 197] = [0, 120, 255]
    output, _ = candidate.render(image, context)
    np.testing.assert_array_equal(output, image)
    np.testing.assert_array_equal(context['tables'], before)


def test_nonidentity_common_exterior_and_last_node_interval():
    """@luna-comment
    Input: Four nonidentity LUTs and real boundary-inclusive analysis nodes.
    Output: Exact common-exterior0/255 and last-interval scalar assertions.
    Key Steps: Render extreme native pixels; independently bracket638/639 coordinates.
    Key Params: Last nodes632/639, seven-pixel interval; four different rails.
    State/Dependencies: NumPy, no image statistics or files.
    """
    values = np.arange(256)
    tables = np.stack([candidate.curve(values, low, (low+high)/2, high, .7)
                       for low, high in [(20, 180), (40, 200), (60, 220), (80, 240)]]).astype(np.float32).reshape(2, 2, 256)
    image = np.zeros((9, 11, 3), np.uint8)
    image[:, 5:] = 255
    context = dict(nodes=np.array([0, 639]), tables=tables, native_shape=[9, 11])
    output, _ = candidate.render(image, context)
    np.testing.assert_array_equal(output, image)
    nodes = np.r_[np.arange(0, 640, 8), 639]
    left, right, weight = candidate.axis_map(640, nodes)
    assert nodes[left[638]] == 632 and nodes[right[638]] == 639
    assert abs(weight[638]-6/7) < 1e-7
    assert nodes[right[639]] == 639 and weight[639] == 1
