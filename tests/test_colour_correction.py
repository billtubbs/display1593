import numpy as np
import pytest

from display1593.colour_correction import (
    BRIGHTNESS_STEP,
    N_LEVELS,
    apply_colour_lut,
    colour_lut,
    make_rgb_scales,
)

# The 2016 calibration at brightness setting 0 (old repo's
# data/rgb_scales.npy[0]), red/green/blue.
RGB_SCALES_2016 = np.array(
    [
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5,
         6, 7, 8, 10, 11, 13, 14, 16, 19, 21, 24, 28, 31, 36, 40, 46],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6,
         6, 7, 9, 10, 11, 13, 15, 17, 20, 22, 25, 29, 33, 38, 43, 49],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 7,
         9, 10, 12, 14, 16, 19, 22, 25, 29, 34, 39, 45, 52, 60, 70, 80],
    ]
).T  # fmt: skip


def test_unit_gains_reproduce_2016_calibration():
    scales = make_rgb_scales(gains=(1, 1, 1), brightness=0)
    assert scales.shape == (N_LEVELS, 3)
    np.testing.assert_array_equal(np.rint(scales), RGB_SCALES_2016)


def test_default_gains_dim_red_only():
    default = make_rgb_scales(brightness=0)
    unit = make_rgb_scales(gains=(1, 1, 1), brightness=0)
    np.testing.assert_allclose(default[:, 0], 0.9 * unit[:, 0])
    np.testing.assert_allclose(default[:, 1:], unit[:, 1:])


def test_brightness_scales_by_step():
    s0 = make_rgb_scales(brightness=0)
    np.testing.assert_allclose(
        make_rgb_scales(brightness=3), s0 * BRIGHTNESS_STEP**3
    )
    np.testing.assert_allclose(
        make_rgb_scales(brightness=-2), s0 * BRIGHTNESS_STEP**-2
    )


@pytest.mark.parametrize("brightness", [-3, 0, 2.5, 7])
def test_lut_end_points_and_monotonic(brightness):
    lut = colour_lut(brightness)
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    np.testing.assert_array_equal(lut[0], 0)
    top = np.rint(make_rgb_scales(brightness=brightness)[-1])
    np.testing.assert_array_equal(lut[255], top)
    assert np.all(np.diff(lut.astype(int), axis=0) >= 0)


def test_lut_hits_calibrated_levels():
    lut = colour_lut(0, gains=(1, 1, 1))
    # Intensity 255 * i / 31 is exactly level i; check the levels whose
    # intensity is an integer (i = 0 and 31).
    np.testing.assert_array_equal(lut[0], RGB_SCALES_2016[0])
    np.testing.assert_array_equal(lut[255], RGB_SCALES_2016[-1])


def test_lut_clips_above_255():
    lut = colour_lut(brightness=12)
    assert lut[255, 2] == 255


def test_apply_colour_lut_per_channel():
    lut = colour_lut(5)
    rgb = np.array(
        [[0, 0, 0], [255, 0, 0], [0, 255, 0], [10, 128, 255]],
        dtype=np.uint8,
    )
    out = apply_colour_lut(rgb, lut)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0], 0)
    np.testing.assert_array_equal(out[1], [lut[255, 0], 0, 0])
    np.testing.assert_array_equal(out[2], [0, lut[255, 1], 0])
    np.testing.assert_array_equal(
        out[3], [lut[10, 0], lut[128, 1], lut[255, 2]]
    )


def test_apply_coverage_is_linear():
    from display1593.colour_correction import apply_coverage

    peak = np.array([110, 0, 200], dtype=np.uint8)
    out = apply_coverage(peak, np.array([0.0, 0.25, 0.5, 1.0, 1.5]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(
        out,
        [[0, 0, 0], [28, 0, 50], [55, 0, 100], [110, 0, 200], [110, 0, 200]],
    )


def test_apply_coverage_per_led_colours():
    from display1593.colour_correction import apply_coverage

    pwm = np.array([[100, 0, 0], [0, 100, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(
        apply_coverage(pwm, [0.5, 1.0]), [[50, 0, 0], [0, 100, 0]]
    )


def test_convert_image_corrects_pixels_then_averages():
    from display1593.image_conversion import convert_image

    lut = colour_lut(3)
    # Uniform image: every LED gets the table value for that colour.
    uniform = np.full((256, 256, 3), [255, 128, 0], dtype=np.uint8)
    expected = np.tile(lut[[255, 128, 0], [0, 1, 2]], (1593, 1))
    np.testing.assert_array_equal(convert_image(uniform, lut), expected)
    # Fine black/white stripes: each LED averages PWM values (light),
    # which is more than the PWM value of the average pixel value.
    stripes = np.zeros((256, 256, 3), dtype=np.uint8)
    stripes[:, ::2] = 255
    out = convert_image(stripes, lut).astype(int)
    assert np.all(out[:, 0] > lut[128, 0])
    assert np.all(out[:, 0] <= lut[255, 0])


def test_convert_image_without_lut_unchanged():
    from display1593.image_conversion import convert_image

    uniform = np.full((256, 256, 3), [255, 128, 0], dtype=np.uint8)
    out = convert_image(uniform)
    np.testing.assert_array_equal(out, np.tile([255, 128, 0], (1593, 1)))
