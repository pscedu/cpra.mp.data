"""Tests for Legend.colorize_pixels, which turns raster pixel values into RGBA."""

import numpy as np
import polars as pl
import pytest

from cpra.mp.data.legends import LEGENDS, Legend

RED = "#ff0000"
GREEN = "#00ff00"
BLUE = "#0000ff"


def discrete_legend():
    """Three classes: < 10 red, 10 to < 20 green, >= 20 blue."""
    return Legend(
        "test",
        pl.DataFrame({"color": [RED, GREEN, BLUE], "upper_bound": [10.0, 20.0, None]}),
        type="discrete",
    )


def continuous_legend():
    """Red at 0 interpolating to blue at 10.

    Unlike a discrete legend, a continuous one has no trailing null bound: every color
    needs a position to interpolate from, so np.interp requires len(colors) == len(breaks).
    Dataset.legend enforces the same rule when parsing a `color_map` query parameter.
    """
    return Legend(
        "test",
        pl.DataFrame({"color": [RED, BLUE], "upper_bound": [0.0, 10.0]}),
        type="continuous",
    )


def pixels(values):
    """Shape pixel values the way Dataset._collapse_tile_bands delivers them: (bands, h, w)."""
    return np.array([[values]], dtype="float64")


class TestDiscrete:
    def test_classifies_by_upper_bound(self):
        result = discrete_legend().colorize_pixels(pixels([5, 15, 25]))
        assert result[0, 0].tolist() == [255, 0, 0, 255]
        assert result[0, 1].tolist() == [0, 255, 0, 255]
        assert result[0, 2].tolist() == [0, 0, 255, 255]

    def test_upper_bound_is_exclusive(self):
        """np.digitize places a value equal to a break into the next class up."""
        result = discrete_legend().colorize_pixels(pixels([9.99, 10.0]))
        assert result[0, 0].tolist() == [255, 0, 0, 255]
        assert result[0, 1].tolist() == [0, 255, 0, 255]

    def test_values_above_the_last_break_use_the_open_class(self):
        result = discrete_legend().colorize_pixels(pixels([1e9]))
        assert result[0, 0].tolist() == [0, 0, 255, 255]

    def test_output_dtype_is_uint8(self):
        assert discrete_legend().colorize_pixels(pixels([5])).dtype == np.uint8

    def test_eight_digit_hex_alpha_is_preserved(self):
        legend = Legend(
            "test",
            pl.DataFrame({"color": ["#0000007f", "#ffffffff"], "upper_bound": [1.0, None]}),
            type="discrete",
        )
        result = legend.colorize_pixels(pixels([0, 5]))
        assert result[0, 0].tolist() == [0, 0, 0, 127]
        assert result[0, 1].tolist() == [255, 255, 255, 255]

    def test_six_digit_hex_gets_opaque_alpha(self):
        result = discrete_legend().colorize_pixels(pixels([5]))
        assert result[0, 0][3] == 255


class TestContinuous:
    def test_interpolates_between_breaks(self):
        """Halfway from red to blue is (127, 0, 127), alpha unchanged."""
        result = continuous_legend().colorize_pixels(pixels([5.0]))
        assert result[0, 0].tolist() == [127, 0, 127, 255]

    def test_endpoints_are_exact(self):
        result = continuous_legend().colorize_pixels(pixels([0.0, 10.0]))
        assert result[0, 0].tolist() == [255, 0, 0, 255]
        assert result[0, 1].tolist() == [0, 0, 255, 255]

    def test_values_are_clamped_outside_the_break_range(self):
        """np.interp holds the endpoint values rather than extrapolating."""
        result = continuous_legend().colorize_pixels(pixels([-100.0, 100.0]))
        assert result[0, 0].tolist() == [255, 0, 0, 255]
        assert result[0, 1].tolist() == [0, 0, 255, 255]

    def test_output_dtype_is_uint8(self):
        assert continuous_legend().colorize_pixels(pixels([5.0])).dtype == np.uint8

    def test_alpha_channel_is_interpolated_too(self):
        """All four channels are stacked, so a fading alpha interpolates with the color."""
        legend = Legend(
            "test",
            pl.DataFrame({"color": ["#ff000000", "#ff0000ff"], "upper_bound": [0.0, 10.0]}),
            type="continuous",
        )
        result = legend.colorize_pixels(pixels([0.0, 5.0, 10.0]))
        assert result.shape[-1] == 4
        assert [row[3] for row in result[0].tolist()] == [0, 127, 255]


class TestRegistry:
    def test_default_legend_exists(self):
        """Dataset.legend falls back to LEGENDS["default"], so it must be present."""
        assert "default" in LEGENDS

    def test_legend_names_match_their_keys(self):
        for name, legend in LEGENDS.items():
            assert legend.name == name

    def test_legends_default_to_discrete(self):
        assert all(legend.type == "discrete" for legend in LEGENDS.values())

    @pytest.mark.parametrize("name", sorted(LEGENDS))
    def test_every_registry_legend_colorizes(self, name):
        """Every shipped legend must survive colorize_pixels without raising."""
        result = LEGENDS[name].colorize_pixels(pixels([0.0, 1.0, 1e6]))
        assert result.shape == (1, 3, 4)

    @pytest.mark.parametrize("name", sorted(LEGENDS))
    def test_the_last_class_covers_large_values(self, name):
        """Regression: a leading null upper_bound shifted every class by one, so the
        open-ended top class was unreachable and the final color never rendered."""
        legend = LEGENDS[name]
        result = legend.colorize_pixels(pixels([1e12]))
        last = legend.break_points["color"].to_list()[-1].lstrip("#")
        expected = [int(last[i : i + 2], 16) for i in (0, 2, 4)]
        assert result[0, 0].tolist()[:3] == expected
