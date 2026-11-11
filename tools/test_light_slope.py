import math
import unittest

from light_slope import (
    SLOPE_CONFIG_CLASS,
    checked_grades,
    forward_x_component,
    pitch_degrees,
    pitch_from_root_state,
    root_quat_wxyz,
)


class LightSlopeChecks(unittest.TestCase):
    def test_route_must_match_the_requested_direction_at_every_sample(self):
        self.assertEqual(checked_grades("up", [0, .015, .030], .5), [.03, .03])
        self.assertEqual(checked_grades("down", [0, -.015, -.030], .5), [-.03, -.03])
        with self.assertRaises(ValueError):
            checked_grades("up", [0, -.015, -.030], .5)
        with self.assertRaises(ValueError):
            checked_grades("down", [0, .015, .030], .5)

    def test_pitch_uses_wxyz_and_keeps_its_sign(self):
        for angle in (-12, 12):
            half = math.radians(angle) / 2
            self.assertAlmostEqual(
                pitch_degrees([math.cos(half), 0, math.sin(half), 0]),
                angle,
            )

    def test_forward_axis_uses_wxyz(self):
        self.assertAlmostEqual(forward_x_component([1, 0, 0, 0]), 1.)
        for angle in (-90, 90, 180):
            half = math.radians(angle) / 2
            self.assertAlmostEqual(
                forward_x_component([math.cos(half), 0, 0, math.sin(half)]),
                math.cos(math.radians(angle)),
                places=6,
            )

    def test_root_state_xyzw_is_reordered_before_calculation(self):
        state = [0., 0., 0., 0., 0., 0., 1.]
        self.assertEqual(root_quat_wxyz(state), [1., 0., 0., 0.])
        self.assertAlmostEqual(forward_x_component(root_quat_wxyz(state)), 1.)
        half = math.radians(10.) / 2
        pitched = [0., 0., 0., 0., math.sin(half), 0., math.cos(half)]
        self.assertAlmostEqual(pitch_from_root_state(pitched), 10.)

    def test_down_uses_the_inverted_terrain_configuration(self):
        self.assertEqual(SLOPE_CONFIG_CLASS["up"], "HfPyramidSlopedTerrainCfg")
        self.assertEqual(SLOPE_CONFIG_CLASS["down"], "HfInvertedPyramidSlopedTerrainCfg")


if __name__ == "__main__":
    unittest.main()
