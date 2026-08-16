from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ADDON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_DIR))

import conformal  # noqa: E402


class ConformalPreviewTests(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected) -> None:
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=10)

    def test_generator_word_and_inverse_roundtrip(self) -> None:
        angle = 0.37
        word = [
            {"type": "translation", "offset": [0.3, -0.5, 0.9]},
            {"type": "rotation", "quaternion_wxyz": [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]},
            {"type": "uniform_scale", "factor": -1.7},
            {"type": "sphere_reflection", "center": [2.0, 0.0, 0.0], "radius": 1.3},
        ]
        point = (0.2, 1.1, -0.4)
        transformed = conformal.apply_word(word, point)
        recovered = conformal.apply_word(conformal.inverse_word(word), transformed)
        self.assertPointAlmostEqual(recovered, point)

    def test_world_words_and_cross_frame_conversion(self) -> None:
        frames = [
            {"name": "world", "parent": None, "generators": []},
            {
                "name": "translated",
                "parent": 0,
                "generators": [{"type": "translation", "offset": [2, 0, 0]}],
            },
            {
                "name": "scaled child",
                "parent": 1,
                "generators": [{"type": "uniform_scale", "factor": 2}],
            },
        ]
        self.assertPointAlmostEqual(conformal.apply_word(conformal.world_word(frames, 2), [1, 0, 0]), [4, 0, 0])
        self.assertPointAlmostEqual(conformal.convert_point(frames, [1, 0, 0], 2, 1), [2, 0, 0])

    def test_preserve_world_reparent_word(self) -> None:
        frames = [
            {"name": "world", "parent": None, "generators": []},
            {
                "name": "left",
                "parent": 0,
                "generators": [{"type": "translation", "offset": [2, 0, 0]}],
            },
            {
                "name": "right",
                "parent": 0,
                "generators": [{"type": "translation", "offset": [0, 3, 0]}],
            },
            {
                "name": "child",
                "parent": 1,
                "generators": [{"type": "uniform_scale", "factor": 2}],
            },
        ]
        point = [0.5, -0.25, 1.0]
        before = conformal.apply_word(conformal.world_word(frames, 3), point)
        new_word = conformal.preserve_world_reparent_word(frames, 3, 2)
        frames[3] = {"name": "child", "parent": 2, "generators": new_word}
        after = conformal.apply_word(conformal.world_word(frames, 3), point)
        self.assertPointAlmostEqual(after, before)

    def test_sphere_pole_and_wall_side_are_explicit(self) -> None:
        reflection = {"type": "sphere_reflection", "center": [1, 2, 3], "radius": 2}
        with self.assertRaisesRegex(conformal.ConformalPreviewError, "pole"):
            conformal.apply_generator(reflection, [1, 2, 3])
        wall = {"geometry": {"type": "sphere", "center": [0, 0, 0], "radius": 2}}
        self.assertEqual(conformal.classify_wall_side(wall, [0, 0, 0]), -1)
        self.assertEqual(conformal.classify_wall_side(wall, [2, 0, 0]), 0)
        self.assertEqual(conformal.classify_wall_side(wall, [3, 0, 0]), 1)


if __name__ == "__main__":
    unittest.main()
