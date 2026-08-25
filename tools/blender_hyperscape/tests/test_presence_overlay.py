from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ADDON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_DIR))

import presence_overlay  # noqa: E402


PEER = "81000000-0000-4000-8000-000000000001"
ENTITY = "81000000-0000-4000-8000-000000000002"


def envelope(*, inversion: bool = False) -> dict:
    return {
        "header": {"sender": PEER},
        "presence": {
            "camera": {
                "eye": [2.0, 0.0, 0.0],
                "forward": [-1.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
            },
            "selection": [ENTITY],
            "focus": {
                "center": [0.0, 0.0, 0.0],
                "radius": 1.0,
                "inversion_enabled": inversion,
            },
        },
    }


class PresenceOverlayTests(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected) -> None:
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=10)

    def test_inverted_camera_returns_to_the_ordinary_source_chart(self) -> None:
        sample = envelope(inversion=True)["presence"]
        eye, forward, up = presence_overlay.source_camera_frame(
            sample["camera"], sample["focus"]
        )
        self.assertPointAlmostEqual(eye, [0.5, 0.0, 0.0])
        self.assertPointAlmostEqual(forward, [1.0, 0.0, 0.0])
        self.assertPointAlmostEqual(up, [0.0, 1.0, 0.0])

        output_eye, output_forward, output_up = presence_overlay.source_camera_frame(
            sample["camera"], {**sample["focus"], "inversion_enabled": False}
        )
        self.assertPointAlmostEqual(output_eye, [2.0, 0.0, 0.0])
        self.assertPointAlmostEqual(output_forward, [-1.0, 0.0, 0.0])
        self.assertPointAlmostEqual(output_up, [0.0, 1.0, 0.0])

    def test_inverted_camera_recovers_an_oblique_tangent_frame(self) -> None:
        center = (1.0, -2.0, 0.5)
        radius = 3.0
        source_eye = (2.0, 1.0, 2.0)
        source_forward = (0.0, 0.0, -1.0)
        source_up = (0.0, 1.0, 0.0)

        def reflect(point, direction):
            delta = tuple(point[axis] - center[axis] for axis in range(3))
            norm_squared = sum(component * component for component in delta)
            radial = tuple(component / math.sqrt(norm_squared) for component in delta)
            output_point = tuple(
                center[axis] + radius * radius * delta[axis] / norm_squared
                for axis in range(3)
            )
            radial_dot = sum(
                radial[axis] * direction[axis] for axis in range(3)
            )
            output_direction = tuple(
                direction[axis] - 2.0 * radial[axis] * radial_dot
                for axis in range(3)
            )
            return output_point, output_direction

        output_eye, output_forward = reflect(source_eye, source_forward)
        _, output_up = reflect(source_eye, source_up)
        eye, forward, up = presence_overlay.source_camera_frame(
            {
                "eye": output_eye,
                "forward": output_forward,
                "up": output_up,
            },
            {
                "center": center,
                "radius": radius,
                "inversion_enabled": True,
            },
        )
        self.assertPointAlmostEqual(eye, source_eye)
        self.assertPointAlmostEqual(forward, source_forward)
        self.assertPointAlmostEqual(up, source_up)

    def test_camera_focus_and_selection_build_one_finite_peer_batch(self) -> None:
        selected_segment = ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0))
        batches = presence_overlay.build_overlay_batches(
            [envelope()], {ENTITY: [selected_segment]}
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].peer_id, PEER)
        self.assertEqual(
            batches[0].segments,
            10 + 3 * presence_overlay.FOCUS_RING_SEGMENTS + 1,
        )
        self.assertTrue(
            all(math.isfinite(component) for point in batches[0].positions for component in point)
        )
        self.assertEqual(
            presence_overlay.peer_color(PEER), presence_overlay.peer_color(PEER)
        )

    def test_poles_nil_peers_and_invalid_samples_fail_closed(self) -> None:
        pole = envelope(inversion=True)
        pole["presence"]["camera"]["eye"] = [0.0, 0.0, 0.0]
        nil = envelope()
        nil["header"]["sender"] = "00000000-0000-0000-0000-000000000000"
        malformed = envelope()
        malformed["presence"]["camera"]["forward"] = [0.0, 0.0, 0.0]
        self.assertEqual(
            presence_overlay.build_overlay_batches([pole, nil, malformed]), ()
        )


if __name__ == "__main__":
    unittest.main()
