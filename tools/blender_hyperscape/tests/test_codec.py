from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import sys
import unittest


ADDON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_DIR))

import codec  # noqa: E402


def sample_payload(path_node: int = 0) -> dict:
    return {
        "version": "0.1",
        "frames": [
            {"name": "world", "parent": None, "generators": []},
            {
                "name": "room",
                "parent": 0,
                "generators": [
                    {"type": "translation", "offset": [1.0, 0.0, 0.0]},
                    {
                        "type": "sphere_reflection",
                        "center": [0.0, 0.0, 0.0],
                        "radius": 2.0,
                    },
                ],
            },
        ],
        "walls": [
            {
                "name": "round wall",
                "frame": 1,
                "geometry": {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 2.0},
            }
        ],
        "anchors": [{"name": "inside-out", "frame": 1, "flipped_walls": [0]}],
        "paths": [
            {
                "name": "travel",
                "node": path_node,
                "looping": False,
                "keyframes": [
                    {"time_seconds": 0.0, "point": [1.0, 0.0, 0.0]},
                    {"time_seconds": 1.0, "point": [2.0, 0.0, 0.0]},
                ],
            }
        ],
        "constraints": [],
    }


class CodecTests(unittest.TestCase):
    def test_json_roundtrip_preserves_fallback_and_unrelated_extras(self) -> None:
        document = {
            "asset": {"version": "2.0"},
            "extras": {"application": "kept"},
            "nodes": [
                {"name": "subject", "extras": {"ordinary": 17}},
                {
                    "name": "unbound",
                    "extras": {"hyperscape": {"frame": 99}, "ordinary": "also kept"},
                },
            ],
        }
        raw = json.dumps(document).encode()
        encoded = codec.inject_asset(raw, sample_payload(), {0: {"frame": 1, "anchor": 0, "path": 0}})
        decoded, container = codec.decode_gltf(encoded)
        payload, bindings = codec.extract_asset(encoded)

        self.assertFalse(container.is_glb)
        self.assertEqual(decoded["extras"]["application"], "kept")
        self.assertEqual(decoded["nodes"][0]["extras"]["ordinary"], 17)
        self.assertEqual(decoded["nodes"][1]["extras"], {"ordinary": "also kept"})
        self.assertEqual(payload, sample_payload())
        self.assertEqual(bindings, [{"frame": 1, "anchor": 0, "path": 0}, None])

    def test_glb_roundtrip_preserves_non_json_chunks_byte_for_byte(self) -> None:
        document = {"asset": {"version": "2.0"}, "nodes": [{"name": "subject"}]}
        json_chunk = json.dumps(document).encode()
        json_chunk += b" " * ((-len(json_chunk)) % 4)
        private_chunk = b"opaque-private!!"
        private_chunk += b"\x00" * ((-len(private_chunk)) % 4)
        chunks = ((codec.JSON_CHUNK, json_chunk), (0x12345678, private_chunk))
        size = 12 + sum(8 + len(data) for _, data in chunks)
        raw = bytearray(struct.pack("<4sII", b"glTF", 2, size))
        for kind, data in chunks:
            raw.extend(struct.pack("<II", len(data), kind))
            raw.extend(data)

        encoded = codec.inject_asset(bytes(raw), sample_payload(), {0: {"frame": 1, "path": 0}})
        payload, bindings = codec.extract_asset(encoded)
        _, container = codec.decode_gltf(encoded)

        self.assertEqual(payload, sample_payload())
        self.assertEqual(bindings, [{"frame": 1, "path": 0}])
        self.assertEqual(container.chunks[1], (0x12345678, private_chunk))

    def test_sparse_named_node_map_validates_against_real_node_count(self) -> None:
        payload = sample_payload(path_node=2)
        codec.validate_payload(payload, 3)
        with self.assertRaisesRegex(codec.HyperscapeCodecError, "outside"):
            codec.validate_payload(payload, 2)

    def test_unique_node_name_mapping_drops_ambiguous_names(self) -> None:
        document = {
            "nodes": [
                {"name": "unique"},
                {"name": "duplicate"},
                {"name": "duplicate"},
                {},
            ]
        }
        self.assertEqual(codec.unique_node_indices_by_name(document), {"unique": 0})

    def test_validation_rejects_invalid_frame_tree_and_path_times(self) -> None:
        payload = sample_payload()
        payload["frames"][1]["parent"] = 1
        with self.assertRaisesRegex(codec.HyperscapeCodecError, "precede"):
            codec.validate_payload(payload, 1)

        payload = sample_payload()
        payload["paths"][0]["keyframes"][1]["time_seconds"] = 0.0
        with self.assertRaisesRegex(codec.HyperscapeCodecError, "strictly increase"):
            codec.validate_payload(payload, 1)

    def test_path_transitions_validate_stable_chart_frame_anchor_and_order(self) -> None:
        payload = sample_payload()
        payload["paths"][0]["coordinate_frame"] = 0
        payload["paths"][0]["transitions"] = [
            {"time_seconds": 0.5, "frame": 1, "anchor": 0},
            {"time_seconds": 0.75, "frame": 0},
        ]
        codec.validate_payload(payload, 1, [{"frame": 1, "anchor": 0, "path": 0}])

        mismatch = copy.deepcopy(payload)
        mismatch["paths"][0]["transitions"][0]["frame"] = 0
        with self.assertRaisesRegex(codec.HyperscapeCodecError, "anchor frame"):
            codec.validate_payload(mismatch, 1)

        unordered = copy.deepcopy(payload)
        unordered["paths"][0]["transitions"][1]["time_seconds"] = 0.25
        with self.assertRaisesRegex(codec.HyperscapeCodecError, "strictly increase"):
            codec.validate_payload(unordered, 1)

    def test_injection_is_copying_not_aliasing(self) -> None:
        payload = sample_payload()
        encoded = codec.inject_asset(
            b'{"asset":{"version":"2.0"},"nodes":[{}]}',
            payload,
            {0: {"frame": 1, "path": 0}},
        )
        changed = copy.deepcopy(payload)
        changed["frames"][0]["name"] = "mutated"
        extracted, _ = codec.extract_asset(encoded)
        self.assertEqual(extracted["frames"][0]["name"], "world")


if __name__ == "__main__":
    unittest.main()
