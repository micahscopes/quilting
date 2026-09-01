from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ADDON_DIR = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = REPOSITORY / "crates" / "hyperscape-protocol" / "fixtures"
sys.path.insert(0, str(ADDON_DIR))

import protocol  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def fixture(self, name: str) -> tuple[str, dict]:
        text = (FIXTURES / name).read_text(encoding="utf-8")
        return text, json.loads(text)

    def test_rust_authored_fixture_roundtrips_canonically(self) -> None:
        text, envelope = self.fixture("authored-set-transform-v0.1.json")
        protocol.validate_authored_envelope(envelope)
        self.assertEqual(protocol.canonical_json(envelope), text)
        constructed = protocol.set_transform_envelope(
            message_id="00000000-0000-0000-0000-000000000001",
            sender="00000000-0000-0000-0000-000000000002",
            sequence=3,
            entity="00000000-0000-0000-0000-000000000004",
            translation=[1, 2, 3],
            rotation_wxyz=[1, 0, 0, 0],
            scale=[1, 1, 1],
            version=protocol.LEGACY_PROTOCOL_VERSION,
        )
        self.assertEqual(constructed, envelope)

    def test_rust_conformal_frame_fixture_roundtrips_canonically(self) -> None:
        text, envelope = self.fixture("authored-set-conformal-frame-v0.2.json")
        protocol.validate_authored_envelope(envelope)
        self.assertEqual(protocol.canonical_json(envelope), text)
        constructed = protocol.set_conformal_frame_transform_envelope(
            message_id="00000000-0000-0000-0000-000000000001",
            sender="00000000-0000-0000-0000-000000000002",
            sequence=3,
            frame="00000000-0000-0000-0000-000000000004",
            generators=envelope["command"]["generators"],
        )
        self.assertEqual(constructed, envelope)

        with self.assertRaisesRegex(
            protocol.HyperscapeProtocolError, "requires protocol 0.2"
        ):
            protocol.set_conformal_frame_transform_envelope(
                message_id="00000000-0000-0000-0000-000000000001",
                sender="00000000-0000-0000-0000-000000000002",
                sequence=3,
                frame="00000000-0000-0000-0000-000000000004",
                generators=[],
                version=protocol.LEGACY_PROTOCOL_VERSION,
            )

        invalid = json.loads(json.dumps(envelope))
        invalid["command"]["generators"] = [
            {"type": "uniform_scale", "factor": 0.0}
        ]
        with self.assertRaisesRegex(
            protocol.HyperscapeProtocolError, "factor must be nonzero"
        ):
            protocol.validate_authored_envelope(invalid)

        oversized = json.loads(json.dumps(envelope))
        oversized["command"]["generators"] = [
            {"type": "translation", "offset": [0.0, 0.0, 0.0]}
        ] * (protocol.MAX_CONFORMAL_GENERATORS_PER_FRAME + 1)
        with self.assertRaisesRegex(
            protocol.HyperscapeProtocolError, "too many generators"
        ):
            protocol.validate_authored_envelope(oversized)

    def test_presence_is_sender_ordered_and_expires_from_local_receipt(self) -> None:
        text, envelope = self.fixture("presence-camera-v0.1.json")
        protocol.validate_presence_envelope(envelope)
        self.assertEqual(protocol.canonical_json(envelope), text)
        inbox = protocol.PresenceInbox()
        self.assertTrue(inbox.accept(envelope, 10.0))
        self.assertFalse(inbox.accept(envelope, 10.1))
        self.assertEqual(inbox.live(11.49), [envelope])
        self.assertEqual(inbox.live(11.5), [])

    def test_authored_echo_guard_is_bounded_and_consuming(self) -> None:
        ids = [f"00000000-0000-0000-0000-{value:012d}" for value in range(1, 4)]
        guard = protocol.AuthoredEchoGuard(capacity=2)
        for message_id in ids:
            guard.record_local(message_id)
        self.assertFalse(guard.consume_echo(ids[0]))
        self.assertTrue(guard.consume_echo(ids[1]))
        self.assertFalse(guard.consume_echo(ids[1]))
        self.assertTrue(guard.consume_echo(ids[2]))

    def test_invalid_version_nil_ids_and_nonfinite_values_are_rejected(self) -> None:
        _, authored = self.fixture("authored-set-transform-v0.1.json")
        authored["header"]["version"]["minor"] = 3
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "version"):
            protocol.validate_authored_envelope(authored)

        _, presence = self.fixture("presence-camera-v0.1.json")
        presence["header"]["sender"] = "00000000-0000-0000-0000-000000000000"
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "nil"):
            protocol.validate_presence_envelope(presence)

        _, presence = self.fixture("presence-camera-v0.1.json")
        presence["presence"]["camera"]["eye"][0] = float("nan")
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "finite"):
            protocol.validate_presence_envelope(presence)

    def test_python_values_follow_json_and_rust_numeric_rules(self) -> None:
        _, authored = self.fixture("authored-set-transform-v0.1.json")
        authored["header"]["version"]["minor"] = True
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "version"):
            protocol.validate_authored_envelope(authored)

        _, authored = self.fixture("authored-set-transform-v0.1.json")
        authored["header"]["sequence"] = 1 << 64
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "64-bit"):
            protocol.validate_authored_envelope(authored)

        _, authored = self.fixture("authored-set-transform-v0.1.json")
        authored["command"]["transform"]["translation"][0] = "1.0"
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "numeric"):
            protocol.validate_authored_envelope(authored)

    def test_nullable_rust_option_fields_are_accepted(self) -> None:
        _, presence = self.fixture("presence-camera-v0.1.json")
        presence["presence"].update(
            {"camera": None, "focus": None, "active_cue": None, "animation_seconds": None}
        )
        protocol.validate_presence_envelope(presence)

    def test_presence_sender_identity_is_uuid_normalized(self) -> None:
        _, first = self.fixture("presence-camera-v0.1.json")
        _, second = self.fixture("presence-camera-v0.1.json")
        first["header"]["sender"] = "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"
        second["header"]["sender"] = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        second["header"]["message_id"] = "00000000-0000-0000-0000-000000000006"
        second["header"]["sequence"] += 1
        inbox = protocol.PresenceInbox()
        self.assertTrue(inbox.accept(first, 1.0))
        self.assertTrue(inbox.accept(second, 1.1))
        self.assertEqual(inbox.live(1.2), [second])

    def test_presence_inbox_matches_rust_policy(self) -> None:
        _, first = self.fixture("presence-camera-v0.1.json")
        inbox = protocol.PresenceInbox(capacity=4)
        inbox.record_local(first)
        self.assertEqual(
            inbox.admit(first, 10.0),
            protocol.PresenceInbox.IGNORED_ECHO,
        )
        self.assertEqual(
            inbox.admit(first, 10.1),
            protocol.PresenceInbox.IGNORED_DUPLICATE,
        )
        self.assertEqual(inbox.live(10.1), [])

        stale = json.loads(json.dumps(first))
        stale["header"]["message_id"] = "00000000-0000-0000-0000-000000000006"
        stale["header"]["sequence"] -= 1
        stale["presence"]["ttl_millis"] = protocol.MAX_PRESENCE_TTL_MILLIS
        self.assertEqual(
            inbox.admit(stale, 10.2),
            protocol.PresenceInbox.IGNORED_STALE,
        )

        newer = json.loads(json.dumps(first))
        newer["header"]["message_id"] = "00000000-0000-0000-0000-000000000007"
        newer["header"]["sequence"] += 1
        self.assertEqual(
            inbox.admit(newer, 10.3),
            protocol.PresenceInbox.APPLIED,
        )
        self.assertEqual(inbox.live(11.79), [newer])
        self.assertEqual(inbox.live(11.8), [])

    def test_local_peer_frame_keeps_authored_and_presence_disjoint(self) -> None:
        _, authored = self.fixture("authored-set-transform-v0.1.json")
        _, presence = self.fixture("presence-camera-v0.1.json")
        self.assertEqual(
            protocol.local_peer_frame("authored", authored),
            {"lane": "authored", "envelope": authored},
        )
        self.assertEqual(
            protocol.local_peer_frame("presence", presence),
            {"lane": "presence", "envelope": presence},
        )
        with self.assertRaises(protocol.HyperscapeProtocolError):
            protocol.validate_local_peer_frame(
                {"lane": "authored", "envelope": presence}
            )
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "lane"):
            protocol.local_peer_frame("durable_presence", presence)

    def test_authored_record_frames_are_opaque_canonical_and_separate(self) -> None:
        project_id = "30000000-0000-4000-8000-000000000001"
        frame = protocol.authored_record_frame(
            project_id=project_id,
            record_base64="AP8Q",
        )
        self.assertEqual(
            frame,
            {
                "lane": "authored_record",
                "version": {"major": 0, "minor": 1},
                "project_id": project_id,
                "record_base64": "AP8Q",
            },
        )
        protocol.validate_local_peer_frame(frame)

        padded = dict(frame, record_base64="AP8Q=")
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "unpadded"):
            protocol.validate_local_peer_frame(padded)

        extra = dict(frame, authority=True)
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "fields"):
            protocol.validate_local_peer_frame(extra)

        wrong_version = dict(frame, version={"major": 0, "minor": 2})
        with self.assertRaisesRegex(protocol.HyperscapeProtocolError, "version"):
            protocol.validate_local_peer_frame(wrong_version)

    def test_presence_constructor_matches_the_checked_in_rust_fixture(self) -> None:
        _, fixture = self.fixture("presence-camera-v0.1.json")
        constructed = protocol.presence_envelope(
            message_id="00000000-0000-0000-0000-000000000001",
            sender="00000000-0000-0000-0000-000000000002",
            sequence=3,
            ttl_millis=1500,
            camera={
                "eye": [0, 0, 3],
                "forward": [0, 0, -1],
                "up": [0, 1, 0],
            },
            selection=["00000000-0000-0000-0000-000000000005"],
            animation_seconds=2,
            version=protocol.LEGACY_PROTOCOL_VERSION,
        )
        self.assertEqual(constructed, fixture)

    def test_authoring_lease_constructor_matches_the_rust_fixture(self) -> None:
        text, fixture = self.fixture("presence-authoring-lease-v0.1.json")
        protocol.validate_presence_envelope(fixture)
        self.assertEqual(protocol.canonical_json(fixture), text)
        constructed = protocol.presence_envelope(
            message_id="00000000-0000-0000-0000-000000000011",
            sender="00000000-0000-0000-0000-000000000012",
            sequence=9,
            ttl_millis=1500,
            authoring_leases=[
                {
                    "lease_id": "00000000-0000-0000-0000-000000000013",
                    "target": {
                        "asset": "00000000-0000-0000-0000-000000000014",
                        "entity": "00000000-0000-0000-0000-000000000015",
                    },
                }
            ],
            version=protocol.LEGACY_PROTOCOL_VERSION,
        )
        self.assertEqual(constructed, fixture)

        duplicate = json.loads(json.dumps(fixture))
        duplicate["presence"]["authoring_leases"].append(
            {
                "lease_id": "00000000-0000-0000-0000-000000000016",
                "target": fixture["presence"]["authoring_leases"][0]["target"],
            }
        )
        with self.assertRaisesRegex(
            protocol.HyperscapeProtocolError, "repeats.*target"
        ):
            protocol.validate_presence_envelope(duplicate)

    def test_authored_inbox_matches_rust_stale_duplicate_and_echo_policy(self) -> None:
        _, first = self.fixture("authored-set-transform-v0.1.json")
        inbox = protocol.AuthoredInbox(capacity=4)
        inbox.record_local(first)
        self.assertEqual(inbox.accept(first), protocol.AuthoredInbox.IGNORED_ECHO)
        self.assertEqual(
            inbox.accept(first),
            protocol.AuthoredInbox.IGNORED_DUPLICATE,
        )

        stale = json.loads(json.dumps(first))
        stale["header"]["message_id"] = "00000000-0000-0000-0000-000000000006"
        stale["header"]["sequence"] -= 1
        self.assertEqual(
            inbox.accept(stale),
            protocol.AuthoredInbox.IGNORED_STALE,
        )

        newer = json.loads(json.dumps(first))
        newer["header"]["message_id"] = "00000000-0000-0000-0000-000000000007"
        newer["header"]["sequence"] += 1
        self.assertEqual(inbox.accept(newer), protocol.AuthoredInbox.APPLIED)


if __name__ == "__main__":
    unittest.main()
