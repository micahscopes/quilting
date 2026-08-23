from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ADDON_DIR = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = REPOSITORY / "fixtures" / "protocol"
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
        )
        self.assertEqual(constructed, envelope)

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
        authored["header"]["version"]["minor"] = 2
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
        second["header"]["sequence"] += 1
        inbox = protocol.PresenceInbox()
        self.assertTrue(inbox.accept(first, 1.0))
        self.assertTrue(inbox.accept(second, 1.1))
        self.assertEqual(inbox.live(1.2), [second])


if __name__ == "__main__":
    unittest.main()
