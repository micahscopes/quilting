from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from urllib.parse import parse_qs, urlparse


ADDON_DIR = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = REPOSITORY / "fixtures" / "protocol"
sys.path.insert(0, str(ADDON_DIR))

import protocol  # noqa: E402
import relay  # noqa: E402


def fixture_frame(lane: str) -> dict:
    name = (
        "authored-set-transform-v0.1.json"
        if lane == "authored"
        else "presence-camera-v0.1.json"
    )
    envelope = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return protocol.local_peer_frame(lane, envelope)


def batch(
    *,
    generation: str = "generation-a",
    requested_after: int = 0,
    resume_after: int = 0,
    latest: int = 0,
    gap: bool = False,
    has_more: bool = False,
    deliveries: list[tuple[int, dict]] | None = None,
) -> dict:
    frames = [
        {
            "cursor": str(cursor),
            "frameJson": json.dumps(frame, separators=(",", ":"), allow_nan=False),
        }
        for cursor, frame in (deliveries or [])
    ]
    oldest = frames[0]["cursor"] if frames else None
    return {
        "generation": generation,
        "requestedAfter": str(requested_after),
        "resumeAfter": str(resume_after),
        "oldestCursor": oldest,
        "latestCursor": str(latest),
        "gap": gap,
        "hasMore": has_more,
        "frames": frames,
    }


class RelayTransportTests(unittest.TestCase):
    def transport(self, **options) -> relay.LocalRelayTransport:
        return relay.LocalRelayTransport(
            "http://127.0.0.1:42117",
            "test-token",
            **options,
        )

    def test_configuration_rejects_ambiguous_origins_and_credentials(self) -> None:
        invalid_urls = [
            "",
            "ws://127.0.0.1:42117",
            "http://user@127.0.0.1:42117",
            "http://127.0.0.1:42117/path",
            "http://127.0.0.1:42117?token=secret",
            "http://127.0.0.1:not-a-port",
            "http://[::1",
        ]
        for value in invalid_urls:
            with self.subTest(value=value):
                with self.assertRaises(relay.RelayTransportError):
                    relay.LocalRelayTransport(value, "token")
        with self.assertRaisesRegex(relay.RelayTransportError, "token"):
            relay.LocalRelayTransport("http://127.0.0.1:42117", "not a token")
        with self.assertRaisesRegex(relay.RelayTransportError, "poll limit"):
            self.transport(poll_limit=0)

    def test_invalid_delivery_is_atomic(self) -> None:
        transport = self.transport()
        authored = fixture_frame("authored")
        invalid = copy.deepcopy(fixture_frame("presence"))
        invalid["lane"] = "authored"
        response = batch(
            latest=2,
            deliveries=[(1, authored), (2, invalid)],
        )
        with self.assertRaises(protocol.HyperscapeProtocolError):
            transport._accept_batch(response, 0)
        self.assertEqual(transport.drain(), [])
        self.assertEqual(transport.status().cursor, 0)
        self.assertIsNone(transport.status().generation)

    def test_gap_delivers_only_retained_contiguous_suffix(self) -> None:
        transport = self.transport()
        response = batch(
            resume_after=1,
            latest=3,
            gap=True,
            deliveries=[
                (2, fixture_frame("authored")),
                (3, fixture_frame("presence")),
            ],
        )
        self.assertFalse(transport._accept_batch(response, 0))
        self.assertEqual(
            [delivery.cursor for delivery in transport.drain()],
            [2, 3],
        )
        status = transport.status()
        self.assertEqual(status.cursor, 3)
        self.assertEqual(status.gaps, 1)
        self.assertTrue(transport._degraded)

    def test_generation_change_discards_response_and_repolls_from_zero(self) -> None:
        transport = self.transport()
        self.assertFalse(transport._accept_batch(batch(), 0))
        changed = batch(
            generation="generation-b",
            latest=1,
            deliveries=[(1, fixture_frame("authored"))],
        )
        self.assertTrue(transport._accept_batch(changed, 0))
        self.assertEqual(transport.drain(), [])
        status = transport.status()
        self.assertEqual(status.generation, "generation-b")
        self.assertEqual(status.cursor, 0)
        self.assertEqual(status.restarts, 1)
        self.assertEqual(status.gaps, 1)

        self.assertFalse(transport._accept_batch(changed, 0))
        self.assertEqual([item.cursor for item in transport.drain()], [1])

    def test_cursor_and_pagination_inconsistencies_are_rejected(self) -> None:
        transport = self.transport()
        skipped = batch(
            latest=2,
            deliveries=[(2, fixture_frame("authored"))],
        )
        with self.assertRaisesRegex(relay.RelayTransportError, "contiguous"):
            transport._accept_batch(skipped, 0)

        stuck = batch(latest=1, has_more=True)
        with self.assertRaisesRegex(relay.RelayTransportError, "forward progress"):
            transport._accept_batch(stuck, 0)

    def test_failed_post_retries_before_later_authored_frame(self) -> None:
        calls: list[str] = []
        fail_first = True

        def request_json(method: str, path: str, body: str | None) -> dict:
            nonlocal fail_first
            self.assertEqual((method, path), ("POST", "/v1/frame"))
            assert body is not None
            calls.append(body)
            if fail_first:
                fail_first = False
                raise relay.RelayTransportError("deliberate failure")
            return {"generation": "generation-a", "cursor": str(len(calls))}

        transport = self.transport(request_json=request_json)
        first = fixture_frame("authored")
        second = copy.deepcopy(first)
        second["envelope"]["header"]["message_id"] = (
            "00000000-0000-0000-0000-000000000006"
        )
        second["envelope"]["header"]["sequence"] += 1
        transport.send(first)
        transport.send(second)

        with self.assertRaisesRegex(relay.RelayTransportError, "deliberate"):
            transport._flush_outbound()
        transport._flush_outbound()
        decoded_ids = [
            json.loads(payload)["envelope"]["header"]["message_id"]
            for payload in calls
        ]
        self.assertEqual(decoded_ids, [decoded_ids[0], decoded_ids[0], decoded_ids[2]])
        self.assertEqual(transport.status().sent_frames, 2)

    def test_worker_does_network_io_and_stops_without_touching_main_thread(self) -> None:
        worker_threads: set[int] = set()
        posted = threading.Event()
        main_thread = threading.get_ident()

        def request_json(method: str, path: str, body: str | None) -> dict:
            worker_threads.add(threading.get_ident())
            if method == "POST":
                posted.set()
                return {"generation": "generation-a", "cursor": "1"}
            query = parse_qs(urlparse(path).query)
            requested = int(query["after"][0])
            return batch(generation="generation-a", requested_after=requested)

        transport = self.transport(
            poll_interval_seconds=0.005,
            request_json=request_json,
        )
        transport.send(fixture_frame("presence"))
        transport.start()
        self.assertTrue(posted.wait(1.0))
        deadline = time.monotonic() + 1.0
        while transport.status().state == "connecting" and time.monotonic() < deadline:
            time.sleep(0.005)
        transport.stop()
        self.assertEqual(transport.status().state, "stopped")
        self.assertTrue(worker_threads)
        self.assertNotIn(main_thread, worker_threads)


if __name__ == "__main__":
    unittest.main()
