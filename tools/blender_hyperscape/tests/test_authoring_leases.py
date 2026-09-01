from __future__ import annotations

from pathlib import Path
import sys
import unittest


ADDON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_DIR))

import authoring_leases  # noqa: E402
import protocol  # noqa: E402


ASSET_A = "10000000-0000-4000-8000-000000000001"
ASSET_B = "10000000-0000-4000-8000-000000000002"
ENTITY_A = "20000000-0000-4000-8000-000000000001"
ENTITY_B = "20000000-0000-4000-8000-000000000002"
PEER_A = "30000000-0000-4000-8000-000000000001"
PEER_B = "30000000-0000-4000-8000-000000000002"
def remote_presence(
    *, peer: str, lease: str, asset: str, entity: str, sequence: int = 1
) -> dict:
    return protocol.presence_envelope(
        message_id=f"50000000-0000-4000-8000-{sequence:012d}",
        sender=peer,
        sequence=sequence,
        ttl_millis=1500,
        authoring_leases=[
            {
                "lease_id": lease,
                "target": {"asset": asset, "entity": entity},
            }
        ],
    )


class AuthoringLeaseTests(unittest.TestCase):
    def test_default_uuid_factory_produces_a_valid_claim(self) -> None:
        controller = authoring_leases.AuthoringLeaseController()
        claims = controller.synchronize(
            "10000000-0000-4000-8000-000000000001",
            ["10000000-0000-4000-8000-000000000002"],
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            authoring_leases.normalize_stable_id(
                claims[0]["lease_id"], "lease ID"
            ),
            claims[0]["lease_id"],
        )

    def controller(self) -> authoring_leases.AuthoringLeaseController:
        lease_ids = iter(
            f"40000000-0000-4000-8000-{value:012d}" for value in range(1, 16)
        )
        return authoring_leases.AuthoringLeaseController(lambda: next(lease_ids))

    def test_claim_ids_refresh_stably_and_omission_releases(self) -> None:
        controller = self.controller()
        first = controller.synchronize(ASSET_A, [ENTITY_B, ENTITY_A])
        second = controller.synchronize(ASSET_A, [ENTITY_A, ENTITY_B])
        self.assertEqual(first, second)
        envelope = protocol.presence_envelope(
            message_id="50000000-0000-4000-8000-000000000100",
            sender=PEER_A,
            sequence=1,
            ttl_millis=1500,
            authoring_leases=first,
        )
        protocol.validate_presence_envelope(envelope)
        self.assertEqual(
            [claim["target"]["entity"] for claim in first],
            [ENTITY_A, ENTITY_B],
        )

        released = controller.synchronize(ASSET_A, [ENTITY_B])
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0], first[1])
        reacquired = controller.synchronize(ASSET_A, [ENTITY_A, ENTITY_B])
        self.assertNotEqual(reacquired[0]["lease_id"], first[0]["lease_id"])

        self.assertEqual(controller.synchronize(None, [ENTITY_A]), ())

    def test_asset_change_never_reuses_old_target_claim(self) -> None:
        controller = self.controller()
        first = controller.synchronize(ASSET_A, [ENTITY_A])[0]
        second = controller.synchronize(ASSET_B, [ENTITY_A])[0]
        self.assertNotEqual(first["lease_id"], second["lease_id"])
        self.assertEqual(second["target"]["asset"], ASSET_B)

    def test_remote_projection_is_scoped_deduplicated_and_sorted(self) -> None:
        lease_a = "40000000-0000-4000-8000-000000000010"
        lease_b = "40000000-0000-4000-8000-000000000011"
        peer_a = remote_presence(
            peer=PEER_A, lease=lease_a, asset=ASSET_A, entity=ENTITY_A
        )
        peer_b = remote_presence(
            peer=PEER_B, lease=lease_b, asset=ASSET_A, entity=ENTITY_A
        )
        other_asset = remote_presence(
            peer=PEER_B,
            lease="40000000-0000-4000-8000-000000000012",
            asset=ASSET_B,
            entity=ENTITY_B,
            sequence=2,
        )
        projected = authoring_leases.remote_holders(
            [peer_b, peer_a, peer_a, other_asset], ASSET_A
        )
        self.assertEqual(
            projected,
            {
                ENTITY_A: (
                    authoring_leases.LeaseHolder(PEER_A, lease_a),
                    authoring_leases.LeaseHolder(PEER_B, lease_b),
                )
            },
        )
        self.assertEqual(
            authoring_leases.remote_holders(
                [peer_a, peer_b], ASSET_A, exclude_peer=PEER_A
            ),
            {ENTITY_A: (authoring_leases.LeaseHolder(PEER_B, lease_b),)},
        )

    def test_nil_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(authoring_leases.AuthoringLeaseError, "nil"):
            self.controller().synchronize(
                "00000000-0000-0000-0000-000000000000", [ENTITY_A]
            )

    def test_claim_limit_fails_closed(self) -> None:
        self.assertEqual(
            authoring_leases.MAX_CLAIMS,
            protocol.MAX_AUTHORING_LEASES_PER_PRESENCE,
        )
        entities = [
            f"20000000-0000-4000-8000-{value:012d}"
            for value in range(1, authoring_leases.MAX_CLAIMS + 2)
        ]
        with self.assertRaisesRegex(authoring_leases.AuthoringLeaseError, "more than"):
            self.controller().synchronize(ASSET_A, entities)


if __name__ == "__main__":
    unittest.main()
