"""Dependency-free advisory authoring-lease state for Blender adapters.

Leases coordinate concurrent editors; they never authorize a durable command.
The controller deliberately derives desire from its caller and emits complete
presence snapshots, so omission is an immediate release and TTL is the crash
fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid


MAX_CLAIMS = 256


class AuthoringLeaseError(ValueError):
    """A lease target or holder has invalid stable identity."""


def normalize_stable_id(value: Any, context: str) -> str:
    try:
        parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise AuthoringLeaseError(f"{context} must be a UUID") from error
    if parsed.int == 0:
        raise AuthoringLeaseError(f"{context} must not be nil")
    return str(parsed)


@dataclass(frozen=True, order=True)
class LeaseHolder:
    peer_id: str
    lease_id: str


class AuthoringLeaseController:
    """Keep stable claim IDs while a Blender selection desires its targets."""

    def __init__(self, id_factory: Callable[[], Any] = uuid.uuid4) -> None:
        self._id_factory = id_factory
        self._asset_id: str | None = None
        self._lease_ids: dict[str, str] = {}

    def synchronize(
        self,
        asset_id: str | None,
        desired_entities: Iterable[str],
    ) -> tuple[dict[str, Any], ...]:
        if asset_id is None or not asset_id.strip():
            self.clear()
            return ()
        asset_id = normalize_stable_id(asset_id.strip(), "authoring asset ID")
        desired = {
            normalize_stable_id(entity, "authoring entity ID")
            for entity in desired_entities
        }
        if len(desired) > MAX_CLAIMS:
            raise AuthoringLeaseError(
                f"authoring presence cannot claim more than {MAX_CLAIMS} entities"
            )
        if asset_id != self._asset_id:
            self._lease_ids.clear()
            self._asset_id = asset_id
        self._lease_ids = {
            entity: lease_id
            for entity, lease_id in self._lease_ids.items()
            if entity in desired
        }
        for entity in sorted(desired):
            if entity not in self._lease_ids:
                self._lease_ids[entity] = normalize_stable_id(
                    self._id_factory(), "authoring lease ID"
                )
        return tuple(
            {
                "lease_id": self._lease_ids[entity],
                "target": {"asset": asset_id, "entity": entity},
            }
            for entity in sorted(self._lease_ids)
        )

    def clear(self) -> None:
        self._asset_id = None
        self._lease_ids.clear()


def remote_holders(
    envelopes: Sequence[Mapping[str, Any]],
    asset_id: str,
    *,
    exclude_peer: str | None = None,
) -> dict[str, tuple[LeaseHolder, ...]]:
    """Project admitted live presence into stable, asset-scoped holders."""

    asset_id = normalize_stable_id(asset_id, "authoring asset ID")
    excluded = (
        normalize_stable_id(exclude_peer, "excluded peer ID")
        if exclude_peer is not None
        else None
    )
    holders: dict[str, set[LeaseHolder]] = {}
    for envelope in envelopes:
        header = envelope.get("header")
        presence = envelope.get("presence")
        if not isinstance(header, Mapping) or not isinstance(presence, Mapping):
            raise AuthoringLeaseError("admitted presence envelope is malformed")
        peer_id = normalize_stable_id(header.get("sender"), "lease holder peer ID")
        if peer_id == excluded:
            continue
        claims = presence.get("authoring_leases", [])
        if not isinstance(claims, list):
            raise AuthoringLeaseError("presence authoring leases must be an array")
        for claim in claims:
            if not isinstance(claim, Mapping) or not isinstance(
                claim.get("target"), Mapping
            ):
                raise AuthoringLeaseError("authoring lease claim is malformed")
            target = claim["target"]
            if normalize_stable_id(
                target.get("asset"), "authoring lease asset ID"
            ) != asset_id:
                continue
            entity = normalize_stable_id(
                target.get("entity"), "authoring lease entity ID"
            )
            lease_id = normalize_stable_id(
                claim.get("lease_id"), "authoring lease ID"
            )
            holders.setdefault(entity, set()).add(LeaseHolder(peer_id, lease_id))
    return {
        entity: tuple(sorted(entity_holders))
        for entity, entity_holders in sorted(holders.items())
    }
