"""Versioned protocol primitives for vLLM V1 true KV migration.

This module is deliberately independent of vLLM's private classes.  It is the
Phase-1 contract that a maintained vLLM fork can implement inside EngineCore,
Scheduler and KVCacheManager.  It provides an executable state machine for
snapshot validation and two-phase block transfer; it does not move production
GPU memory by itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


PROTOCOL_VERSION = 1


class MigrationError(RuntimeError):
    """Base error for invalid migration state or payloads."""


class MigrationState(str, Enum):
    RUNNING = "running"
    MIGRATING_OUT = "migrating_out"
    OUT_READY = "out_ready"
    IMPORTING = "importing"
    IMPORT_READY = "import_ready"
    RELEASED = "released"


def _canonical_checksum(fields: Mapping[str, object]) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RequestMigrationSnapshot:
    """Serializable request state required to resume at the next token."""

    request_id: str
    migration_epoch: int
    prompt_token_ids: tuple[int, ...]
    all_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    num_computed_tokens: int
    max_tokens: int
    sampling_params: bytes = b""
    stop_reason: int | str | None = None
    kv_layout_version: str = "vllm-v1"
    kv_group_block_counts: tuple[int, ...] = ()
    feature_flags: frozenset[str] = frozenset()
    protocol_version: int = PROTOCOL_VERSION
    checksum: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise MigrationError("request_id must not be empty")
        if self.migration_epoch <= 0:
            raise MigrationError("migration_epoch must be positive")
        if self.num_computed_tokens < 0 or self.max_tokens < 0:
            raise MigrationError("token counts must be non-negative")
        if self.num_computed_tokens > len(self.all_token_ids):
            raise MigrationError("num_computed_tokens exceeds token state")
        if self.protocol_version != PROTOCOL_VERSION:
            raise MigrationError(
                f"unsupported migration protocol {self.protocol_version}")
        if any(count < 0 for count in self.kv_group_block_counts):
            raise MigrationError("KV block counts must be non-negative")
        if not self.checksum:
            object.__setattr__(self, "checksum", self.compute_checksum())

    def _checksum_fields(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "migration_epoch": self.migration_epoch,
            "prompt_token_ids": self.prompt_token_ids,
            "all_token_ids": self.all_token_ids,
            "output_token_ids": self.output_token_ids,
            "num_computed_tokens": self.num_computed_tokens,
            "max_tokens": self.max_tokens,
            "sampling_params": self.sampling_params.hex(),
            "stop_reason": self.stop_reason,
            "kv_layout_version": self.kv_layout_version,
            "kv_group_block_counts": self.kv_group_block_counts,
            "feature_flags": sorted(self.feature_flags),
        }

    def compute_checksum(self) -> str:
        return _canonical_checksum(self._checksum_fields())

    def validate_checksum(self) -> None:
        if self.checksum != self.compute_checksum():
            raise MigrationError("request snapshot checksum mismatch")


@dataclass(frozen=True)
class KVBlock:
    """Portable representation of one logical block for a fake/real backend."""

    group: int
    ordinal: int
    payload: bytes
    checksum: str = ""

    def __post_init__(self) -> None:
        if self.group < 0 or self.ordinal < 0:
            raise MigrationError("block coordinates must be non-negative")
        expected = hashlib.sha256(self.payload).hexdigest()
        if self.checksum and self.checksum != expected:
            raise MigrationError("KV block checksum mismatch")
        if not self.checksum:
            object.__setattr__(self, "checksum", expected)


@dataclass(frozen=True)
class ImportReservation:
    request_id: str
    migration_epoch: int
    block_ids: tuple[tuple[int, ...], ...]


class MigrationCoordinator:
    """Small reference state machine used by Phase-1 tests.

    A production adapter should call the same transitions from vLLM's
    EngineCore thread and replace the in-memory block dictionaries with
    KVCacheManager reservations and a connector data plane.
    """

    def __init__(self) -> None:
        self._states: dict[str, MigrationState] = {}
        self._snapshots: dict[str, RequestMigrationSnapshot] = {}
        self._source_blocks: dict[str, tuple[KVBlock, ...]] = {}
        self._reservations: dict[tuple[str, int], ImportReservation] = {}
        self._imported: dict[tuple[str, int], tuple[KVBlock, ...]] = {}
        self._next_block_id = 0

    def register_running(self, request_id: str, blocks: Sequence[KVBlock] = ()) -> None:
        if request_id in self._states:
            raise MigrationError(f"request already registered: {request_id}")
        self._states[request_id] = MigrationState.RUNNING
        self._source_blocks[request_id] = tuple(blocks)

    def prepare_migration_out(self, snapshot: RequestMigrationSnapshot,
                              blocks: Sequence[KVBlock]) -> None:
        snapshot.validate_checksum()
        request_id = snapshot.request_id
        if self._states.get(request_id) != MigrationState.RUNNING:
            raise MigrationError("request is not running")
        if len(blocks) != sum(snapshot.kv_group_block_counts):
            raise MigrationError("snapshot block count does not match payload")
        self._states[request_id] = MigrationState.MIGRATING_OUT
        self._snapshots[request_id] = snapshot
        self._source_blocks[request_id] = tuple(blocks)

    def export_migration(self, request_id: str) -> tuple[RequestMigrationSnapshot, tuple[KVBlock, ...]]:
        if self._states.get(request_id) != MigrationState.MIGRATING_OUT:
            raise MigrationError("request is not in MIGRATING_OUT")
        self._states[request_id] = MigrationState.OUT_READY
        return self._snapshots[request_id], self._source_blocks[request_id]

    def reserve_import(self, snapshot: RequestMigrationSnapshot) -> ImportReservation:
        snapshot.validate_checksum()
        key = (snapshot.request_id, snapshot.migration_epoch)
        if key in self._reservations:
            raise MigrationError("duplicate migration epoch")
        groups = tuple(tuple(self._allocate_id(count))
                      for count in snapshot.kv_group_block_counts)
        reservation = ImportReservation(snapshot.request_id,
                                        snapshot.migration_epoch, groups)
        self._reservations[key] = reservation
        self._states[snapshot.request_id] = MigrationState.IMPORTING
        return reservation

    def _allocate_id(self, count: int) -> list[int]:
        ids = list(range(self._next_block_id, self._next_block_id + count))
        self._next_block_id += count
        return ids

    def import_blocks(self, reservation: ImportReservation,
                      blocks: Sequence[KVBlock]) -> None:
        key = (reservation.request_id, reservation.migration_epoch)
        if self._reservations.get(key) != reservation:
            raise MigrationError("unknown import reservation")
        expected = sum(len(group) for group in reservation.block_ids)
        if len(blocks) != expected:
            raise MigrationError("import block count mismatch")
        for block in blocks:
            KVBlock(block.group, block.ordinal, block.payload, block.checksum)
        self._imported[key] = tuple(blocks)
        self._states[reservation.request_id] = MigrationState.IMPORT_READY

    def commit_import(self, reservation: ImportReservation) -> None:
        key = (reservation.request_id, reservation.migration_epoch)
        if self._states.get(reservation.request_id) != MigrationState.IMPORT_READY:
            raise MigrationError("import is not ready")
        if key not in self._imported:
            raise MigrationError("no imported KV payload")
        # The destination becomes schedulable only after the complete payload
        # has been validated.  Keeping this transition explicit mirrors the
        # EngineCore commit message in the vLLM fork.
        self._states[reservation.request_id] = MigrationState.RUNNING

    def commit_export(self, request_id: str, migration_epoch: int) -> None:
        if self._states.get(request_id) != MigrationState.OUT_READY:
            raise MigrationError("export is not ready")
        snapshot = self._snapshots[request_id]
        if snapshot.migration_epoch != migration_epoch:
            raise MigrationError("migration epoch mismatch")
        self._states[request_id] = MigrationState.RELEASED
        self._source_blocks.pop(request_id, None)

    def imported_blocks(self, request_id: str, migration_epoch: int) -> tuple[KVBlock, ...]:
        """Return committed destination payload for test/adaptor inspection."""
        return self._imported.get((request_id, migration_epoch), ())

    def abort(self, request_id: str, migration_epoch: int) -> None:
        state = self._states.get(request_id)
        snapshot = self._snapshots.get(request_id)
        if snapshot is not None and snapshot.migration_epoch != migration_epoch:
            raise MigrationError("migration epoch mismatch")
        if state in (MigrationState.MIGRATING_OUT, MigrationState.OUT_READY):
            self._states[request_id] = MigrationState.RUNNING
        elif state in (MigrationState.IMPORTING, MigrationState.IMPORT_READY):
            self._states.pop(request_id, None)
        else:
            raise MigrationError(f"cannot abort from state {state}")
        self._reservations.pop((request_id, migration_epoch), None)
        self._imported.pop((request_id, migration_epoch), None)

    def state(self, request_id: str) -> MigrationState:
        try:
            return self._states[request_id]
        except KeyError as exc:
            raise MigrationError(f"unknown request: {request_id}") from exc

    def source_blocks(self, request_id: str) -> tuple[KVBlock, ...]:
        return self._source_blocks.get(request_id, ())
