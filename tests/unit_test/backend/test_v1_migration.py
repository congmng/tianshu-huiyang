import pytest

from llumnix.backends.vllm.v1_migration import (
    KVBlock,
    MigrationCoordinator,
    MigrationError,
    MigrationState,
    RequestMigrationSnapshot,
)


def snapshot(epoch=1, blocks=2):
    return RequestMigrationSnapshot(
        request_id="req-1", migration_epoch=epoch,
        prompt_token_ids=(1, 2), all_token_ids=(1, 2, 3, 4),
        output_token_ids=(3, 4), num_computed_tokens=4, max_tokens=8,
        kv_group_block_counts=(blocks,),
    )


def blocks(count=2):
    return tuple(KVBlock(0, i, f"block-{i}".encode()) for i in range(count))


def test_snapshot_checksum_round_trip_and_tamper_detection():
    value = snapshot()
    value.validate_checksum()
    tampered = RequestMigrationSnapshot(
        **{**value.__dict__, "all_token_ids": (1, 2, 3, 99)})
    with pytest.raises(MigrationError, match="checksum"):
        tampered.validate_checksum()


def test_two_phase_transfer_releases_source_only_after_import_commit():
    source = MigrationCoordinator()
    target = MigrationCoordinator()
    source.register_running("req-1", blocks())
    snap = snapshot()
    source.prepare_migration_out(snap, source.source_blocks("req-1"))
    _, payload = source.export_migration("req-1")
    reservation = target.reserve_import(snap)
    target.import_blocks(reservation, payload)
    target.commit_import(reservation)
    assert target.state("req-1") == MigrationState.RUNNING
    assert target.imported_blocks("req-1", 1) == payload
    assert source.state("req-1") == MigrationState.OUT_READY
    assert source.source_blocks("req-1")
    source.commit_export("req-1", 1)
    assert source.state("req-1") == MigrationState.RELEASED
    assert source.source_blocks("req-1") == ()


def test_abort_restores_source_and_discards_target_reservation():
    source = MigrationCoordinator()
    target = MigrationCoordinator()
    source.register_running("req-1", blocks())
    snap = snapshot()
    source.prepare_migration_out(snap, source.source_blocks("req-1"))
    source.export_migration("req-1")
    reservation = target.reserve_import(snap)
    target.import_blocks(reservation, blocks())
    target.abort("req-1", 1)
    source.abort("req-1", 1)
    assert source.state("req-1") == MigrationState.RUNNING
    assert source.source_blocks("req-1")
    with pytest.raises(MigrationError):
        target.state("req-1")


def test_epoch_and_block_count_are_checked():
    coordinator = MigrationCoordinator()
    coordinator.register_running("req-1")
    with pytest.raises(MigrationError, match="block count"):
        coordinator.prepare_migration_out(snapshot(blocks=2), blocks(1))
    coordinator.prepare_migration_out(snapshot(blocks=0), ())
    coordinator.export_migration("req-1")
    with pytest.raises(MigrationError, match="epoch"):
        coordinator.commit_export("req-1", 7)
