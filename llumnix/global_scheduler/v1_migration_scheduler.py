# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capability-aware instance pairing for vLLM V1 true KV migration."""

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.global_scheduler.migration_policy import PairMigrationConstraints
from llumnix.global_scheduler.migration_scheduler import MigrationScheduler

logger = init_logger(__name__)

# Minimum control/data-plane contract required for a production Decode-to-
# Decode migration.  ``incremental_precopy`` and ``seeded_rng`` are optional
# optimisations; they are selected separately by Manager when enabled.
V1_REQUIRED_MIGRATION_CAPABILITIES = frozenset({
    "token_boundary_freeze",
    "kv_snapshot",
    "native_nccl",
})


def v1_migration_capable(info: InstanceInfo,
                         required: Iterable[str] = ()) -> bool:
    """Return whether an instance can participate in V1 true KV migration."""
    capabilities = getattr(info, "migration_capabilities", frozenset()) or frozenset()
    required = set(required) if required else V1_REQUIRED_MIGRATION_CAPABILITIES
    return required.issubset(capabilities)


def v1_migration_compatibility_key(info: InstanceInfo) -> tuple:
    """Return the compatibility identity for safe V1 migration pairing."""
    return (
        int(getattr(info, "migration_protocol_version", 0) or 0),
        str(getattr(info, "migration_kv_layout_version", "") or ""),
        str(getattr(info, "device_class", "") or ""),
        str(getattr(info, "corex_stack", "") or ""),
    )


class V1MigrationScheduler:
    """Thin capability gate around the existing pair-migration scheduler.

    Manager must never pair a V1 instance with a legacy instance or with a V1
    instance that cannot safely freeze/resume requests.  This scheduler builds
    a capability-filtered view for each call and delegates the actual pairing
    policy to the already-tested ``MigrationScheduler``.
    """

    def __init__(self, pair_migration_policy: str,
                 migrate_out_load_threshold: float,
                 required_capabilities: Iterable[str] = ()) -> None:
        self.pair_migration_policy = pair_migration_policy
        self.migrate_out_load_threshold = migrate_out_load_threshold
        self.required_capabilities = frozenset(
            required_capabilities or V1_REQUIRED_MIGRATION_CAPABILITIES
        )
        self._group_schedulers: Dict[tuple, MigrationScheduler] = {}

    def update_instance_infos(self, instance_infos: Dict[str, InstanceInfo]) -> None:
        capable = {
            instance_id: info
            for instance_id, info in instance_infos.items()
            if v1_migration_capable(info, self.required_capabilities)
        }
        grouped: Dict[tuple, Dict[str, InstanceInfo]] = defaultdict(dict)
        for instance_id, info in capable.items():
            grouped[v1_migration_compatibility_key(info)][instance_id] = info
        self._group_schedulers = {}
        for key, group_infos in grouped.items():
            scheduler = MigrationScheduler(
                self.pair_migration_policy, self.migrate_out_load_threshold, False
            )
            scheduler.update_instance_infos(group_infos)
            self._group_schedulers[key] = scheduler

    def pair_migration(
        self, pair_migration_type: PairMigrationConstraints
    ) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for scheduler in self._group_schedulers.values():
            pairs.extend(scheduler.pair_migration(pair_migration_type))
        return pairs
