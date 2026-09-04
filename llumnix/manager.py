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

import asyncio
import random
import time
import csv
import os
from typing import Dict, List, Tuple, Union, Iterable
from collections import defaultdict
import traceback
from functools import partial

import ray
import ray.actor
from ray.util.state import list_placement_groups, list_actors
from ray.util.state.exception import ServerUnavailable
from ray.util.placement_group import PlacementGroup, placement_group_table

from llumnix.llumlet.llumlet import Llumlet
from llumnix.logging.logger import init_logger
from llumnix.global_scheduler.global_scheduler import GlobalScheduler
from llumnix.global_scheduler.migration_scheduler import PairMigrationConstraints
from llumnix.global_scheduler.migration_filter import CustomFilter
from llumnix.instance_info import InstanceInfo, InstanceType
from llumnix.arg_utils import ManagerArgs, EntrypointsArgs, InstanceArgs, LaunchArgs
from llumnix.server_info import ServerInfo
from llumnix.backends.backend_interface import BackendType
from llumnix.utils import (
    random_uuid,
    clear_gloo_backend_state,
    get_server_name,
    get_instance_name,
    get_manager_name,
    get_placement_group_name,
    INSTANCE_NAME_PREFIX,
    SERVER_NAME_PREFIX,
    run_async_func_sync,
)
from llumnix.entrypoints.utils import LaunchMode
from llumnix.queue.queue_type import QueueType
from llumnix.constants import (
    CLEAR_REQUEST_INSTANCE_INTERVAL,
    NO_INSTANCE_RETRY_GENERATE_INTERVAL,
    WAIT_ALL_MIGRATIONS_DONE_INTERVAL,
    AUTO_SCALE_UP_INTERVAL,
    WAIT_PLACEMENT_GROUP_TIMEOUT,
    CHECK_DEPLOYMENT_STATES_INTERVAL,
    WATCH_DEPLOYMENT_INTERVAL,
    WATCH_DEPLOYMENT_INTERVAL_PENDING_INSTANCE,
)
from llumnix.launcher import Launcher
from llumnix.metrics.timestamps import set_timestamp
from llumnix.entrypoints.vllm.api_server_actor import APIServerActor

logger = init_logger(__name__)

# TODO(s5u13b): Handle exception of ray operations.
# TODO(s5u13b): Refactor manager to divide functions into different classes.


class Manager:
    def __init__(
        self,
        entrypoints_args: EntrypointsArgs,
        manager_args: ManagerArgs,
        instance_args: InstanceArgs,
        engine_args,
        launch_args: LaunchArgs,
        work_dir: str,
    ) -> None:
        os.chdir(work_dir)
        self.job_id = ray.get_runtime_context().get_job_id()
        self.worker_id = ray.get_runtime_context().get_worker_id()
        self.actor_id = ray.get_runtime_context().get_actor_id()
        self.node_id = ray.get_runtime_context().get_node_id()
        logger.info(
            "Manager(job_id={}, worker_id={}, actor_id={}, node_id={})".format(
                self.job_id, self.worker_id, self.actor_id, self.node_id
            )
        )
        self.actor_name = get_manager_name()
        self.manager_args = manager_args

        # used in global deployment.
        self.entrypoints_args = entrypoints_args
        self.instance_args = instance_args
        self.engine_args = engine_args
        self.launch_args = launch_args

        # vLLM 0.11's V1 engine has no compatibility with Llumnix's legacy
        # block-manager migration coordinator.  Disable migration at the
        # manager boundary as well as CLI parsing, covering programmatic and
        # global deployments that construct Manager directly.
        self.is_vllm_v1 = False
        if launch_args is not None and launch_args.backend_type == BackendType.VLLM:
            try:
                import vllm
                self.is_vllm_v1 = getattr(vllm, "__version__", "").startswith("0.11")
            except ImportError:
                pass

        # launch args
        if launch_args is not None:
            self.launch_mode: LaunchMode = launch_args.launch_mode
            self.backend_type: BackendType = launch_args.backend_type

        # migration args
        self.enable_migration = manager_args.enable_migration and not self.is_vllm_v1
        if self.is_vllm_v1 and manager_args.enable_migration:
            logger.warning(
                "vLLM V1 detected in Manager; disabling legacy KV-cache migration"
            )
        self.pair_migration_frequency = manager_args.pair_migration_frequency
        self.enable_pd_disagg = manager_args.enable_pd_disagg

        # scaling args
        self.enable_scaling = manager_args.enable_scaling
        self.max_instances = manager_args.max_instances
        self.min_instances = manager_args.min_instances
        self.scaling_interval = manager_args.scaling_interval
        self.scaling_policy = manager_args.scaling_policy
        self.scale_up_threshold = manager_args.scale_up_threshold
        self.scale_down_threshold = manager_args.scale_down_threshold

        self.polling_interval = manager_args.polling_interval

        self.is_group_kind_migration_backend = (
            manager_args.is_group_kind_migration_backend
        )
        global_scheduler_config = manager_args.create_global_scheduler_config(
            self.is_group_kind_migration_backend
        )
        self.global_scheduler = GlobalScheduler(global_scheduler_config)

        self.launcher: Launcher = Launcher(
            self.global_scheduler,
            manager_args.enable_port_increment,
            manager_args.enable_port_offset_store,
            manager_args.enable_pd_disagg,
            manager_args.enable_engine_pd_disagg,
            manager_args.pd_ratio,
        )

        # log args
        self.log_requests = not manager_args.disable_log_requests_manager
        self.log_instance_info = manager_args.log_instance_info
        if self.log_instance_info:
            self._init_instance_info_csv(manager_args)
            self.instance_last_logged_empty = {}

        # instance states
        self.num_instances = 0
        self.instances: Dict[str, Llumlet] = {}
        # A placement group is already capacity reserved before its Llumlet
        # calls scale_up().  Track that gap locally because the CoreX Ray
        # wheel can lack the optional dashboard State API.
        self._pending_instance_ids = set()
        # Retain the handle for asynchronously-created PGs too.  Without the
        # State API there is no reliable way to look up a timed-out PG by
        # name on the next scaling interval.
        self._pending_placement_groups: Dict[str, PlacementGroup] = {}
        self.instance_migrating: Dict[str, bool] = {}
        self.pending_rebuild_migration_instances = 0
        # Global launches used to rely on the auto-scaling loop to create
        # their initial replicas. Keep fixed-size startup explicit and
        # idempotent instead: the loop is correctly absent when scaling is
        # disabled, but the configured initial instances must still exist.
        self._global_initial_instances_launched = False

        # request states
        self.request_instance: Dict[str, str] = {}
        # V1 P/D owns two backend requests for one public request.  Keep the
        # complete set so cancellation reaches both the producer and consumer.
        self.request_instances: Dict[str, set[str]] = {}

        # migration states
        self.num_instance_info_updates = 0
        self.migrating = False

        # auto-scaling states
        self.scale_up_time = -1
        self.scale_down_time = -1
        self.scaling_up = False
        self.scaling_down = False
        self.last_check_scale_time = time.time()
        # CoreX's production Ray wheel deliberately omits the optional
        # dashboard dependencies.  Global deployment can still create and
        # manage placement groups through the Ray control plane; only the
        # dashboard-backed State API reconciliation is unavailable.
        self._state_api_available = True

        # tasks
        # When manager starts, it automatically connects to all existing instances.
        run_async_func_sync(self._connect_to_instances())
        asyncio.create_task(self._poll_instance_info_loop(self.polling_interval))
        asyncio.create_task(
            self._clear_request_instance_loop(CLEAR_REQUEST_INSTANCE_INTERVAL)
        )

        if hasattr(self, "launch_mode") and self.launch_mode == LaunchMode.GLOBAL:
            assert self.entrypoints_args is not None and self.engine_args is not None
            self.last_timeout_instance_id = None
            # The auto-scale loop allocates a new placement group on every
            # cycle. It must not run for a fixed-size deployment
            # (``enable_scaling=False``), otherwise a two-instance P/D launch
            # accumulates pending GPU placement groups when no capacity is
            # available.
            if self.enable_scaling:
                asyncio.create_task(self._auto_scale_up_loop(AUTO_SCALE_UP_INTERVAL))
            asyncio.create_task(
                self._check_deployment_states_loop(CHECK_DEPLOYMENT_STATES_INTERVAL)
            )
            if self.manager_args.enable_pd_disagg:
                asyncio.create_task(
                    self._check_pd_deployment_states_loop(
                        CHECK_DEPLOYMENT_STATES_INTERVAL
                    )
                )

    def _disable_state_api_reconciliation(self) -> None:
        """Disable dashboard-only reconciliation after an unavailable State API.

        Placement groups and named actors remain usable through the Ray control
        plane in CoreX's minimal Ray build.  Retrying dashboard requests in
        each background loop only consumes a CPU core and hides real startup
        errors, so each loop must stop its optional reconciliation work.
        """
        if self._state_api_available:
            logger.warning(
                "Ray State API is unavailable; global deployment continues "
                "without dashboard placement-group reconciliation."
            )
        self._state_api_available = False

    async def generate(
        self,
        request_id: str,
        server_info: ServerInfo,
        *args,
        **kwargs,
    ) -> None:
        while self.num_instances == 0:
            logger.warning(
                "No instance available now, sleep {}s, "
                "and regenerate request {}.".format(
                    NO_INSTANCE_RETRY_GENERATE_INTERVAL, request_id
                )
            )
            await asyncio.sleep(NO_INSTANCE_RETRY_GENERATE_INTERVAL)

        # V1 prompt token hashes are intentionally optional: normal requests
        # retain the established load/queue policies, while callers that have
        # a prefix hash sequence can opt into cache-aware placement.
        block_hashes = kwargs.pop("llumnix_kv_block_hashes", None)
        # API clients may supply token IDs instead of precomputed hashes. The
        # V1 index computes hashes with the same algorithm as EngineCore.
        if block_hashes is None:
            token_ids = kwargs.pop("llumnix_kv_token_ids", None)
            block_size = kwargs.pop("llumnix_kv_block_size", None)
            if token_ids is not None and block_size:
                from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex
                block_hashes = KVCacheAffinityIndex().prefix_hashes(
                    token_ids,
                    int(block_size),
                    kwargs.pop("llumnix_kv_hash_algo", "sha256_cbor"),
                )
        if block_hashes is None and args and isinstance(args[0], str):
            # Query one live V1 engine for tokenizer/config compatible hashes.
            # This is best-effort and must not delay serving if an instance is
            # starting, remote, or running a legacy backend.
            for instance in self.instances.values():
                try:
                    block_hashes = await instance.get_prompt_block_hashes.remote(args[0])
                    if block_hashes:
                        break
                except (ray.exceptions.RayActorError, AttributeError):
                    continue
        pd_v1 = self.is_vllm_v1 and self.enable_pd_disagg
        prefill_id = decode_id = None
        if pd_v1:
            prefill_id, decode_id = self._select_v1_pd_instances(block_hashes)
            while prefill_id is None or decode_id is None:
                # A P/D deployment can briefly lose an entire role while Ray
                # replaces a failed actor.  Do not fall through to ordinary
                # dispatch (which could route a request to Decode-only or
                # start a single engine without KV handoff); retain the
                # request until both role pools are available again.
                logger.warning(
                    "V1 P/D role pool incomplete for request %s; retrying in %ss",
                    request_id, NO_INSTANCE_RETRY_GENERATE_INTERVAL,
                )
                await asyncio.sleep(NO_INSTANCE_RETRY_GENERATE_INTERVAL)
                prefill_id, decode_id = self._select_v1_pd_instances(block_hashes)
        if prefill_id is None:
            instance_id, request_expected_steps = self.global_scheduler.dispatch(block_hashes)
        else:
            instance_id, request_expected_steps = prefill_id, float("inf")
        try:
            set_timestamp(server_info, "manager_generate_timestamp", time.time())
            if prefill_id is not None:
                prefill_info = self.global_scheduler.instance_info[prefill_id]
                decode_info = self.global_scheduler.instance_info[decode_id]
                prefill_endpoint = getattr(prefill_info, "kv_endpoint", None)
                decode_endpoint = getattr(decode_info, "kv_endpoint", None)
                from llumnix.backends.vllm.v1_kv_transfer import valid_p2p_endpoint
                if not valid_p2p_endpoint(prefill_endpoint) or not valid_p2p_endpoint(decode_endpoint):
                    logger.warning(
                        "V1 P/D endpoints unavailable; falling back to a single request"
                    )
                    await self.instances[instance_id].generate.remote(
                        request_id, server_info, request_expected_steps, *args, **kwargs
                    )
                else:
                    producer_kwargs = dict(kwargs)
                    from llumnix.backends.vllm.v1_kv_transfer import decorate_p2p_pd_request_id
                    shared_p2p_id = decorate_p2p_pd_request_id(
                        request_id, decode_endpoint, prefill_endpoint
                    )
                    producer_kwargs.update(
                        llumnix_kv_decode_address=decode_endpoint,
                        llumnix_p2p_request_id=shared_p2p_id,
                        llumnix_suppress_output=True,
                    )
                    producer_args = list(args)
                    if len(producer_args) >= 2:
                        from llumnix.backends.vllm.v1_kv_transfer import producer_sampling_params
                        producer_args[1] = producer_sampling_params(producer_args[1])
                    try:
                        await self.instances[prefill_id].generate.remote(
                            request_id, server_info, request_expected_steps, *producer_args,
                            **producer_kwargs
                        )
                        consumer_kwargs = dict(kwargs)
                        consumer_kwargs.update(
                            llumnix_kv_prefill_address=prefill_endpoint,
                            llumnix_p2p_request_id=shared_p2p_id,
                            llumnix_public_request_id=request_id,
                        )
                        await self.instances[decode_id].generate.remote(
                            request_id, server_info, float("inf"), *args,
                            **consumer_kwargs
                        )
                    except Exception:
                        # A failure can happen after either stream was
                        # accepted. Cancel both sides so an already-started
                        # consumer cannot remain blocked waiting for KV and a
                        # producer cannot continue sending orphaned layers.
                        cleanup_ids = (prefill_id, decode_id)
                        cleanup_results = await asyncio.gather(
                            *(self.instances[iid].abort.remote(request_id)
                              for iid in cleanup_ids),
                            return_exceptions=True,
                        )
                        for iid, result in zip(cleanup_ids, cleanup_results):
                            if isinstance(result, Exception):
                                logger.warning(
                                    "failed to clean up V1 P/D request %s on %s",
                                    request_id,
                                    iid,
                                )
                        raise
                    instance_id = decode_id
                    self.request_instances[request_id] = {prefill_id, decode_id}
            else:
                await self.instances[instance_id].generate.remote(
                    request_id, server_info, request_expected_steps, *args, **kwargs
                )
            # Request bookkeeping is functional state, not merely logging.
            # In particular V1 P/D requests must remain addressable for abort
            # fan-out even when request logging is disabled.
            self.request_instance[request_id] = instance_id
            self.request_instances.setdefault(request_id, {instance_id})
            if self.log_requests:
                logger.info("manager receive request {}".format(request_id))
                logger.info(
                    "dispath request {} to instance {}".format(request_id, instance_id)
                )
        except (ray.exceptions.RayActorError, KeyError):
            logger.info(
                "Instance {} is dead, regenerate request {}.".format(
                    instance_id, request_id
                )
            )
            self.scale_down(instance_id)

    def _select_v1_pd_instances(self, block_hashes=None):
        """Select P/D roles through the production affinity-aware scheduler.

        Decode-only instances remain excluded from ordinary dispatch, but are
        deliberately visible to this constrained role selector.
        """
        prefill_ids = [
            iid for iid, info in self.global_scheduler.instance_info.items()
            if getattr(info, "instance_type", None) in (InstanceType.PREFILL, "prefill")
        ]
        decode_ids = [
            iid for iid, info in self.global_scheduler.instance_info.items()
            if getattr(info, "instance_type", None) in (InstanceType.DECODE, "decode")
        ]
        if not prefill_ids or not decode_ids:
            return None, None
        scheduler = self.global_scheduler.dispatch_scheduler
        scheduler.update_instance_infos(self.global_scheduler.instance_info)
        scheduler.instance_info.update(self.global_scheduler.instance_info)
        return (
            scheduler.dispatch_candidates(prefill_ids, block_hashes),
            scheduler.dispatch_candidates(decode_ids, block_hashes),
        )

    async def abort(self, request_id: Union[str, Iterable[str]]) -> None:
        def abort_done_callback(instance_id: str, request_ids: List[str], fut):
            ret = fut.result()[0]
            if not isinstance(ret, (ray.exceptions.RayActorError, KeyError)):
                if self.log_requests:
                    logger.info("Abort requests: {}.".format(request_ids))
                for req_id in request_ids:
                    if req_id in self.request_instance:
                        del self.request_instance[req_id]
                    else:
                        logger.warning(
                            "request {} is not in request_instance".format(req_id)
                        )
                    self.request_instances.pop(req_id, None)
            else:
                logger.info("Instance {} is dead.".format(instance_id))
                self.scale_down(instance_id)

        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        instance_requests = defaultdict(list)
        for req_id in request_ids:
            # Requests will be free by instance when finished, so it is acceptable to miss aborted requests.
            for instance_id in self.request_instances.get(
                req_id,
                {self.request_instance[req_id]} if req_id in self.request_instance else set(),
            ):
                instance_requests[instance_id].append(req_id)
        tasks = []
        for instance_id, request_ids in instance_requests.items():
            task = asyncio.gather(
                self.instances[instance_id].abort.remote(request_ids),
                return_exceptions=True,
            )
            task.add_done_callback(
                partial(abort_done_callback, instance_id, request_ids)
            )
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    @classmethod
    def from_args(
        cls,
        entrypoints_args: EntrypointsArgs,
        manager_args: ManagerArgs,
        instance_args: InstanceArgs,
        engine_args,
        launch_args: LaunchArgs,
    ) -> "Manager":
        manager_class = ray.remote(
            num_cpus=1,
            max_restarts=-1,
            name=get_manager_name(),
            namespace="llumnix",
            lifetime="detached",
        )(cls)
        manager = manager_class.remote(
            entrypoints_args,
            manager_args,
            instance_args,
            engine_args,
            launch_args,
            os.getcwd(),
        )
        return manager

    def init_instances(
        self,
        request_output_queue_type: QueueType,
        backend_type: BackendType,
        instance_args: InstanceArgs,
        engine_args,
    ) -> Tuple[List[str], List[Llumlet]]:
        async def instance_ready_scale_up(
            instance_id: str, instance: "ray.actor.ActorHandle"
        ):
            await instance.is_ready.remote()
            self.scale_up(instance_id, instance, instance_args)

        instance_ids: List[str] = []
        instances: List[Llumlet] = []
        for _ in range(self.manager_args.initial_instances):
            instance_id = random_uuid()
            placement_group = self.launcher.init_placement_group(
                get_placement_group_name(instance_id), engine_args, backend_type
            )
            self._pending_instance_ids.add(instance_id)
            instance = self.launcher.init_instance(
                instance_id,
                instance_args,
                placement_group,
                request_output_queue_type,
                backend_type,
                engine_args,
            )
            instance_ids.append(instance_id)
            instances.append(instance)
            asyncio.create_task(instance_ready_scale_up(instance_id, instance))

        return instance_ids, instances

    def init_global_instances(self) -> List[str]:
        """Create the configured global-launch replicas exactly once.

        Global serving owns one API actor per Llumlet, unlike local serving
        where the caller owns the API server.  This was historically hidden
        in ``_auto_scale_up_loop``; doing it explicitly prevents fixed-size
        deployments from leaking placement groups or starting with zero
        instances when autoscaling is disabled.
        """
        if self._global_initial_instances_launched:
            return list(self.instances)
        if not (hasattr(self, "launch_mode") and self.launch_mode == LaunchMode.GLOBAL):
            raise RuntimeError("init_global_instances is only valid for global launch")
        self._global_initial_instances_launched = True
        instance_ids: List[str] = []
        for _ in range(self.manager_args.initial_instances):
            instance_id = random_uuid()
            placement_group = self.launcher.init_placement_group(
                get_placement_group_name(instance_id),
                self.engine_args,
                self.backend_type,
                init_server=True,
                block=True,
            )
            self._pending_instance_ids.add(instance_id)
            self.launcher.init_server_and_instance(
                instance_id,
                self.entrypoints_args,
                self.instance_args,
                self.engine_args,
                self.backend_type,
                placement_group,
                instance_ready_cb=self.scale_up,
            )
            instance_ids.append(instance_id)
        return instance_ids

    async def is_ready(self) -> bool:
        """Called by api server, return true when all the instances have been successfully created."""
        tasks = [instance.is_ready.remote() for instance in self.instances.values()]
        is_ready_list = await asyncio.gather(*tasks, return_exceptions=True)
        return all(is_ready_list)

    async def _poll_instance_info_loop(self, interval: float) -> None:
        def get_instance_info_done_callback(instance_id: str, fut):
            try:
                ret = fut.result()[0]
                if isinstance(ret, Exception):
                    logger.error(f"Instance {instance_id} returned error: {ret}")
                    self.scale_down(instance_id)
                    return

                if ret is not None:
                    instance_infos.append(ret)
                    self.global_scheduler.update_instance_infos([ret])
            except Exception as e:
                logger.error(f"Error processing instance info for {instance_id}: {e}")
                self.scale_down(instance_id)

        while True:
            try:
                await asyncio.sleep(interval)
                tasks = []
                instance_infos = []
                for instance_id, instance in self.instances.items():
                    # Use asyncio.gather to wrap ray remote call to add done callback, asyncio.create_task will get error.
                    task = asyncio.gather(
                        instance.get_instance_info.remote(), return_exceptions=True
                    )
                    task.add_done_callback(
                        partial(get_instance_info_done_callback, instance_id)
                    )
                    tasks.append(task)
                if self.num_instance_info_updates % 100 == 0:
                    logger.debug(
                        "Polling instance infos of {} instances starts.".format(
                            self.num_instances
                        )
                    )
                await asyncio.gather(*tasks, return_exceptions=True)
                if self.num_instance_info_updates % 100 == 0:
                    logger.debug(
                        "Polling instance infos of {} instances ends.".format(
                            self.num_instances
                        )
                    )
                self.num_instance_info_updates += 1
                # Push migrate when the instance_info have updated a certain number of times.
                if (
                    self.enable_migration
                    and self.num_instance_info_updates != 0
                    and self.num_instance_info_updates % self.pair_migration_frequency
                    == 0
                ):
                    asyncio.create_task(self._push_migrations())
                if self.log_instance_info:
                    self._log_instance_infos_to_csv(instance_infos)
            # pylint: disable=W0703
            except Exception as e:
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    async def _push_migrations(self) -> None:
        if self.enable_pd_disagg:
            asyncio.create_task(
                self._migrate(PairMigrationConstraints.PREFILL_2_DECODING)
            )
            asyncio.create_task(
                self._migrate(PairMigrationConstraints.DECODING_2_DECODING)
            )
        else:
            asyncio.create_task(self._migrate(PairMigrationConstraints.NO_CONSTRAINTS))

    async def _migrate(self, pair_migration_type: PairMigrationConstraints) -> None:
        # TODO(s5u13b): Remove the migration done callback through decentralized migration refactoring.
        async def migrate_done_callback(
            ret, migrate_instance_pair: Tuple[str, str]
        ) -> None:
            if migrate_instance_pair[0] in self.instance_migrating:
                self.instance_migrating[migrate_instance_pair[0]] = False
            if migrate_instance_pair[1] in self.instance_migrating:
                self.instance_migrating[migrate_instance_pair[1]] = False
            if isinstance(
                ret,
                (ray.exceptions.RayActorError, ray.exceptions.RayTaskError, KeyError),
            ):
                has_error_pair = await self._check_instance_error(migrate_instance_pair)
                for i, has_error in enumerate(has_error_pair):
                    # Instance without error should clear migration states.
                    # TODO(s5u13b): Fix the clear_migration_states to adapt to the many-to-many migration.
                    if not has_error:
                        try:
                            await self.instances[
                                migrate_instance_pair[i]
                            ].clear_migration_states.remote(is_migrate_in=bool(i))
                        except (
                            ray.exceptions.RayActorError,
                            ray.exceptions.RayTaskError,
                            KeyError,
                        ):
                            has_error = True
                for i, has_error in enumerate(has_error_pair):
                    if has_error:
                        instance_id = migrate_instance_pair[i]
                        logger.info("Instance {} is dead.".format(instance_id))
                        self.scale_down(instance_id)
            else:
                migrate_out_request_ids = ret
                if migrate_out_request_ids:
                    migrate_out_request_id = migrate_out_request_ids[0]
                    self.request_instance[migrate_out_request_id] = (
                        migrate_instance_pair[1]
                    )
                logger.info(
                    "Instance {}->{} migrate done, migrate request {}".format(
                        migrate_instance_pair[0],
                        migrate_instance_pair[1],
                        migrate_out_request_ids,
                    )
                )

        def migrate_done_callback_wrapper(
            migrate_instance_pair: Tuple[str, str], fut
        ) -> None:
            ret = fut.result()[0]
            loop = asyncio.get_event_loop()
            loop.create_task(migrate_done_callback(ret, migrate_instance_pair))

        try:
            migrate_instance_pairs = self.global_scheduler.pair_migration(
                pair_migration_type
            )
            migration_tasks = []
            for _, migrate_instance_pair in enumerate(migrate_instance_pairs):
                migrate_out_instance_id, migrate_in_instance_id = migrate_instance_pair
                if (
                    self.instance_migrating[migrate_out_instance_id]
                    or self.instance_migrating[migrate_in_instance_id]
                ):
                    continue
                self.instance_migrating[migrate_out_instance_id] = True
                self.instance_migrating[migrate_in_instance_id] = True
                migrate_in_instance_name = get_instance_name(migrate_in_instance_id)
                task = asyncio.gather(
                    self.instances[migrate_out_instance_id].migrate_out.remote(
                        migrate_in_instance_name
                    ),
                    return_exceptions=True,
                )
                task.add_done_callback(
                    partial(migrate_done_callback_wrapper, migrate_instance_pair)
                )
                migration_tasks.append(task)
            if len(migration_tasks) > 0:
                logger.info("{} migration tasks starts.".format(len(migration_tasks)))
            await asyncio.gather(*migration_tasks, return_exceptions=True)
            if len(migration_tasks) > 0:
                logger.info("{} migration tasks ends.".format(len(migration_tasks)))
        # pylint: disable=W0703
        except Exception as e:
            logger.error("Unexpected exception: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))

    async def _auto_scale_up_loop(self, interval: float) -> None:
        while True:
            try:
                new_pg = None
                new_instance_id = None
                try:
                    if self._state_api_available and self.last_timeout_instance_id is not None:
                        last_timeout_pg_name = get_placement_group_name(
                            self.last_timeout_instance_id
                        )
                        last_timeout_pg_states = list_placement_groups(
                            filters=[("name", "=", last_timeout_pg_name)]
                        )
                        if len(last_timeout_pg_states) > 0:
                            new_instance_id = self.last_timeout_instance_id
                            # pending, created(without server and instance) or rescheduling
                            new_pg = ray.util.get_placement_group(last_timeout_pg_name)
                        # reset
                        self.last_timeout_instance_id = None
                except ServerUnavailable:
                    self._disable_state_api_reconciliation()
                    self.last_timeout_instance_id = None
                if not self._state_api_available and self._pending_placement_groups:
                    # Resume the locally retained pending PG rather than
                    # allocating another one every scaling interval.
                    new_instance_id, new_pg = next(
                        iter(self._pending_placement_groups.items())
                    )
                try:
                    if self._state_api_available:
                        pending_pg_states = list_placement_groups(
                            filters=[("state", "=", "PENDING")]
                        )
                        pending_pg_states.extend(
                            list_placement_groups(filters=[("state", "=", "RESCHEDULING")])
                        )
                        alive_pg_states = list_placement_groups(
                            filters=[("state", "!=", "REMOVED")]
                        )
                    else:
                        pending_pg_states = []
                        alive_pg_states = []
                except ServerUnavailable:
                    # Do not make dashboard availability a prerequisite for
                    # CoreX serving.  We retain scale-up and rely on actor
                    # health checks; state-based stale-PG reclamation resumes
                    # automatically on installations with ray[default].
                    self._disable_state_api_reconciliation()
                    pending_pg_states = []
                    alive_pg_states = []
                for pending_pg_state in pending_pg_states:
                    instance_id = pending_pg_state["name"].split("_")[-1]
                    if new_pg is not None and instance_id == new_instance_id:
                        continue
                    self.scale_down(instance_id)
                if (
                    self.max_instances != -1
                    # When the optional State API is unavailable, the state
                    # list is intentionally empty. Manager's registered
                    # instances are the authoritative lower-bound instead;
                    # otherwise every interval would request another PG.
                    # State API counts PGs cluster-wide, while the local
                    # count covers the registered instances plus PGs still
                    # awaiting Llumlet registration.  Use the larger value:
                    # adding these counts would double-count the same PG on
                    # full Ray installations.
                    and max(
                        len(alive_pg_states),
                        len(self.instances) + len(self._pending_instance_ids),
                    )
                    >= self.max_instances
                ):
                    logger.debug(
                        "The number of alive placement groups has reached the max_instances."
                    )
                    await asyncio.sleep(interval)
                    continue
                if new_pg is None:
                    new_instance_id = random_uuid()
                    new_pg = self.launcher.init_placement_group(
                        get_placement_group_name(new_instance_id),
                        self.engine_args,
                        self.backend_type,
                        init_server=True,
                        block=False,
                    )
                    self._pending_instance_ids.add(new_instance_id)
                    self._pending_placement_groups[new_instance_id] = new_pg
                try:
                    await asyncio.wait_for(new_pg.ready(), WAIT_PLACEMENT_GROUP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug(
                        "Waiting for new placement group {} ready timeout.".format(
                            new_instance_id
                        )
                    )
                    # After timeout, the new placement group might be pending,
                    # created(without server and instance), rescheduling.
                    self.last_timeout_instance_id = new_instance_id
                    await asyncio.sleep(interval)
                    continue
                self.launcher.init_server_and_instance(
                    new_instance_id,
                    self.entrypoints_args,
                    self.instance_args,
                    self.engine_args,
                    self.backend_type,
                    new_pg,
                    instance_ready_cb=self.scale_up,
                )
                logger.info(
                    "Deploy server and instance to new placement group done, instance_id: {}.".format(
                        new_instance_id
                    )
                )
            # pylint: disable=broad-except
            except Exception as e:
                if isinstance(e, ServerUnavailable):
                    self._disable_state_api_reconciliation()
                    await asyncio.sleep(interval)
                    continue
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    def scale_up(
        self,
        instance_id: Union[str, Iterable[str]],
        instance_actor_handle: Union[
            ray.actor.ActorHandle, Iterable[ray.actor.ActorHandle]
        ],
        instance_args: Union[InstanceArgs, Iterable[InstanceArgs]],
    ) -> None:
        if isinstance(instance_id, str):
            instance_id = [
                instance_id,
            ]
            instance_actor_handle = [
                instance_actor_handle,
            ]
            instance_args = [
                instance_args,
            ]
        instance_ids = list(instance_id)
        instance_actor_handles = list(instance_actor_handle)
        instance_args_list = list(instance_args)

        indeed_update = False
        no_pending_instance = self.pending_rebuild_migration_instances == 0

        for idx, ins_id in enumerate(instance_ids):
            self._pending_instance_ids.discard(ins_id)
            self._pending_placement_groups.pop(ins_id, None)
            if ins_id not in self.instances:
                indeed_update = True
                self.instances[ins_id] = instance_actor_handles[idx]
                self.instance_migrating[ins_id] = False
                if self.log_instance_info:
                    self.instance_last_logged_empty[ins_id] = False
                self.pending_rebuild_migration_instances += 1
        self.global_scheduler.scale_up(instance_ids, instance_args_list)
        self.num_instances = len(self.instances)

        # When scaling up, we need to rebuild the migration backend. But if initially self.pending_rebuild_migration_instances != 0,
        # a coroutine is already handling the changes in the number of instances in the cluster and it will account for the changes
        # caused by this scale-up (see rebuild_migration_backend for details). Therefore, we simply return in this case.
        # Specifically, for not group kind migration backend, there is no need to rebuild the group.
        if (
            self.enable_migration
            and self.is_group_kind_migration_backend
            and indeed_update
            and no_pending_instance
        ):
            asyncio.create_task(self._rebuild_migration_backend())

        return self.num_instances

    def scale_down(
        self,
        instance_id: Union[str, Iterable[str]],
        rebuild_migration_backend: bool = True,
    ) -> None:
        if isinstance(instance_id, str):
            instance_id = [
                instance_id,
            ]
        instance_ids = list(instance_id)

        indeed_update = False
        no_pending_instance = self.pending_rebuild_migration_instances == 0

        for ins_id in instance_ids:
            self._pending_instance_ids.discard(ins_id)
            self._pending_placement_groups.pop(ins_id, None)
            self.launcher.clear_instance_ray_resources(ins_id)
            if ins_id in self.instances:
                indeed_update = True
                if ins_id in self.instances:
                    del self.instances[ins_id]
                else:
                    logger.debug("instance {} is not in instances".format(ins_id))
                if ins_id in self.instance_migrating:
                    del self.instance_migrating[ins_id]
                else:
                    logger.debug(
                        "instance {} is not in instance_migrating".format(ins_id)
                    )
                if self.log_instance_info:
                    if ins_id in self.instance_last_logged_empty:
                        del self.instance_last_logged_empty[ins_id]
                    else:
                        logger.debug(
                            "instance {} is not in instance_last_logged_empty".format(
                                ins_id
                            )
                        )
                self.pending_rebuild_migration_instances += 1
        self.global_scheduler.scale_down(instance_ids)
        self.num_instances = len(self.instances)

        if self.enable_migration and self.is_group_kind_migration_backend:
            if len(self.instances) == 0:
                self.pending_rebuild_migration_instances = 0
                clear_gloo_backend_state()
            elif indeed_update and no_pending_instance and rebuild_migration_backend:
                asyncio.create_task(self._rebuild_migration_backend())

        return self.num_instances

    async def _check_deployment_states_loop(self, interval: float) -> None:
        async def watch_instance_deployment_states(instance_id: str):
            # There might be some delays of calling _init_server_and_instance, so sleep first.
            await asyncio.sleep(WATCH_DEPLOYMENT_INTERVAL)
            wait_pending_instance_time = 0.0
            while True:
                instance_state = list_actors(
                    filters=[("name", "=", get_instance_name(instance_id))]
                )
                instance_pending_creation = (
                    len(instance_state) == 1
                    and instance_state[0]["state"] == "PENDING_CREATION"
                )
                if not instance_pending_creation:
                    break
                await asyncio.sleep(WATCH_DEPLOYMENT_INTERVAL)
                wait_pending_instance_time += WATCH_DEPLOYMENT_INTERVAL
                if (
                    wait_pending_instance_time
                    >= WATCH_DEPLOYMENT_INTERVAL_PENDING_INSTANCE
                ):
                    break
            pg_created, server_alive, instance_alive = (
                self._get_instance_deployment_states(instance_id)
            )
            if pg_created and (not server_alive or not instance_alive):
                logger.warning(
                    "Instance {} deployment states incorrect, states: (pg {}, server {}, instance {})".format(
                        instance_id, pg_created, server_alive, instance_alive
                    )
                )
                self.scale_down(instance_id)

        while True:
            try:
                if not self._state_api_available:
                    return
                curr_pgs, curr_servers, curr_instances = (
                    self._get_cluster_deployment_states()
                )
                assert len(curr_pgs) >= max(len(curr_servers), len(curr_instances))
                tasks = []
                for instance_id in curr_pgs:
                    if (
                        instance_id not in curr_servers
                        or instance_id not in curr_instances
                    ):
                        tasks.append(
                            asyncio.create_task(
                                watch_instance_deployment_states(instance_id)
                            )
                        )
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(interval)
            # pylint: disable=broad-except
            except Exception as e:
                if isinstance(e, ServerUnavailable):
                    self._disable_state_api_reconciliation()
                    return
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    # TODO(KuilongCui): Currently, only one naive state check policy is implemented,
    # which prevents the cluster from consisting entirely of prefill or decode instances.
    async def _check_pd_deployment_states_loop(self, interval: float) -> None:
        previous_penging_pg_names = None
        state_api_kwargs = ({"address": os.environ["RAY_ADDRESS"]}
                            if os.environ.get("RAY_ADDRESS") else {})

        while True:
            try:
                if not self._state_api_available:
                    return
                pending_pg_states = list_placement_groups(
                    filters=[("state", "=", "PENDING")], **state_api_kwargs
                )
                rescheduling_pg_states = list_placement_groups(
                    filters=[("state", "=", "RESCHEDULING")], **state_api_kwargs
                )
                all_penging_pg_names = [pg.name for pg in pending_pg_states]

                if previous_penging_pg_names and len(rescheduling_pg_states) == 0:
                    new_pending_pg_states = list_placement_groups(
                        filters=[("state", "=", "PENDING")], **state_api_kwargs
                    )
                    all_new_penging_pg_names = [pg.name for pg in new_pending_pg_states]
                    if (
                        len(
                            set(previous_penging_pg_names).difference(
                                set(all_new_penging_pg_names)
                            )
                        )
                        == 0
                    ):
                        self._check_pd_deployment_states()
                    previous_penging_pg_names = all_new_penging_pg_names
                else:
                    previous_penging_pg_names = all_penging_pg_names

                await asyncio.sleep(interval)
            # pylint: disable=broad-except
            except Exception as e:
                # Dashboard State API is optional in the CoreX Ray wheel.
                # Any failure here must not be treated as a P/D instance
                # failure; disable this best-effort reconciliation loop.
                self._disable_state_api_reconciliation()
                if isinstance(e, ServerUnavailable):
                    return
                logger.warning("Disabling P/D placement-group reconciliation: %s", e)
                return
                logger.error("Unexpected exception: {}".format(e))
                logger.error("Exception traceback: {}".format(traceback.format_exc()))

    def _check_pd_deployment_states(self) -> str:
        prefill_instance_ids = (
            self.global_scheduler.dispatch_scheduler.available_dispatch_instance_set
        )
        cur_num_prefill = len(prefill_instance_ids)
        decode_instance_ids = (
            self.global_scheduler.instance_id_set - prefill_instance_ids
        )
        cur_num_decode = len(decode_instance_ids)

        scale_down_instance_id = ""
        if cur_num_prefill == 0 and cur_num_decode > 0:
            scale_down_instance_id = random.choice(list(decode_instance_ids))
            logger.info(
                "Check pd deployment, pd_ratio: {}, cur_num_prefill: {}, cur_num_decode: {}, "
                "all decode instances is decode instance, scale down decode instance {}".format(
                    self.manager_args.pd_ratio,
                    cur_num_prefill,
                    cur_num_decode,
                    scale_down_instance_id,
                )
            )

        if cur_num_decode == 0 and cur_num_prefill > 0:
            scale_down_instance_id = random.choice(list(prefill_instance_ids))
            logger.info(
                "Check pd deployment, pd_ratio: {}, cur_num_prefill: {}, cur_num_decode: {}, "
                "all instances is prefill instance, scale down prefill instance {}".format(
                    self.manager_args.pd_ratio,
                    cur_num_prefill,
                    cur_num_decode,
                    scale_down_instance_id,
                )
            )

        if scale_down_instance_id:
            self.scale_down(scale_down_instance_id)

        return scale_down_instance_id

    def _get_cluster_deployment_states(
        self,
    ) -> Tuple[
        Dict[str, PlacementGroup], Dict[str, APIServerActor], Dict[str, Llumlet]
    ]:
        curr_pgs: Dict[str, PlacementGroup] = {}
        curr_servers: Dict[str, PlacementGroup] = {}
        curr_instances: Dict[str, Llumlet] = {}

        # The State API otherwise autodetects a local Ray address.  That is
        # ambiguous when an isolated validation head and the production head
        # run on the same host.  Actors inherit RAY_ADDRESS from their
        # launcher, so use it explicitly when it is available.
        state_api_kwargs = {"address": os.environ["RAY_ADDRESS"]} \
            if os.environ.get("RAY_ADDRESS") else {}
        try:
            created_pg_states = list_placement_groups(
                filters=[("state", "=", "CREATED")], **state_api_kwargs)
            alive_actor_states = list_actors(
                filters=[("state", "=", "ALIVE")], **state_api_kwargs)
        except Exception as exc:  # Minimal CoreX Ray dashboards omit State API.
            logger.warning("Ray State API unavailable; using control-plane fallback: %s", exc)
            created_pg_states = [
                {"name": info["name"]}
                for info in placement_group_table().values()
                if info.get("state") == "CREATED"
            ]
            alive_actor_states = [
                {"name": actor["name"]}
                for actor in ray.util.list_named_actors(all_namespaces=True)
                if actor.get("namespace") == "llumnix"
            ]
        for created_pg_state in created_pg_states:
            instance_id = created_pg_state["name"].split("_")[-1]
            curr_pgs[instance_id] = ray.util.get_placement_group(
                created_pg_state["name"]
            )

        for alive_actor_state in alive_actor_states:
            if alive_actor_state["name"].startswith(SERVER_NAME_PREFIX):
                instance_id = alive_actor_state["name"].split("_")[-1]
                curr_servers[instance_id] = ray.get_actor(
                    alive_actor_state["name"], namespace="llumnix"
                )
            elif alive_actor_state["name"].startswith(INSTANCE_NAME_PREFIX):
                instance_id = alive_actor_state["name"].split("_")[-1]
                curr_instances[instance_id] = ray.get_actor(
                    alive_actor_state["name"], namespace="llumnix"
                )

        return curr_pgs, curr_servers, curr_instances

    def _get_instance_deployment_states(self, instance_id: str):
        state_api_kwargs = {"address": os.environ["RAY_ADDRESS"]} \
            if os.environ.get("RAY_ADDRESS") else {}
        try:
            pg_state = list_placement_groups(
                filters=[("name", "=", get_placement_group_name(instance_id))],
                **state_api_kwargs,
            )
            server_state = list_actors(
                filters=[("name", "=", get_server_name(instance_id))],
                **state_api_kwargs,
            )
            instance_state = list_actors(
                filters=[("name", "=", get_instance_name(instance_id))],
                **state_api_kwargs,
            )
        except Exception as exc:
            logger.warning("Ray State API unavailable; using control-plane fallback: %s", exc)
            pgs = placement_group_table()
            pg_state = ([{"state": info.get("state")} for info in pgs.values()
                         if info.get("name") == get_placement_group_name(instance_id)])
            names = {actor["name"] for actor in ray.util.list_named_actors(all_namespaces=True)
                     if actor.get("namespace") == "llumnix"}
            server_state = ([{"state": "ALIVE"}]
                            if get_server_name(instance_id) in names else [])
            instance_state = ([{"state": "ALIVE"}]
                              if get_instance_name(instance_id) in names else [])
        pg_created = len(pg_state) == 1 and pg_state[0]["state"] == "CREATED"
        server_alive = len(server_state) == 1 and server_state[0]["state"] == "ALIVE"
        instance_alive = (
            len(instance_state) == 1 and instance_state[0]["state"] == "ALIVE"
        )

        return pg_created, server_alive, instance_alive

    # TODO(KuilongCui): Add comments for this function.
    async def _rebuild_migration_backend(self) -> None:
        # Wait for all instances to finish migration
        while any(self.instance_migrating.values()):
            await asyncio.sleep(WAIT_ALL_MIGRATIONS_DONE_INTERVAL)

        # During rebuilding migration backend, disable migration.
        origin_config = self.enable_migration
        self.enable_migration = False

        async def run_task(alive_instances: List[str], task_name: str, *args, **kwargs):
            tasks = []
            for instance_name in alive_instances:
                llumlet_handle = self.instances[instance_name]
                tasks.append(
                    llumlet_handle.execute_engine_method.remote(
                        "_run_workers", task_name, *args, **kwargs
                    )
                )
            rets = await asyncio.gather(*tasks, return_exceptions=True)
            dead_instances = set()
            for instance_name, ret in zip(alive_instances, rets):
                if isinstance(ret, ray.exceptions.RayActorError):
                    dead_instances.add(instance_name)
            if len(dead_instances) > 0:
                self.scale_down(dead_instances, rebuild_migration_backend=False)
                clear_gloo_backend_state()
            return dead_instances

        alive_instances = sorted(self.instances.keys())
        pending_task = self.pending_rebuild_migration_instances
        group_name = None
        clear_gloo_backend_state()

        while len(alive_instances) > 0 and self.pending_rebuild_migration_instances > 0:
            dead_instances = set()
            group_name = random_uuid()
            instance_rank = {
                instance_id: index for index, instance_id in enumerate(alive_instances)
            }
            dead_instances.update(
                await run_task(
                    alive_instances,
                    "rebuild_migration_backend",
                    instance_rank,
                    group_name,
                )
            )
            if (
                len(dead_instances) == 0
                and self.pending_rebuild_migration_instances == pending_task
            ):
                dead_instances.update(await run_task(alive_instances, "warmup"))
            if len(dead_instances) == 0:
                self.pending_rebuild_migration_instances -= pending_task
            alive_instances = sorted(set(self.instances.keys()) - dead_instances)
            pending_task = self.pending_rebuild_migration_instances

        if len(alive_instances) == 0:
            self.pending_rebuild_migration_instances = 0
            group_name = None

        migration_filter: CustomFilter = (
            self.global_scheduler.migration_scheduler.migration_filter.get_filter(
                "migration_backend_init_filter"
            )
        )
        migration_filter.set_filter_condtition(
            src_filter=lambda instance_info: instance_info.instance_id
            in alive_instances,
            dst_filter=lambda instance_info: instance_info.instance_id
            in alive_instances,
        )

        logger.info(
            "Rebuild migration backend done, group_name: {}, alive instance ({}): {}.".format(
                group_name, len(alive_instances), alive_instances
            )
        )

        # Restore migrate config
        self.enable_migration = origin_config

    async def _connect_to_instances(self):
        def connect_to_instances_done_callback(
            instance_id: str, instance_actor_handle: "ray.actor.ActorHandle", fut
        ):
            ret = fut.result()[0]
            if not isinstance(ret, Exception):
                scale_up_instance_ids.append(instance_id)
                scale_up_instance_actor_handles.append(instance_actor_handle)
                scale_up_instance_args.append(ret)
                logger.info("Connect to instance {}".format(instance_id))
            else:
                logger.warning(
                    "Connect to instance {} failed, exception: {}".format(
                        instance_id, ret
                    )
                )

        # Must set True despite set namespance to llumnix.
        actor_names_dict = ray.util.list_named_actors(all_namespaces=True)
        instance_actor_names = [
            actor_name_dict["name"]
            for actor_name_dict in actor_names_dict
            if actor_name_dict["name"].startswith(INSTANCE_NAME_PREFIX)
        ]
        instance_actor_handles = [
            ray.get_actor(actor_name, namespace="llumnix")
            for actor_name in instance_actor_names
        ]
        scale_up_instance_ids = []
        scale_up_instance_actor_handles = []
        scale_up_instance_args = []
        tasks = []
        for instance_actor_name, instance_actor_handle in zip(
            instance_actor_names, instance_actor_handles
        ):
            instance_id = instance_actor_name[len("instance_") :]
            if instance_id not in self.instances:
                task = asyncio.gather(
                    instance_actor_handle.get_instance_args.remote(),
                    return_exceptions=True,
                )
                task.add_done_callback(
                    partial(
                        connect_to_instances_done_callback,
                        instance_id,
                        instance_actor_handle,
                    )
                )
                tasks.append(task)
        await asyncio.gather(*tasks)
        # The only function that can add instance actor handles to manager.
        self.scale_up(
            scale_up_instance_ids,
            scale_up_instance_actor_handles,
            scale_up_instance_args,
        )

    async def _check_instance_error(
        self, migrate_instance_pairs: Tuple[str, str]
    ) -> List[bool]:
        def check_instance_error_done_callback(idx: int, instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, (ray.exceptions.RayActorError, KeyError)):
                logger.info("Instance {} is alive.".format(instance_id))
                results[idx] = False
            else:
                logger.info("Instance {} is dead.".format(instance_id))
                results[idx] = True

        results = [None, None]
        tasks = []
        for idx, instance_id in enumerate(migrate_instance_pairs):
            task = asyncio.gather(
                self.instances[instance_id].is_ready.remote(), return_exceptions=True
            )
            task.add_done_callback(
                partial(check_instance_error_done_callback, idx, instance_id)
            )
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def _get_request_instance(self) -> None:
        def get_request_instance_done_callback(instance_id: str, fut):
            ret = fut.result()[0]
            if not isinstance(ret, ray.exceptions.RayActorError):
                instance_requests.append(ret)
                instance_ids.append(instance_id)
            else:
                logger.info("Instance {} is dead.".format(instance_id))
                self.scale_down(instance_id)

        instance_requests = []
        instance_ids = []
        tasks = []
        for instance_id, instance_actor_handle in self.instances.items():
            task = asyncio.gather(
                instance_actor_handle.get_all_request_ids.remote(),
                return_exceptions=True,
            )
            task.add_done_callback(
                partial(get_request_instance_done_callback, instance_id)
            )
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("instance_ids: {}".format(instance_ids))
        logger.debug("instance_requests: {}".format(instance_requests))

        self._reconcile_request_instances(instance_ids, instance_requests)

    def _reconcile_request_instances(
        self, instance_ids: Iterable[str], instance_requests: Iterable[Iterable[str]]
    ) -> None:
        """Replace request bookkeeping from Llumlets' active-request sets.

        A V1 P/D request is present on both producer and consumer actors.
        Rebuilding both maps atomically removes completed requests and keeps
        cancellation fan-out accurate while preserving legacy single-instance
        request routing.
        """
        active_by_request = defaultdict(set)
        for instance_id, requests in zip(instance_ids, instance_requests):
            for request_id in requests:
                active_by_request[request_id].add(instance_id)
        self.request_instance = {
            request_id: next(iter(active_instances))
            for request_id, active_instances in active_by_request.items()
        }
        self.request_instances = {
            request_id: set(active_instances)
            for request_id, active_instances in active_by_request.items()
        }

    async def _clear_request_instance_loop(self, interval: float):
        # Query actors each interval rather than clearing a local map.  In V1
        # P/D a request can belong to two actors, and clearing only one map
        # leaves stale cancellation targets after either stream finishes.
        while True:
            await self._get_request_instance()
            await asyncio.sleep(interval)

    def _init_instance_info_csv(self, manager_args: ManagerArgs) -> None:
        # pylint: disable=consider-using-with
        self.instance_info_file = open(
            manager_args.log_filename + "_instance.csv", "w", encoding="utf-8"
        )
        self.instance_info_csv = csv.writer(self.instance_info_file)
        self.instance_info_csv.writerow(
            [
                "timestamp",
                "instance_id",
                "step_id",
                "gpu_cache_usage",
                "num_available_gpu_blocks",
                "dispatch_load_metric",
                "migration_load_metric",
                "num_running_requests",
                "num_waiting_requests",
                "num_killed_requests",
                "inference_type",
                "bs",
                "profiling_data",
                "seq_lens",
                "num_instances",
                "num_seqs",
                "num_blocks_first_waiting_request",
                "num_blocks_all_waiting_requests",
                "waiting_time_first_waiting_request",
            ]
        )

    def _log_instance_infos_to_csv(self, instance_infos: List[InstanceInfo]) -> None:
        for instance_info in instance_infos:
            instance_id = instance_info.instance_id
            gpu_cache_usage = instance_info.gpu_cache_usage
            should_log = (gpu_cache_usage > 0) or (
                gpu_cache_usage == 0
                and not self.instance_last_logged_empty[instance_id]
            )
            if should_log:
                self.instance_last_logged_empty[instance_id] = gpu_cache_usage == 0
                self.instance_info_csv.writerow(
                    [
                        instance_info.timestamp,
                        instance_info.instance_id,
                        instance_info.step_id,
                        instance_info.gpu_cache_usage,
                        instance_info.num_available_gpu_blocks,
                        instance_info.dispatch_load_metric,
                        instance_info.migration_load_metric,
                        instance_info.num_running_requests,
                        instance_info.num_waiting_requests,
                        instance_info.num_killed_requests,
                        instance_info.inference_type,
                        instance_info.num_batched_tokens,
                        instance_info.profiling_data,
                        instance_info.running_seq_lens,
                        self.num_instances,
                        instance_info.num_seqs,
                        instance_info.num_blocks_first_waiting_request,
                        instance_info.num_blocks_all_waiting_requests,
                        instance_info.waiting_time_first_waiting_request,
                    ]
                )
        self.instance_info_file.flush()

    async def get_all_instances_info(self):
        """获取所有instance的详细信息"""
        instance_infos = []
        for instance_id, instance in self.instances.items():
            try:
                info = await instance.get_instance_info.remote()
                instance_infos.append(info)
            except (ray.exceptions.RayActorError, KeyError):
                logger.warning(f"Instance {instance_id} is not available")
                continue
        return instance_infos
