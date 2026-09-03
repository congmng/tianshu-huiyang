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
import traceback
from typing import List, Union, Iterable
import time

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator
from llumnix.backends.backend_interface import (
    BackendInterface,
    BackendType,
    EngineState,
)
from llumnix.backends.utils import init_backend_engine, get_engine_world_size
from llumnix.llumlet.migration_coordinator import MigrationCoordinator, MigrationStatus
from llumnix.llumlet.local_migration_scheduler import LocalMigrationScheduler
from llumnix.server_info import ServerInfo
from llumnix.internal_config import MigrationConfig
from llumnix.queue.queue_type import QueueType
from llumnix.llumlet.request import LlumnixRequest, RequestStatus
from llumnix.arg_utils import InstanceArgs
from llumnix.utils import get_instance_name
from llumnix.constants import CHECK_ENGINE_STATE_INTERVAL
from llumnix.metrics.timestamps import set_timestamp

logger = init_logger(__name__)


class Llumlet:
    def __init__(
        self,
        instance_id: str,
        instance_args: InstanceArgs,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        backend_type: BackendType,
        engine_args,
    ) -> None:
        try:
            self.job_id = ray.get_runtime_context().get_job_id()
            self.worker_id = ray.get_runtime_context().get_worker_id()
            self.actor_id = ray.get_runtime_context().get_actor_id()
            self.node_id = ray.get_runtime_context().get_node_id()
            self.instance_id = instance_id
            logger.info(
                "Llumlet(job_id={}, worker_id={}, actor_id={}, node_id={}, instance_id={})".format(
                    self.job_id,
                    self.worker_id,
                    self.actor_id,
                    self.node_id,
                    self.instance_id,
                )
            )
            logger.info("Llumlet backend type: {}".format(backend_type))
            self.instance_args: InstanceArgs = instance_args
            self.actor_name = get_instance_name(instance_id)
            logger.info(f"instance_args: {instance_args}")
            self.instance_load_calculator = InstanceLoadCalculator(
                dispatch_load_metric=instance_args.dispatch_load_metric,
                migration_load_metric=instance_args.migration_load_metric,
                enable_defrag=instance_args.enable_defrag,
            )
            migration_config: MigrationConfig = instance_args.create_migration_config()
            # Used only by the V1 connector adapter to derive producer/
            # consumer defaults for P/D deployments.
            migration_config.instance_type = instance_args.instance_type
            self.backend_engine: BackendInterface = init_backend_engine(
                instance_id,
                placement_group,
                request_output_queue_type,
                migration_config,
                backend_type,
                engine_args,
                instance_args.profiling_result_file_path,
            )
            self.is_vllm_v1 = (
                backend_type == BackendType.VLLM
                and self.backend_engine.__class__.__name__ == "V1EngineAdapter"
            )
            if self.is_vllm_v1:
                # V1 owns its scheduler/engine loop.  The legacy Llumnix
                # migration coordinator depends on the removed vLLM 0.6 block
                # manager and therefore must not be started accidentally.
                logger.warning(
                    "Llumnix is using the vLLM V1 serving adapter; KV-cache "
                    "block-manager migration is replaced by connector-driven "
                    "P/D KV handoff."
                )
                self.migration_coordinator = None
                self.migration_scheduler = None
            else:
                self.migration_coordinator = MigrationCoordinator(
                    self.backend_engine,
                    migration_config.migration_last_stage_max_blocks,
                    migration_config.migration_max_stages,
                )
                self.migration_scheduler = LocalMigrationScheduler(
                    migration_config.request_migration_policy, self.backend_engine
                )
            self.log_requests = True

            asyncio.create_task(self._check_engine_state_loop())
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Failed to initialize Llumlet: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))
            raise

    def __repr__(self):
        # Construction can fail before ``instance_id`` is assigned (for
        # example when a placement group cannot reserve a GPU).  Keep Ray's
        # error serialization safe in that partial-initialization path.
        instance_id = getattr(self, "instance_id", "unknown")
        return f"{self.__class__.__name__}(iid={str(instance_id)[:5]})"

    @classmethod
    def from_args(
        cls,
        instance_id: str,
        instance_args: InstanceArgs,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        backend_type: BackendType,
        engine_args,
    ):
        try:
            assert backend_type in [
                backend_type.VLLM,
                backend_type.BLADELLM,
                backend_type.SIM_VLLM,
            ], f"unimplemented backend {backend_type}"
            if backend_type == BackendType.VLLM:
                # V1 AsyncLLM owns the worker process and consumes the full
                # placement-group GPU allocation. Legacy 0.6 used a 0.5 GPU
                # Llumlet plus a separate Ray executor.
                import vllm
                is_v1 = getattr(vllm, "__version__", "").startswith("0.11")
                num_gpus = (get_engine_world_size(engine_args, backend_type)
                            if is_v1 else 0.5)
            elif backend_type == backend_type.BLADELLM:
                world_size = get_engine_world_size(engine_args, backend_type)
                num_gpus = world_size
            else:  # backend_type == BackendType.SIM_VLLM
                num_gpus = 0
            actor_options = dict(
                num_cpus=1,
                num_gpus=num_gpus,
                name=get_instance_name(instance_id),
                namespace="llumnix",
                lifetime="detached",
            )
            actor_runtime_options = dict(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                )
            )
            if backend_type == BackendType.VLLM and is_v1:
                # CoreX Ray accepts runtime environments at ``.options``
                # submission time. Passing this dict to ``ray.remote``
                # itself can be mis-decoded by its actor task handler.
                from llumnix.backends.vllm.v1_kv_transfer import v1_kv_runtime_env
                actor_runtime_options["runtime_env"] = {
                    "env_vars": v1_kv_runtime_env()
                }
            # Some CoreX Ray builds return an ``ActorOptionWrapper`` from
            # ``ray.remote(**opts)(cls)`` which does not expose the usual
            # ``.options`` method.  Keep the normal path for upstream Ray,
            # but fall back to supplying scheduling/runtime options directly
            # to ``ray.remote`` on those builds.
            try:
                llumlet_class = ray.remote(**actor_options)(cls).options(
                    **actor_runtime_options
                )
            except AttributeError:
                llumlet_class = ray.remote(
                    **actor_options, **actor_runtime_options
                )(cls)
            llumlet = llumlet_class.remote(
                instance_id,
                instance_args,
                placement_group,
                request_output_queue_type,
                backend_type,
                engine_args,
            )
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("Failed to initialize Llumlet: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))
            raise

        return llumlet

    async def _check_engine_state_loop(self):
        while True:
            await asyncio.sleep(CHECK_ENGINE_STATE_INTERVAL)
            if self.backend_engine.state == EngineState.CRASHED:
                logger.error(
                    "Llumlet ({}) detected backend engine crashed. Stopping...".format(
                        self.instance_id
                    )
                )
                # pylint: disable=protected-access
                self.backend_engine._stop_event.set()
                await asyncio.sleep(0)
                self_actor = ray.get_actor(name=self.actor_name, namespace="llumnix")
                ray.kill(self_actor)

    async def migrate_out(self, dst_instance_name: str) -> List[str]:
        if self.is_vllm_v1:
            # Never let an externally triggered migration call reach the
            # legacy coordinator: it mutates vLLM 0.6 block-manager state
            # that does not exist in V1.
            logger.warning(
                "Ignoring legacy migration request for V1 Llumlet %s; use "
                "connector-driven P/D KV handoff instead.", self.instance_id
            )
            return []
        migrate_out_requests = self.migration_scheduler.get_migrate_out_requests()

        if len(migrate_out_requests) == 0:
            return []

        for migrate_out_request in migrate_out_requests:
            migrate_out_request.is_migrating = True

        migrated_request_list = []
        for migrate_out_request in migrate_out_requests:
            migrated_request = await self._migrate_out_one_request(
                migrate_out_request, dst_instance_name
            )
            migrated_request_list.extend(migrated_request)
            if len(migrated_request) == 0 and migrate_out_request.eom:
                break
        return migrated_request_list

    async def _migrate_out_one_request(
        self, migrate_out_request: LlumnixRequest, dst_instance_name: str
    ) -> List[LlumnixRequest]:
        try:
            t0 = time.time()
            migrate_in_ray_actor = ray.get_actor(dst_instance_name, namespace="llumnix")
            dst_instance_id = dst_instance_name[len("instance_") :]
            logger.info(
                "{}->{} begin migrate out".format(self.instance_id, dst_instance_id)
            )
            migrated_request = []

            if migrate_out_request.status == RequestStatus.RUNNING:
                migrate_out_request.migration_start_time = time.time()
                status = await self.migration_coordinator.migrate_out_running_request(
                    migrate_in_ray_actor, migrate_out_request
                )
            elif migrate_out_request.status == RequestStatus.WAITING:
                migrate_out_request.migration_start_time = time.time()
                status = await self.migration_coordinator.migrate_out_waiting_request(
                    migrate_in_ray_actor, migrate_out_request
                )
            else:
                return migrated_request

            if status == MigrationStatus.FINISHED:
                await migrate_in_ray_actor.execute_engine_method.remote(
                    "commit_dst_request", migrate_out_request
                )
                self.backend_engine.free_src_request(migrate_out_request)
                self.backend_engine.remove_migrating_out_request_last_stage(
                    migrate_out_request
                )
                migrated_request.append(migrate_out_request.request_id)
            else:  # ABORTED_SRC or ABORTED_DST
                migrate_out_request.reset_migration_args_src()
                migrate_out_request.reset_status()
                # If dst aborts itself, dst proactively frees the pre allocated cache in migrate_in_pre_alloc.
                if status == MigrationStatus.ABORTED_SRC:
                    await migrate_in_ray_actor.execute_migration_method.remote(
                        "free_dst_pre_alloc_cache", migrate_out_request.request_id
                    )
            t1 = time.time()
            logger.info(
                "Instance {}->{} migrate done, migrate request {}, migration status: {}, len: {} blocks, cost: {} ms".format(
                    self.instance_id,
                    dst_instance_id,
                    migrated_request,
                    status,
                    sum(migrate_out_request.stage_num_blocks_list),
                    (t1 - t0) * 1000,
                )
            )
        except ray.exceptions.RayActorError:
            logger.info(
                "Instance {} is dead.".format(dst_instance_name[len("instance_") :])
            )
            raise
        # pylint: disable=W0703
        except Exception as e:
            logger.error("Unexpected exception: {}".format(e))
            logger.error("Exception traceback: {}".format(traceback.format_exc()))
            raise
        return migrated_request

    # TODO(KuilongCui): only the metrics-related information needs to be synchronously loaded for the manager
    def get_instance_info(self) -> InstanceInfo:
        if self.is_vllm_v1:
            instance_info = InstanceInfo(instance_id=self.instance_id)
            instance_info.node_id = self.node_id
            try:
                instance_info.node_ip = ray.get_runtime_context().get_node_ip_address()
            except (AttributeError, RuntimeError):
                instance_info.node_ip = ""
            # CoreX's Ray worker context can expose an empty address even
            # though the control plane has the node manager address.  Keep
            # topology observability usable for cross-host KV affinity/P-D
            # deployments by falling back to the authoritative node table.
            if not instance_info.node_ip:
                for node in ray.nodes():
                    if node.get("NodeID") == self.node_id:
                        instance_info.node_ip = node.get(
                            "NodeManagerAddress", ""
                        )
                        break
            self.backend_engine.update_instance_info(instance_info)
            instance_info.kv_endpoint = self.backend_engine.get_kv_endpoint(
                instance_info.node_ip or None
            )
        else:
            instance_info: InstanceInfo = self.backend_engine.engine.instance_info
        instance_info.instance_type = self.instance_args.instance_type
        self.instance_load_calculator.compute_instance_load(instance_info)
        return instance_info

    def is_v1_adapter(self) -> bool:
        return self.is_vllm_v1

    def is_ready(self) -> bool:
        return True

    def get_instance_args(self) -> InstanceArgs:
        return self.instance_args

    def get_all_request_ids(self) -> List[str]:
        return self.backend_engine.get_all_request_ids()

    def get_prompt_block_hashes(self, prompt: str):
        """Best-effort V1 prefix hashes used by Manager before dispatch."""
        if not self.is_vllm_v1:
            return ()
        return self.backend_engine.get_prompt_block_hashes(prompt)

    def generate(
        self,
        request_id: str,
        server_info: ServerInfo,
        expected_steps: int,
        *args,
        **kwargs,
    ) -> None:
        set_timestamp(server_info, "llumlet_generate_timestamp", time.time())
        suppress_output = bool(kwargs.pop("llumnix_suppress_output", False))
        public_request_id = kwargs.pop("llumnix_public_request_id", request_id)
        request = self.backend_engine.add_request(
            request_id, server_info, expected_steps, *args, **kwargs
        )
        if self.is_vllm_v1:
            asyncio.create_task(
                self._forward_v1_outputs(
                    request_id, server_info, request,
                    suppress_output=suppress_output,
                    public_request_id=public_request_id,
                )
            )

    async def _forward_v1_outputs(
        self, request_id, server_info, request, *, suppress_output=False,
        public_request_id=None,
    ) -> None:
        """Bridge V1 AsyncLLM output into the existing Llumnix queue."""
        try:
            async for output in request:
                # P2pNcclConnector appends routing metadata to producer
                # request IDs. Restore the public ID before sending output to
                # the API queue so clients and Manager bookkeeping remain
                # stable.
                if suppress_output:
                    continue
                if self.is_vllm_v1:
                    output.request_id = public_request_id or self.backend_engine.public_request_id(
                        output.request_id
                    )
                await self._put_v1_outputs([output], server_info)
        except Exception:
            logger.error("V1 request %s failed: %s", request_id, traceback.format_exc())
            # EngineCore may terminate a stream without emitting a final
            # output (worker failure, connector timeout, or malformed peer
            # payload). Explicitly abort the internal alias so a P/D peer is
            # not left waiting and its KV buffers are released.
            try:
                await self.backend_engine.abort(request_id)
            except Exception:
                logger.warning(
                    "Failed to abort failed V1 request %s", request_id,
                    exc_info=True,
                )
        finally:
            self.backend_engine.requests.pop(request_id, None)
            self.backend_engine._request_id_aliases.pop(request_id, None)
            try:
                self.backend_engine.running.remove(request_id)
            except ValueError:
                pass

    async def _put_v1_outputs(self, outputs, server_info) -> None:
        from llumnix.queue.utils import init_request_output_queue_client
        client = init_request_output_queue_client(server_info.request_output_queue_type)
        await client.put_nowait(outputs, server_info)

    def abort(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        return self.backend_engine.abort_request(request_ids)

    def clear_migration_states(self, is_migrate_in: bool) -> None:
        if self.is_vllm_v1:
            logger.warning(
                "Ignoring legacy migration-state cleanup for V1 Llumlet %s.",
                self.instance_id,
            )
            return
        logger.info(
            "Instance {} clear_migration_states, is_migrate_in: {}".format(
                self.instance_id, is_migrate_in
            )
        )
        if is_migrate_in:
            # If migrate out instance dies during migration, migrate in instance directly free the pre-allocated cache of the migrating in request.
            logger.info("clear_migration_states: free_dst_pre_alloc_cache")
            self.backend_engine.free_dst_pre_alloc_cache()
        else:
            # If migrate in instance dies during migration, migrate out instance should add the migrating out request in last stage.
            # back to the running request queue.
            migrating_out_requests_last_stage = (
                self.backend_engine.pop_migrating_out_requests_last_stage()
            )
            for backend_request in migrating_out_requests_last_stage:
                logger.info(
                    "clear_migration_states: add request {} back to engine".format(
                        backend_request.request_id
                    )
                )
                assert RequestStatus.is_migrating(
                    backend_request.status
                ), "The status of request in migrating_out_requests_last_stage should be \
                     RequestStatus.WAITING_MIGRATING or RequestStatus.RUNNING_MIGRATING"
                if backend_request.status == RequestStatus.RUNNING_MIGRATING:
                    self.backend_engine.add_running_request(backend_request)
                else:  # WAITING_MIGRATING
                    self.backend_engine.add_waiting_request(backend_request)

    def execute_migration_method(self, method, *args, **kwargs):
        if self.is_vllm_v1:
            raise NotImplementedError(
                "KV-cache migration is unavailable for the vLLM V1 adapter"
            )
        executor = getattr(self.migration_coordinator, method)
        return executor(*args, **kwargs)

    def execute_engine_method(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return executor(*args, **kwargs)

    async def execute_engine_method_async(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return await executor(*args, **kwargs)
