from typing import Tuple

from vllm.engine.arg_utils import AsyncEngineArgs

from llumnix.logging.logger import init_logger
from llumnix.backends.backend_interface import BackendType
from llumnix.backends.vllm.utils import check_engine_args
from llumnix.arg_utils import EntrypointsArgs, ManagerArgs, InstanceArgs, LlumnixArgumentParser
from llumnix.entrypoints.utils import LaunchMode

logger = init_logger(__name__)


def add_cli_args(parser: LlumnixArgumentParser) -> "Namespace":
    parser.set_namespace("llumnix")
    parser = EntrypointsArgs.add_cli_args(parser)
    parser = ManagerArgs.add_cli_args(parser)
    parser = InstanceArgs.add_cli_args(parser)
    parser.set_namespace("vllm")
    parser = AsyncEngineArgs.add_cli_args(parser)
    cli_args = parser.parse_args()

    return cli_args

def get_args(cfg, launch_mode: LaunchMode, parser: LlumnixArgumentParser, cli_args: "Namespace") \
        -> Tuple[EntrypointsArgs, ManagerArgs, InstanceArgs, AsyncEngineArgs]:
    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    instance_args: InstanceArgs = InstanceArgs.from_llumnix_config(cfg)
    instance_args.init_from_engine_args(engine_args, BackendType.VLLM)
    manager_args = ManagerArgs.from_llumnix_config(cfg)
    manager_args.init_from_instance_args(instance_args)
    entrypoints_args = EntrypointsArgs.from_llumnix_config(cfg)

    EntrypointsArgs.check_args(entrypoints_args, parser)
    ManagerArgs.check_args(manager_args, parser)
    InstanceArgs.check_args(instance_args, manager_args, launch_mode, parser)
    # vLLM 0.11 V1 removed ``worker_use_ray`` and the old per-worker engine
    # path.  The V1 adapter owns its worker lifecycle, so only apply legacy
    # validation to old vLLM installations.
    import vllm
    if not getattr(vllm, "__version__", "").startswith("0.11"):
        check_engine_args(engine_args, instance_args)
    else:
        # V1 migration is connector-based. Keep the manager migration loop
        # disabled for the legacy coordinator; ``kvtransfer`` is configured
        # later when the V1 backend is constructed.
        if instance_args.migration_backend != "kvtransfer":
            manager_args.enable_migration = False
            logger.warning(
                "vLLM %s detected: legacy KV-cache migration is disabled; "
                "set migration_backend=kvtransfer to use V1 connectors.",
                vllm.__version__,
            )

    logger.info("entrypoints_args: {}".format(entrypoints_args))
    logger.info("manager_args: {}".format(manager_args))
    logger.info("instance_args: {}".format(instance_args))
    logger.info("engine_args: {}".format(engine_args))

    return entrypoints_args, manager_args, instance_args, engine_args
