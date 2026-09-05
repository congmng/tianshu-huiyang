from pathlib import Path

from llumnix.config import get_llumnix_config
from llumnix.arg_utils import InstanceArgs, ManagerArgs


def test_corex44_v1_pd_config_uses_connector_driven_handoff():
    config = get_llumnix_config(str(Path("configs/corex44_v1_pd.yml")))
    manager = ManagerArgs.from_llumnix_config(config)
    instance = InstanceArgs.from_llumnix_config(config)
    assert manager.enable_pd_disagg is True
    assert manager.pd_ratio == [1, 1]
    assert manager.initial_instances == 2
    assert instance.migration_backend == "kvtransfer"
    assert instance.migration_backend_transfer_type == "CoreXP2pNcclConnector"
    assert instance.dispatch_load_metric == "virtual_usage"
