#!/usr/bin/env python3
"""Exercise CoreX's native NCCL P2P communicator without loading a model.

Start a consumer first, then a producer.  Unlike the ZMQ-staging gate this
uses vLLM's actual ``P2pNcclEngine`` and therefore isolates communicator
initialisation, stream synchronisation, and ``ncclSend``/``ncclRecv`` from
the V1 scheduler and model allocator.  It is intentionally a diagnostic,
not a production transport selector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the shim first filters only CoreX's unavailable optional window
# symbols and replaces vLLM's forced CUMEM=1 context with CUMEM=0.
from llumnix.backends.vllm.corex_p2p_connector import (  # noqa: E402
    CoreXNcclP2pEngine,
)
import torch  # noqa: E402


class Config(SimpleNamespace):
    def get_from_extra_config(self, name, default):
        return getattr(self, "extra", {}).get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("producer", "consumer"), required=True)
    parser.add_argument("--host", required=True, help="local routable address")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--peer", help="consumer host:port; required for producer")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-channels", type=int, default=1)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.role == "producer" and not args.peer:
        raise SystemExit("--peer is required for producer")
    if not torch.cuda.is_available():
        raise SystemExit("CoreX CUDA device is required")
    config = Config(
        kv_ip=args.host,
        kv_port=args.port,
        kv_buffer_size=1 << 30,
        extra={"send_type": "PUT", "nccl_num_channels": str(args.num_channels)},
    )
    # The upstream engine otherwise calls vLLM's generic ``get_ip``. Passing
    # the probe address explicitly makes loopback and multi-NIC diagnostics
    # reproducible and ensures the ZMQ identity matches the advertised peer.
    engine = CoreXNcclP2pEngine(args.device, config, hostname=args.host)
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    try:
        if args.role == "consumer":
            for round_id in range(args.rounds):
                tensor_id = f"corex-native-nccl-probe#{round_id}"
                received = engine.recv_tensor(tensor_id, args.peer)
                expected = torch.arange(args.elements, device=engine.device, dtype=torch.float16)
                torch.testing.assert_close(received, expected)
            print(
                f"PASS role=consumer device={received.device} elements={received.numel()} "
                f"rounds={args.rounds} mean={received.float().mean().item():.1f}",
                flush=True,
            )
            return
        expected = torch.arange(args.elements, device=engine.device, dtype=torch.float16)
        for round_id in range(args.rounds):
            tensor_id = f"corex-native-nccl-probe#{round_id}"
            if not engine.send_tensor(tensor_id, expected, args.peer):
                raise RuntimeError("consumer rejected native NCCL tensor")
        print(
            f"PASS role=producer device={expected.device} elements={expected.numel()} rounds={args.rounds} "
            f"peer={args.peer}",
            flush=True,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
