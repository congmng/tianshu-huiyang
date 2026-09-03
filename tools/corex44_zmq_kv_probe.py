#!/usr/bin/env python3
"""Verify one real CoreX GPU BF16 KV-staging transfer across two hosts.

Run the consumer first, then the producer.  The tool deliberately exercises
the same ``CoreXZmqP2pEngine`` used by the V1 P/D connector: GPU tensor ->
CPU wire buffer -> TCP/ZMQ -> CPU buffer -> peer GPU tensor.  It neither
starts Ray nor loads a model, so it is a fast deployment gate before a full
Qwen P/D run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from llumnix.backends.vllm.corex_p2p_connector import CoreXZmqP2pEngine


class Config:
    def __init__(self, host: str, port: int, timeout_s: float):
        self.kv_ip = host
        self.kv_port = port
        self._timeout_s = timeout_s

    def get_from_extra_config(self, name, default):
        return self._timeout_s if name == "zmq_recv_timeout_s" else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("producer", "consumer"), required=True)
    parser.add_argument("--host", required=True, help="this host's routable IP")
    parser.add_argument("--port", type=int, required=True, help="this host's listen port")
    parser.add_argument("--peer", help="peer host:port; required for producer")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.role == "producer" and not args.peer:
        raise SystemExit("--peer is required for producer")
    if not torch.cuda.is_available():
        raise SystemExit("CoreX CUDA device is required")
    engine = CoreXZmqP2pEngine(args.device, Config(args.host, args.port, args.timeout))
    tensor_id = "corex-gpu-bf16-probe#layer"
    try:
        if args.role == "producer":
            expected = torch.arange(16, device=engine.device, dtype=torch.bfloat16)
            expected = expected.reshape(4, 4)
            if not engine.send_tensor(tensor_id, expected, args.peer):
                raise RuntimeError("peer rejected BF16 payload")
            print(
                f"PASS role=producer device={engine.device} dtype={expected.dtype} "
                f"shape={tuple(expected.shape)} peer={args.peer}", flush=True
            )
        else:
            received = engine.recv_tensor(tensor_id)
            expected = torch.arange(16, device=engine.device, dtype=torch.bfloat16)
            expected = expected.reshape(4, 4)
            torch.testing.assert_close(received, expected)
            if received.device != engine.device:
                raise RuntimeError(f"received on {received.device}, expected {engine.device}")
            print(
                f"PASS role=consumer device={received.device} dtype={received.dtype} "
                f"shape={tuple(received.shape)} mean={received.float().mean().item():.1f}",
                flush=True,
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
