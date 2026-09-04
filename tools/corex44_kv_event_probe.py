#!/usr/bin/env python3
"""Cross-host vLLM V1 KV-event and affinity integration probe."""
from __future__ import annotations
import argparse, json, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex, KVEventSubscriber

def args():
    p = argparse.ArgumentParser(); p.add_argument('--role', choices=('publisher','consumer'), required=True)
    p.add_argument('--host', required=True); p.add_argument('--port', type=int, required=True)
    p.add_argument('--timeout', type=float, default=15); return p.parse_args()

def hashes(): return KVCacheAffinityIndex().prefix_hashes((1,2,3,4,5,6,7,8), 4, 'sha256_cbor')

def batch(values):
    from vllm.distributed.kv_events import BlockStored, KVEventBatch
    return KVEventBatch(ts=time.time(), events=[BlockStored(values, None, [1,2,3,4], 4, None, 'GPU')])

def main():
    a = args(); endpoint = f'tcp://{a.host}:{a.port}'; values = hashes()
    if a.role == 'publisher':
        from vllm.config import KVEventsConfig
        from vllm.distributed.kv_events import EventPublisherFactory
        # vLLM interprets a concrete TCP endpoint as a *connect* target.
        # The event owner must bind, while consumers connect to the supplied
        # concrete host; keep that distinction explicit in this probe.
        pub = EventPublisherFactory.create(KVEventsConfig(enable_kv_cache_events=True, publisher='zmq', endpoint=f'tcp://*:{a.port}'))
        try:
            n = 0; end = time.monotonic() + a.timeout
            while time.monotonic() < end: pub.publish(batch(values)); n += 1; time.sleep(.25)
            print(json.dumps({'role':'publisher','events_sent':n,'hashes':[x.hex() for x in values]}, sort_keys=True), flush=True)
        finally: pub.shutdown()
    else:
        index = KVCacheAffinityIndex(); ready = threading.Event()
        def apply(events): index.apply('remote-cached', events); ready.set()
        sub = KVEventSubscriber(endpoint, apply)
        try:
            if not ready.wait(a.timeout): raise RuntimeError('timed out waiting for vLLM KV event')
            affinity = index.affinity('remote-cached', values); rank = index.rank(values, ('local-empty','remote-cached'))
            if affinity != 1.0 or rank != ['remote-cached','local-empty']: raise RuntimeError(f'unexpected result {affinity=} {rank=}')
            print(json.dumps({'role':'consumer','affinity':affinity,'rank':rank,'hashes':[x.hex() for x in values]}, sort_keys=True), flush=True)
        finally: sub.close()
if __name__ == '__main__': main()
