# CoreX 4.4 validation

This is the reproducible test matrix for the Python 3.12 / CoreX 4.4 / vLLM
0.11.2 adaptation. Run from the checkout after sourcing the environment:

```bash
source tools/corex44_env.sh
python tools/run_corex44_validation.py unit
python tools/run_corex44_validation.py integration \
  --local-ip 10.31.10.62 --remote-ip 10.31.10.210
python tools/run_corex44_validation.py e2e --tp 2
```

The unit layer uses an isolated local Ray runtime and does not touch a shared
cluster. It includes both scheduler-level affinity tests and a Manager P/D
role-pool test, so cached prefixes are checked through the production
Prefill/Decode selection boundary as well as in isolation. Integration first compares versions, affinity hashes and the source
fingerprint. It then sends actual vLLM `BlockStored` events from the local
publisher to the V1 subscriber on `10.31.10.210` (using an SSH reverse tunnel
when arbitrary inter-node ZMQ ports are firewalled), which must rebuild the
index and rank the remote cached candidate first. Finally it runs a real BF16 GPU staging
transfer; the consumer is started on `10.31.10.210` and the producer on
`10.31.10.62`. E2E loads the local Qwen3-14B weights and validates TP=1 or
TP=2 generation.

The integration runner allocates ephemeral ports and cleans its SSH child on
failure. It never runs `ray stop`, deletes model weights, or removes `.ray-*`
directories. The replay-event unit test also retries publisher binds to absorb
the unavoidable probe-to-bind race when an unrelated local ZMQ/Ray process
claims an ephemeral port. Use `--dry-run` to inspect commands without allocating GPUs.

The legacy vLLM 0.6 block-manager migration tests are intentionally excluded
from this V1 gate because those private APIs do not exist in vLLM 0.11.2. V1
request movement is covered by connector-driven P/D KV handoff and the KV
affinity unit/integration tests.
