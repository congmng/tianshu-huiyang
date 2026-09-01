#!/usr/bin/env python3
"""Run a single vLLM chat completion on the project-local CoreX runtime.

Example:
  source tools/corex44_env.sh
  CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model \
    .conda-corex44/bin/python tools/run_vllm_chat_smoke.py
"""

import os
import pathlib
import time

from vllm import LLM, SamplingParams


def main() -> None:
    model = pathlib.Path(os.environ["MODEL_PATH"]).expanduser().resolve()
    if not (model / "config.json").is_file():
        raise SystemExit(f"MODEL_PATH is not a complete Hugging Face model directory: {model}")

    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    started = time.monotonic()
    llm = LLM(
        model=str(model),
        tensor_parallel_size=tensor_parallel_size,
        dtype="float16",
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.70")),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "256")),
        enforce_eager=True,
    )
    outputs = llm.generate(
        ["请用一句话介绍 Iluvatar CoreX。"],
        SamplingParams(temperature=0, max_tokens=24, seed=0),
    )
    completion = outputs[0].outputs[0].text.strip()
    if not completion:
        raise RuntimeError("vLLM returned an empty completion")
    print(f"model={model}")
    print(f"tensor_parallel_size={tensor_parallel_size}")
    print(f"elapsed_seconds={time.monotonic() - started:.2f}")
    print(f"completion={completion}")
    print("corex_vllm_chat: PASS")


if __name__ == "__main__":
    main()
