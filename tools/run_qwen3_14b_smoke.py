#!/usr/bin/env python3
"""Run one Qwen3-14B chat completion through CoreX vLLM 0.11.2.

Usage:
  source tools/corex44_env.sh
  CUDA_VISIBLE_DEVICES=0 python tools/run_qwen3_14b_smoke.py
"""

import pathlib
import os
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vllm import LLM, SamplingParams

MODEL = pathlib.Path(__file__).resolve().parents[1] / ".models" / "Qwen3-14B"
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.96"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "512"))


def main() -> None:
    if not (MODEL / "model.safetensors.index.json").is_file():
        raise SystemExit(f"Qwen3-14B weights are incomplete: {MODEL}")

    started = time.monotonic()
    llm = LLM(
        model=str(MODEL),
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )
    messages = [
        {"role": "system", "content": "You are a concise and helpful assistant."},
        {"role": "user", "content": "用一句话介绍Iluvatar CoreX。"},
    ]
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    outputs = llm.generate(
        [prompt], SamplingParams(temperature=0.0, max_tokens=32, seed=0)
    )
    text = outputs[0].outputs[0].text.strip()
    if not text:
        raise RuntimeError("Qwen3 returned an empty completion")
    print(f"model={MODEL}")
    print(f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}")
    print(f"max_model_len={MAX_MODEL_LEN}")
    print(f"elapsed_seconds={time.monotonic() - started:.2f}")
    print(f"completion={text}")
    print("qwen3_14b_corex_vllm: PASS")


if __name__ == "__main__":
    main()
