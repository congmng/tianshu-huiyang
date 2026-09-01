#!/usr/bin/env bash
# Runtime environment for the project-local CoreX 4.4.0 Llumnix validation.
# Source this file; it never modifies the host driver, toolkit, or shell RC files.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source tools/corex44_env.sh" >&2
  exit 2
fi

_corex44_root="/data1/congmng/llumnix"
export CONDA_PREFIX="${_corex44_root}/.conda-corex44"
export PATH="/usr/local/corex-4.4.0/bin:${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/corex-4.4.0/lib64:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Triton/Inductor compiles a small CUDA-driver helper at runtime for
# tensor-parallel vLLM workers.  CoreX provides cuda.h here, while the
# vendor wheel otherwise only searches /usr/local/include.
export CPATH="/usr/local/corex-4.4.0/include${CPATH:+:${CPATH}}"
export C_INCLUDE_PATH="/usr/local/corex-4.4.0/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="/usr/local/corex-4.4.0/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

# CoreX single-node settings verified by tools/corex44_smoke.py and vLLM BGE.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_ENFORCE_CUDA_GRAPH="${VLLM_ENFORCE_CUDA_GRAPH:-0}"

unset _corex44_root
