#!/usr/bin/env bash
# Runtime environment for the project-local CoreX 4.4.0 Llumnix validation.
# Source this file; it never modifies the host driver, toolkit, or shell RC files.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source tools/corex44_env.sh" >&2
  exit 2
fi

_corex44_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Prefer a project-local clone on the primary host. The second CoreX host
# uses the verified shared environment; callers may override either choice.
_corex44_default_env="${_corex44_root}/.conda-corex44"
if [[ ! -x "${_corex44_default_env}/bin/python" && -x "/data1/congmng/conda-envs/ds-corex44/bin/python" ]]; then
  _corex44_default_env="/data1/congmng/conda-envs/ds-corex44"
fi
export CONDA_PREFIX="${LLUMNIX_COREX_PYTHON_ENV:-${_corex44_default_env}}"
if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  echo "CoreX Python environment not found: ${CONDA_PREFIX}" >&2
  return 1
fi
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
# vLLM's sha256_cbor prefix hashes include Python's hash seed for some cache
# metadata paths. Set it before Python starts so both hosts derive identical
# ownership keys; configure_v1_kv_transfer also preserves this invariant.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

unset _corex44_root _corex44_default_env
