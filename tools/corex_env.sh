#!/usr/bin/env bash
# CoreX stack environment for Llumnix.
#
# Supported values for LLUMNIX_COREX_STACK are ``44`` and ``45``.  When unset,
# the script infers the stack from the active ``/usr/local/corex`` symlink or
# the project-local fallback environment.  ``tools/corex44_env.sh`` and
# ``tools/corex45_env.sh`` remain thin wrappers for deployment scripts that
# need an explicit stack.
#
# The script only exports variables and never modifies host drivers or shell
# RC files.  Source it:
#     source tools/corex_env.sh
# or, for the CoreX 4.5 V300 node:
#     LLUMNIX_COREX_STACK=45 source tools/corex_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

_corex_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${LLUMNIX_COREX_STACK:-}" ]]; then
  _corex_stack="${LLUMNIX_COREX_STACK}"
else
  _corex_stack="44"
  if [[ -n "${LLUMNIX_COREX_ROOT:-}" ]]; then
    _corex_stack="$(basename "${LLUMNIX_COREX_ROOT}" | sed 's/^corex-//' | tr -d '.')"
    _corex_stack="${_corex_stack%%-*}"
  elif [[ -f /usr/local/corex/release-corex.txt ]]; then
    if grep -q '4\.5\.0' /usr/local/corex/release-corex.txt 2>/dev/null; then
      _corex_stack="45"
    fi
  fi
fi

case "${_corex_stack}" in
  44|4.4|4.4.0)
    _corex_stack="44"
    _corex_sdk_root="${LLUMNIX_COREX_ROOT:-/usr/local/corex-4.4.0}"
    ;;
  45|4.5|4.5.0)
    _corex_stack="45"
    _corex_sdk_root="${LLUMNIX_COREX_ROOT:-/usr/local/corex-4.5.0}"
    ;;
  *)
    echo "Unsupported LLUMNIX_COREX_STACK=${LLUMNIX_COREX_STACK}; expected 44 or 45" >&2
    return 1
    ;;
esac

# Prefer a project-local Python environment on the primary host. The second
# CoreX host uses the verified shared environment; callers may override either
# choice with LLUMNIX_COREX_PYTHON_ENV.
_corex_default_env=""
if [[ "${_corex_stack}" == "45" ]]; then
  _corex_candidates=(
    "${_corex_project_root}/.conda-corex45"
    "/data1/congmng/conda-envs/ds-corex45"
    "${CONDA_PREFIX:-}"
  )
else
  _corex_candidates=(
    "${_corex_project_root}/.conda-corex44"
    "/data1/congmng/conda-envs/ds-corex44"
    "${CONDA_PREFIX:-}"
  )
fi
for _corex_candidate in "${_corex_candidates[@]}"; do
  if [[ -n "${_corex_candidate}" && -x "${_corex_candidate}/bin/python" ]]; then
    _corex_default_env="${_corex_candidate}"
    break
  fi
done

export CONDA_PREFIX="${LLUMNIX_COREX_PYTHON_ENV:-${_corex_default_env}}"
if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  echo "CoreX ${_corex_stack} Python environment not found: ${CONDA_PREFIX}" >&2
  return 1
fi

export LLUMNIX_COREX_STACK="${_corex_stack}"
export LLUMNIX_COREX_ROOT="${_corex_sdk_root}"

# CoreX 4.5 V300 nodes split vendor runtime libraries across the toolkits
# under /data/tianshu/20260720/corex/corex/corex-toolkit.  The 4.4 image
# consolidates them under /usr/local/corex-4.4.0/lib64; include both layouts
# so the same sourceable script works on either deployment.
_corex_lib_paths=("${_corex_sdk_root}/lib64")
_corex_include_paths=("${_corex_sdk_root}/include")
_corex_toolkit_root="${LLUMNIX_COREX_TOOLKIT_ROOT:-/data/tianshu/20260720/corex/corex/corex-toolkit}"
if [[ -d "${_corex_toolkit_root}" ]]; then
  while IFS= read -r _corex_libdir; do
    _corex_lib_paths+=("${_corex_libdir}")
  done < <(find "${_corex_toolkit_root}" -mindepth 2 -maxdepth 2 -type d -name lib64 | sort)
  while IFS= read -r _corex_incdir; do
    _corex_include_paths+=("${_corex_incdir}")
  done < <(find "${_corex_toolkit_root}" -mindepth 2 -maxdepth 2 -type d -name include | sort)
fi
if [[ -d "/data/tianshu/20260720/driver/corex/lib64" ]]; then
  _corex_lib_paths+=("/data/tianshu/20260720/driver/corex/lib64")
fi
# The 4.5 vllm_iluvatar plugin ships native torch/ixformer extensions in the
# Python site-packages tree rather than the SDK lib64 directory.  Expose them
# so importing ``vllm_iluvatar._C`` can resolve its shared-library deps on
# V300 nodes without changing the 4.4 runtime path.
if [[ "${_corex_stack}" == "45" ]]; then
  _corex_ixsdk_root="${_corex_toolkit_root}/ixsdk"
  _corex_libdevice_path="${_corex_ixsdk_root}/nvvm/libdevice/libdevice.compute_bi.10.bc"
  if [[ -d "${_corex_ixsdk_root}" ]]; then
    export CUDA_HOME="${CUDA_HOME:-${_corex_ixsdk_root}}"
  fi
  if [[ -f "${_corex_libdevice_path}" ]]; then
    export TRITON_LIBDEVICE_PATH="${TRITON_LIBDEVICE_PATH:-${_corex_libdevice_path}}"
  fi

  for _corex_python_libdir in \
      "${CONDA_PREFIX}/lib/python3.12/site-packages/torch/lib" \
      "${CONDA_PREFIX}/lib/python3.12/site-packages/ixformer"; do
    if [[ -d "${_corex_python_libdir}" ]]; then
      _corex_lib_paths+=("${_corex_python_libdir}")
    fi
  done

  # ``.conda-corex45`` is a venv layered on the shared CoreX 4.5 base env.
  # Python processes .pth files in the venv site-packages before adding the
  # base env site-packages to sys.path.  The Iluvatar startup .pth imports
  # torch while applying its early compat patch, so expose the base site-packages
  # through PYTHONPATH to make torch/typing_extensions importable at that stage.
  if [[ -f "${CONDA_PREFIX}/pyvenv.cfg" ]]; then
    _corex_venv_home="$(awk -F ' = ' '$1 == "home" {print $2}' "${CONDA_PREFIX}/pyvenv.cfg")"
    if [[ -n "${_corex_venv_home}" ]]; then
      _corex_base_prefix="${_corex_venv_home%/bin}"
      _corex_base_site="${_corex_base_prefix}/lib/python3.12/site-packages"
      if [[ -d "${_corex_base_site}" ]]; then
        export PYTHONPATH="${CONDA_PREFIX}/lib/python3.12/site-packages:${_corex_base_site}${PYTHONPATH:+:${PYTHONPATH}}"
      fi
    fi
  fi
fi
_corex_ld_library_path="$(IFS=:; echo "${_corex_lib_paths[*]}"):${CONDA_PREFIX}/lib"
_corex_cpath="$(IFS=:; echo "${_corex_include_paths[*]}"):${CONDA_PREFIX}/include"

export PATH="${_corex_sdk_root}/bin:${CONDA_PREFIX}/bin:${PATH}"
if [[ -n "${LLUMNIX_RAY_OVERLAY:-}" && -d "${LLUMNIX_RAY_OVERLAY}" ]]; then
  export PYTHONPATH="${LLUMNIX_RAY_OVERLAY}:${PYTHONPATH:+${PYTHONPATH}}"
fi
export LD_LIBRARY_PATH="${_corex_ld_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${_corex_ld_library_path}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CPATH="${_corex_cpath}${CPATH:+:${CPATH}}"
export C_INCLUDE_PATH="${_corex_cpath}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export CPLUS_INCLUDE_PATH="${_corex_cpath}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

# CoreX single-node settings verified by tools/corex44_smoke.py and vLLM BGE.
# They are also required by the CoreX 4.5 NCCL/trition stack.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_ENFORCE_CUDA_GRAPH="${VLLM_ENFORCE_CUDA_GRAPH:-0}"
# vLLM's sha256_cbor prefix hashes include Python's hash seed for some cache
# metadata paths. Set it before Python starts so both hosts derive identical
# ownership keys; configure_v1_kv_transfer also preserves this invariant.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

unset _corex_project_root _corex_stack _corex_sdk_root _corex_candidates _corex_candidate _corex_default_env _corex_lib_paths _corex_include_paths _corex_toolkit_root _corex_ixsdk_root _corex_libdevice_path _corex_libdir _corex_incdir _corex_ld_library_path _corex_cpath _corex_python_libdir
