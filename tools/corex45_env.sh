#!/usr/bin/env bash
# Explicit CoreX 4.5 / Iluvatar TG-V300 environment wrapper.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source tools/corex45_env.sh" >&2
  exit 2
fi
export LLUMNIX_COREX_STACK=45
source "$(dirname "${BASH_SOURCE[0]}")/corex_env.sh"
