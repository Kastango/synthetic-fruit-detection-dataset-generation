#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -n "${PYTHON_COMMAND:-}" ]]; then
  python_command="${PYTHON_COMMAND}"
elif command -v python3.11 >/dev/null 2>&1; then
  python_command="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
  python_command="python3.12"
else
  echo "Python 3.11 ou 3.12 é necessário. Defina PYTHON_COMMAND." >&2
  exit 2
fi

venv_dir="${PIPELINE_VENV:-${project_dir}/.venv}"
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_command}" -m venv "${venv_dir}"
fi
if ! "${venv_dir}/bin/python" -m pip --version >/dev/null 2>&1; then
  "${venv_dir}/bin/python" -m ensurepip --upgrade
fi

case "${stage}" in
  preprocess)
    requirements="requirements-preprocess.txt"
    ;;
  train|select|test|all)
    requirements="requirements.txt"
    ;;
  *)
    requirements="requirements-base.txt"
    ;;
esac

"${venv_dir}/bin/python" -m pip install --disable-pip-version-check \
  --requirement "${project_dir}/${requirements}"

cd "${project_dir}"
exec "${venv_dir}/bin/python" scripts/reproduce.py "${stage}" "$@"
