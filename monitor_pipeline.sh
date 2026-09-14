#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
session_name="fruit-training"
log_path="${project_dir}/artifacts/pipeline-live.log"
state_path="${project_dir}/artifacts/pipeline_state.json"

count_files() {
  local path="$1"
  local pattern="${2:-*}"
  if [[ -d "${path}" ]]; then
    find "${path}" -type f -name "${pattern}" 2>/dev/null | wc -l
  else
    printf '0\n'
  fi
}

render_status() {
  printf 'Pipeline de detecção de frutas — %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    pane_dead="$(tmux display-message -p -t "${session_name}" '#{pane_dead}')"
    if [[ "${pane_dead}" == "1" ]]; then
      printf 'Sessão: encerrada (tmux: %s)\n' "${session_name}"
    else
      printf 'Sessão: em execução (tmux: %s)\n' "${session_name}"
    fi
  else
    printf 'Sessão: não encontrada\n'
  fi

  if [[ -f "${state_path}" ]]; then
    jq -r '
      if .current then
        "Etapa: " + .current.stage + "\nInício: " + .current.started_at
      elif .last_failure then
        "Última falha: " + .last_failure.stage
      else
        "Etapa: nenhuma etapa ativa"
      end
    ' "${state_path}"
    printf 'Etapas concluídas: %s\n' "$(jq '.stages | length' "${state_path}")"
  fi

  printf '\nAtivos: recortes %s/127 | mapas de profundidade %s/228\n' \
    "$(count_files "${project_dir}/data/assets/regenerated/pictures_trimmed")" \
    "$(count_files "${project_dir}/data/assets/regenerated/backgrounds_map")"
  printf 'Síntese: %s imagens | Treinos concluídos: %s/42\n' \
    "$(count_files "${project_dir}/data/generated/confirmatory_pool/images" '*.jpg')" \
    "$(count_files "${project_dir}/runs/confirmatory/training" 'result.json')"

  if command -v nvidia-smi >/dev/null 2>&1; then
    printf '\nGPU:\n'
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
      --format=csv,noheader
  fi

  if [[ -f "${log_path}" ]]; then
    printf '\nÚltimas atualizações:\n'
    # Ultralytics atualiza a mesma linha usando carriage returns. Limitar os
    # bytes antes de normalizá-los evita imprimir o histórico inteiro.
    tail -c 32768 "${log_path}" \
      | tr '\r' '\n' \
      | sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g' \
      | tail -n 12
  fi
}

case "${1:-live}" in
  live)
    while true; do
      clear
      render_status
      sleep 5
    done
    ;;
  once|status)
    render_status
    ;;
  logs|log)
    touch "${log_path}"
    exec tail -n 100 -F "${log_path}"
    ;;
  attach)
    exec tmux attach-session -t "${session_name}"
    ;;
  *)
    echo "uso: $0 [live|once|logs|attach]" >&2
    exit 2
    ;;
esac
