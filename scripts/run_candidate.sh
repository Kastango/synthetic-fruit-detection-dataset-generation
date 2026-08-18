#!/usr/bin/env bash
# Roda um candidato completo do loop de otimização: duas seeds treinadas
# (41 e 42), avaliação em real train+val (115 imagens, real-assisted) para
# cada seed, e FID contra o mesmo corpus real. Ferramenta interna do loop;
# não faz parte do protocolo confirmatório e nunca toca o teste real.
set -Eeuo pipefail

trial_id="$1"
synthesis_config="$2"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_python="${project_dir}/.venv/bin/python"
summary_dir="${project_dir}/runs/optimization/candidate_summaries"
mkdir -p "$summary_dir"

for seed in 41 42; do
  "${venv_python}" "${project_dir}/scripts/optimization_trial.py" \
    --trial-id "${trial_id}-s${seed}" \
    --synthesis-config "${synthesis_config}" \
    --train-images 104 --val-images 48 \
    --model yolov8n.pt \
    --epochs 15 --imgsz 512 --batch 8 \
    --seed "${seed}" --device cpu --workers 8

  "${venv_python}" "${project_dir}/scripts/evaluate_on_real_val.py" \
    --checkpoint "${project_dir}/runs/optimization/training/${trial_id}-s${seed}__s${seed}/weights/best.pt" \
    --name "${trial_id}-s${seed}" --include-train --device cpu \
    --output "${summary_dir}/${trial_id}-s${seed}-real.json"
done

"${venv_python}" "${project_dir}/scripts/compute_realism_gap.py" \
  --synthetic-dir "${project_dir}/data/generated/optimization/${trial_id}-s41/images/val" \
  --name "${trial_id}" --device cpu \
  --output "${summary_dir}/${trial_id}-fid.json"

"${venv_python}" "${project_dir}/scripts/summarize_candidate.py" --trial-id "${trial_id}"
