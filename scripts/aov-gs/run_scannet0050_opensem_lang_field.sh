#!/usr/bin/env bash
##################################################
# ScanNet scene0050_02 — LangSplatV2 language field (s, m, l) + NVS validation.
#
# Runs:
#   results/ScanNet/scene0050_02/ActiveOpenSem/run_0
#   results/ScanNet/scene0050_02/ActiveOpenSem_replica_nbv/run_0
#
# Scheduling: exactly 1 train job per GPU (cuda:GPU0 + cuda:GPU1 in parallel).
#
# Train prerequisites per run dir:
#   splatam/final/params*.npz, keyframe_poses.json, language_features/
#
# Validate prerequisites:
#   lang_field_{s,m,l}k64_l1/lang_field.pt per run
#   data/scannet_sim_nvs/scene0050_02/traj.txt
#   data/scannet_sim_nvs/scene0050_02/results_habitat/semantic/semantic*.npy
#   data/ScanNet/scene0050_02/traj.txt  (GS train-frame alignment)
#
# Usage (from AOV-GS-V2 root):
#   bash scripts/aov-gs/run_scannet0050_opensem_lang_field.sh
#
# Single phase:
#   PHASE=train bash scripts/aov-gs/run_scannet0050_opensem_lang_field.sh
#   PHASE=validate bash scripts/aov-gs/run_scannet0050_opensem_lang_field.sh
#
# Background:
#   mkdir -p logs/scannet0050_opensem_lang_field
#   nohup bash scripts/aov-gs/run_scannet0050_opensem_lang_field.sh \
#     > logs/scannet0050_opensem_lang_field/orchestrator.log 2>&1 &
#   echo $! > logs/scannet0050_opensem_lang_field/orchestrator.pid
#   tail -f logs/scannet0050_opensem_lang_field/orchestrator.log
#
# Env:
#   PHASE=all|train|validate   (default all)
#   GPU0=0 GPU1=1
#   RUN_TAG=run_0 NVS_ROOT=data/scannet_sim_nvs
#   NUM_ITERS=12000 TRAIN_DOWNSCALE=0.5
#   RENDER_CHECKPOINT=auto
#   GPU0=0 GPU1=1   (slot 0 → cuda:GPU0, slot 1 → cuda:GPU1, never 2 jobs on one GPU)
#   SKIP_DONE=1 FORCE_RETRAIN=0
#   LOG_DIR=logs/scannet0050_opensem_lang_field
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

SCENE="scene0050_02"
PHASE="${PHASE:-all}"
RUN_TAG="${RUN_TAG:-run_0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPUS=("${GPU0}" "${GPU1}")
DEVICE="cuda:0"
NVS_ROOT="${NVS_ROOT:-data/scannet_sim_nvs}"
CLASS_INFO="${CLASS_INFO:-configs/ScanNet/class_info_file.json}"
TRAIN_TRAJ="${TRAIN_TRAJ:-data/ScanNet/${SCENE}/traj.txt}"

SKIP_DONE="${SKIP_DONE:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
NUM_ITERS="${NUM_ITERS:-12000}"
TRAIN_DOWNSCALE="${TRAIN_DOWNSCALE:-0.5}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-64}"
VQ_LAYER_NUM="${VQ_LAYER_NUM:-1}"
TOPK="${TOPK:-4}"
RENDER_CHECKPOINT="${RENDER_CHECKPOINT:-auto}"
LOG_EVERY="${LOG_EVERY:-500}"
MIN_RECOVER_ITER_FRAC="${MIN_RECOVER_ITER_FRAC:-90}"

RUN_RELS=(
  "results/ScanNet/${SCENE}/ActiveOpenSem/${RUN_TAG}"
  "results/ScanNet/${SCENE}/ActiveOpenSem_replica_nbv/${RUN_TAG}"
)

LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/scannet0050_opensem_lang_field}"
mkdir -p "${LOG_DIR}"

set +u
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate aov-gs 2>/dev/null || true
elif [[ -f /home/optimus/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/optimus/anaconda3/etc/profile.d/conda.sh
  conda activate aov-gs 2>/dev/null || true
fi
set -u

if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x /home/optimus/anaconda3/envs/aov-gs/bin/python ]]; then
  PY=/home/optimus/anaconda3/envs/aov-gs/bin/python
elif [[ -x /home/optimus/anaconda3/envs/active-gs/bin/python ]]; then
  PY=/home/optimus/anaconda3/envs/active-gs/bin/python
elif [[ -x /home/optimus/anaconda3/envs/active-sgm/bin/python ]]; then
  PY=/home/optimus/anaconda3/envs/active-sgm/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

export PYTHONPATH="${PROJ_DIR}:${PROJ_DIR}/third_parties/splatam:${PYTHONPATH:-}"

log() { echo "[$(date -Is)] $*"; }

_run_abs() {
  local rel="$1"
  if [[ "${rel}" = /* ]]; then
    echo "${rel}"
  else
    echo "${PROJ_DIR}/${rel}"
  fi
}

_run_label() {
  local rel="$1"
  local abs
  abs="$(_run_abs "${rel}")"
  echo "$(basename "$(dirname "${abs}")")_$(basename "${abs}")"
}

_level_done() {
  local rd="$1"
  local level="$2"
  [[ -f "${rd}/lang_field_${level}k${CODEBOOK_SIZE}_l${VQ_LAYER_NUM}/lang_field.pt" ]]
}

_best_iteration() {
  local best="$1"
  [[ -f "${best}" ]] || { echo 0; return 0; }
  "${PY}" - "${best}" <<'PY'
import sys
import torch
d = torch.load(sys.argv[1], map_location="cpu")
print(int(d.get("best_iteration", 0)))
PY
}

_min_recover_iters() {
  echo $(( NUM_ITERS * MIN_RECOVER_ITER_FRAC / 100 ))
}

_recover_level_from_best() {
  local rd="$1"
  local level="$2"
  local label="$3"
  local dir="${rd}/lang_field_${level}k${CODEBOOK_SIZE}_l${VQ_LAYER_NUM}"
  local best="${dir}/best.pt"
  local final="${dir}/lang_field.pt"
  local best_it min_it

  [[ -f "${final}" ]] && return 0
  [[ -f "${best}" ]] || return 1

  best_it="$(_best_iteration "${best}")"
  min_it="$(_min_recover_iters)"
  if [[ "${best_it}" -lt "${min_it}" ]]; then
    log "SKIP RECOVER ${label}/${level}: best.pt @ iter ${best_it} < ${min_it}"
    return 1
  fi

  log "RECOVER ${label}/${level}: best.pt (iter ${best_it}) → lang_field.pt"
  cp -a "${best}" "${final}"
  return 0
}

_check_prereqs() {
  local rel="$1"
  local rd="$2"
  local missing=()

  [[ -d "${rd}" ]] || missing+=("(dir missing)")
  [[ -d "${rd}/language_features" ]] || missing+=("language_features/")
  [[ -f "${rd}/keyframe_poses.json" ]] || missing+=("keyframe_poses.json")
  if [[ ! -f "${rd}/splatam/final/params.npz" && ! -f "${rd}/splatam/final/params0.npz" ]]; then
    missing+=("splatam/final/params*.npz")
  fi

  if [[ "${#missing[@]}" -gt 0 ]]; then
    log "ERROR ${rel}: missing ${missing[*]}"
    return 1
  fi
  return 0
}

_train_level() {
  local rel="$1"
  local level="$2"
  local gpu="$3"
  local label log_file
  label="$(_run_label "${rel}")"
  log_file="${LOG_DIR}/train_${label}_${level}_gpu${gpu}.log"

  log "TRAIN ${label}/${level} on cuda:${gpu} → ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  bash "${SCRIPT_DIR}/03_train_gaussian_lang_field.sh" \
    "${rel}" \
    "${CODEBOOK_SIZE}" \
    "${level}" \
    "${NUM_ITERS}" \
    "${DEVICE}" \
    "${VQ_LAYER_NUM}" \
    "${TOPK}" \
    "${RENDER_CHECKPOINT}" \
    "${TRAIN_DOWNSCALE}" \
    "${LOG_EVERY}" >"${log_file}" 2>&1
}

_run_train_pool() {
  local -a queue=()
  local rel rd lvl label

  for rel in "${RUN_RELS[@]}"; do
    rd="$(_run_abs "${rel}")"
    label="$(_run_label "${rel}")"

    if [[ "${FORCE_RETRAIN}" == "1" ]]; then
      log "FORCE_RETRAIN=1 — removing lang_field_* for ${label}"
      rm -rf "${rd}"/lang_field_*
    fi

    for lvl in s m l; do
      if _level_done "${rd}" "${lvl}" && [[ "${SKIP_DONE}" == "1" ]]; then
        log "SKIP ${label}/${lvl}: checkpoint exists"
        continue
      fi
      queue+=("${rel}|${lvl}")
    done
  done

  if [[ "${#queue[@]}" -eq 0 ]]; then
    log "All levels done for all runs."
    return 0
  fi

  log "Train pool: ${#queue[@]} jobs — 1 per GPU (cuda:${GPU0}, cuda:${GPU1})"
  log "  downscale=${TRAIN_DOWNSCALE} render_ckpt=${RENDER_CHECKPOINT}"
  for item in "${queue[@]}"; do
    log "  queued ${item%|*}/${item#*|}"
  done

  local qi=0 fail=0
  local -a failed_jobs=()
  local -a SLOT_PIDS=("" "")
  local -a SLOT_RELS=("" "")
  local -a SLOT_LEVELS=("" "")

  _reap_slots() {
    local i pid rc rd lbl lvl gpu
    for i in 0 1; do
      pid="${SLOT_PIDS[$i]}"
      [[ -n "${pid}" ]] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        continue
      fi
      set +e
      wait "${pid}"
      rc=$?
      set -e
      lbl="$(_run_label "${SLOT_RELS[$i]}")"
      rd="$(_run_abs "${SLOT_RELS[$i]}")"
      lvl="${SLOT_LEVELS[$i]}"
      gpu="${GPUS[$i]}"
      if [[ "${rc}" -ne 0 ]]; then
        log "FAIL ${lbl}/${lvl} gpu${gpu} (rc=${rc})"
        tail -25 "${LOG_DIR}/train_${lbl}_${lvl}_gpu${gpu}.log" 2>/dev/null || true
        if _recover_level_from_best "${rd}" "${lvl}" "${lbl}"; then
          log "RECOVERED ${lbl}/${lvl} from best.pt"
        else
          failed_jobs+=("${lbl}/${lvl}")
          fail=1
        fi
      elif ! _level_done "${rd}" "${lvl}"; then
        _recover_level_from_best "${rd}" "${lvl}" "${lbl}" \
          || { failed_jobs+=("${lbl}/${lvl}"); fail=1; }
      else
        log "DONE ${lbl}/${lvl} gpu${gpu}"
      fi
      SLOT_PIDS[$i]=""
      SLOT_RELS[$i]=""
      SLOT_LEVELS[$i]=""
    done
  }

  _start_slot() {
    local slot="$1"
    [[ -z "${SLOT_PIDS[$slot]}" ]] || return 1
    [[ "${qi}" -lt ${#queue[@]} ]] || return 1

    local item="${queue[$qi]}"
    local r="${item%|*}"
    local l="${item#*|}"
    qi=$((qi + 1))
    _train_level "${r}" "${l}" "${GPUS[$slot]}" &
    SLOT_PIDS[$slot]=$!
    SLOT_RELS[$slot]="${r}"
    SLOT_LEVELS[$slot]="${l}"
    log "START $(_run_label "${r}")/${l} → cuda:${GPUS[$slot]} only (pid=${SLOT_PIDS[$slot]})"
  }

  _any_running() {
    [[ -n "${SLOT_PIDS[0]}" || -n "${SLOT_PIDS[1]}" ]]
  }

  while [[ "${qi}" -lt ${#queue[@]} ]] || _any_running; do
    _reap_slots
    _start_slot 0 || true
    _start_slot 1 || true
    _any_running && sleep 3
  done

  if [[ "${#failed_jobs[@]}" -gt 0 ]]; then
    log "Failed jobs: ${failed_jobs[*]}"
    return 1
  fi
  return 0
}

_level_ready() {
  _level_done "$1" "$2"
}

_all_levels_done() {
  local rd="$1"
  local lvl
  for lvl in s m l; do
    _level_ready "${rd}" "${lvl}" || return 1
  done
  return 0
}

_info_semantic_scannet() {
  local out="${PROJ_DIR}/configs/ScanNet/nyu40_info_semantic.json"
  if [[ -f "${out}" ]]; then
    echo "${out}"
    return 0
  fi
  log "Building ${out} (NYU40 ids for Habitat GT semantics)"
  (
    export PYTHONPATH="${PROJ_DIR}:${PROJ_DIR}/scripts:${PYTHONPATH:-}"
    "${PY}" -c "
import lang_field_eval_utils as lfu
from pathlib import Path
out = Path('${out}')
lfu.write_scannet_nyu40_info_semantic(out)
print(out)
"
  )
}

_check_validate_prereqs() {
  local traj="${PROJ_DIR}/${NVS_ROOT}/${SCENE}/traj.txt"
  local sem_dir="${PROJ_DIR}/${NVS_ROOT}/${SCENE}/results_habitat/semantic"
  local train_traj="${PROJ_DIR}/${TRAIN_TRAJ}"
  local missing=()

  [[ -f "${traj}" ]] || missing+=("${NVS_ROOT}/${SCENE}/traj.txt")
  if [[ ! -d "${sem_dir}" ]] || {
    [[ -z "$(ls -A "${sem_dir}"/semantic*.npy 2>/dev/null)" ]] \
      && [[ -z "$(ls -A "${sem_dir}"/semantic_map_*.npy 2>/dev/null)" ]]
  }; then
    missing+=("${NVS_ROOT}/${SCENE}/results_habitat/semantic/semantic*.npy")
  fi
  [[ -f "${train_traj}" ]] || missing+=("${TRAIN_TRAJ}")

  if [[ "${#missing[@]}" -gt 0 ]]; then
    log "ERROR validate: missing ${missing[*]}"
    return 1
  fi
  return 0
}

_validate_run() {
  local rel="$1"
  local gpu="$2"
  local rd label traj out summary log_file info_sem
  rd="$(_run_abs "${rel}")"
  label="$(_run_label "${rel}")"
  traj="${PROJ_DIR}/${NVS_ROOT}/${SCENE}/traj.txt"
  out="${rd}/lang_field_traj_eval"
  summary="${out}/miou_summary.txt"
  log_file="${LOG_DIR}/validate_${label}_gpu${gpu}.log"

  log "VALIDATE ${label} on cuda:${gpu} → ${log_file}"

  if ! _all_levels_done "${rd}"; then
    log "ERROR ${label}: not all lang_field levels ready"
    return 1
  fi

  if [[ "${SKIP_DONE}" == "1" && -f "${summary}" ]]; then
    log "SKIP ${label}: ${summary} exists"
    grep "Overall mIoU" "${summary}" 2>/dev/null || true
    return 0
  fi

  info_sem="$(_info_semantic_scannet)" || return 1

  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PY}" scripts/validate_lang_field_traj.py \
    --scene "${SCENE}" \
    --result_dir "${rd}" \
    --traj_txt "${traj}" \
    --out_dir "${out}" \
    --device "${DEVICE}" \
    --class_info_file "${PROJ_DIR}/${CLASS_INFO}" \
    --info_semantic "${info_sem}" \
    --replica_train_traj "${PROJ_DIR}/${TRAIN_TRAJ}" \
    --codebook_size "${CODEBOOK_SIZE}" \
    --vq_layer_num "${VQ_LAYER_NUM}" >"${log_file}" 2>&1
  local rc=$?
  set -e

  if [[ "${rc}" -eq 0 && -f "${summary}" ]]; then
    log "OK ${label} gpu${gpu}"
    grep "Overall mIoU" "${summary}" 2>/dev/null || true
    return 0
  fi

  log "FAIL ${label} validate gpu${gpu} (rc=${rc})"
  tail -30 "${log_file}" 2>/dev/null || true
  return 1
}

_run_validate_pool() {
  local rel fail=0 qi=0
  local -a failed_jobs=()
  local -a SLOT_PIDS=("" "")
  local -a SLOT_RELS=("" "")
  local -a todo=("${RUN_RELS[@]}")

  log "Validate pool: ${#todo[@]} runs — 1 per GPU (cuda:${GPU0}, cuda:${GPU1})"
  for rel in "${todo[@]}"; do
    log "  queued ${rel}"
  done

  _reap_val_slots() {
    local i pid rc lbl gpu
    for i in 0 1; do
      pid="${SLOT_PIDS[$i]}"
      [[ -n "${pid}" ]] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        continue
      fi
      set +e
      wait "${pid}"
      rc=$?
      set -e
      lbl="$(_run_label "${SLOT_RELS[$i]}")"
      gpu="${GPUS[$i]}"
      if [[ "${rc}" -ne 0 ]]; then
        failed_jobs+=("${lbl}")
        fail=1
      fi
      SLOT_PIDS[$i]=""
      SLOT_RELS[$i]=""
    done
  }

  _start_val_slot() {
    local slot="$1"
    [[ -z "${SLOT_PIDS[$slot]}" ]] || return 1
    [[ "${qi}" -lt ${#todo[@]} ]] || return 1
    local r="${todo[$qi]}"
    qi=$((qi + 1))
    _validate_run "${r}" "${GPUS[$slot]}" &
    SLOT_PIDS[$slot]=$!
    SLOT_RELS[$slot]="${r}"
    log "START validate $(_run_label "${r}") → cuda:${GPUS[$slot]} only (pid=${SLOT_PIDS[$slot]})"
  }

  _any_val_running() {
    [[ -n "${SLOT_PIDS[0]}" || -n "${SLOT_PIDS[1]}" ]]
  }

  while [[ "${qi}" -lt ${#todo[@]} ]] || _any_val_running; do
    _reap_val_slots
    _start_val_slot 0 || true
    _start_val_slot 1 || true
    _any_val_running && sleep 3
  done

  if [[ "${#failed_jobs[@]}" -gt 0 ]]; then
    log "Validate failed: ${failed_jobs[*]}"
    return 1
  fi
  return 0
}

_count_gaussians() {
  local rd="$1"
  local ck
  for ck in "${rd}/splatam/final/params0.npz" "${rd}/splatam/final/params.npz"; do
    if [[ -f "${ck}" ]]; then
      "${PY}" - "${ck}" <<'PY'
import sys
import numpy as np
print(int(np.load(sys.argv[1])["means3D"].shape[0]))
PY
      return 0
    fi
  done
  echo "?"
}

log "=== ScanNet ${SCENE} lang-field (train + validate) ==="
log "  PHASE           : ${PHASE}"
log "  Runs            : ${RUN_RELS[*]}"
log "  GPU pool        : cuda:${GPU0} + cuda:${GPU1} (1 job per GPU, never shared)"
log "  NVS traj        : ${NVS_ROOT}/${SCENE}/traj.txt"
log "  Levels          : s, m, l (${NUM_ITERS} iters)"
log "  TRAIN_DOWNSCALE : ${TRAIN_DOWNSCALE}"
log "  RENDER_CHECKPOINT: ${RENDER_CHECKPOINT}"
log "  SKIP_DONE       : ${SKIP_DONE}"
log "  Log dir         : ${LOG_DIR}"
for rel in "${RUN_RELS[@]}"; do
  n="$(_count_gaussians "$(_run_abs "${rel}")")"
  log "  Gaussians ${rel}: ${n}"
done

FAIL=0

if [[ "${PHASE}" == "all" || "${PHASE}" == "train" ]]; then
  for rel in "${RUN_RELS[@]}"; do
    _check_prereqs "${rel}" "$(_run_abs "${rel}")" || FAIL=1
  done
  if [[ "${FAIL}" -ne 0 ]]; then
    log "Train prerequisite check failed."
    exit 1
  fi
  if ! _run_train_pool; then
    log "Training finished with errors."
    exit 1
  fi
  log "Training finished successfully."
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "validate" ]]; then
  if ! _check_validate_prereqs; then
    exit 1
  fi
  for rel in "${RUN_RELS[@]}"; do
    rd="$(_run_abs "${rel}")"
    if ! _all_levels_done "${rd}"; then
      log "ERROR ${rel}: lang_field s/m/l not ready for validate"
      FAIL=1
    fi
  done
  if [[ "${FAIL}" -ne 0 ]]; then
    exit 1
  fi
  if ! _run_validate_pool; then
    log "Validation finished with errors."
    exit 1
  fi
  log "Validation finished successfully."
fi

log "Done."
log "  Train  → results/ScanNet/${SCENE}/{ActiveOpenSem,ActiveOpenSem_replica_nbv}/${RUN_TAG}/lang_field_*"
log "  Valid  → .../lang_field_traj_eval/miou_summary.txt"
