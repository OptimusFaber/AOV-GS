#!/usr/bin/env bash
##################################################
# MP3D lang-field pipeline (train s/m/l + traj validation).
#
# Global training pool: up to PARALLEL_TRAIN level jobs on cuda:0.
# When one level finishes, the next queued job starts immediately.
# After ALL runs have s/m/l, validate each run (up to PARALLEL_VALIDATE).
#
# Default runs (2 scenes × ActiveGeom + ActiveOpenSem, run_0 each):
#   GdvgFV5R1Z5  — ActiveGeom, ActiveOpenSem
#   gZ6f7yhEvPG  — ActiveGeom, ActiveOpenSem
#
# Select a subset via SCENE / SCENES (see Env below).
#
# Prerequisites per run dir:
#   splatam/final/params*.npz, keyframe_poses.json, language_features/
#
# Validation additionally needs NVS bundle:
#   data/mp3d_sim_nvs_v2/<scene>/traj.txt
#   data/mp3d_sim_nvs_v2/<scene>/results_habitat/semantic/semantic_map_*.npy
#
# Usage (from repo root):
#   bash scripts/aov-gs/clore/run_mp3d_lang_field_pipeline.sh
#
# One scene only (ActiveGeom + ActiveOpenSem, run_0):
#   SCENE=GdvgFV5R1Z5 bash scripts/aov-gs/clore/run_mp3d_lang_field_pipeline.sh
#
# Several scenes, sequential training (12 GB GPU):
#   SCENES="GdvgFV5R1Z5 gZ6f7yhEvPG" PARALLEL_TRAIN=1 \
#     bash scripts/aov-gs/clore/run_mp3d_lang_field_pipeline.sh
#
# Custom runs (full override):
#   RUNS="results/MP3D/GdvgFV5R1Z5/ActiveGeom/run_0 ..." \
#     bash scripts/aov-gs/clore/run_mp3d_lang_field_pipeline.sh
#
# Background:
#   mkdir -p logs/lang_field_mp3d
#   nohup bash scripts/aov-gs/clore/run_mp3d_lang_field_pipeline.sh \
#     > logs/lang_field_mp3d/orchestrator.log 2>&1 &
#   echo $! > logs/lang_field_mp3d/orchestrator.pid
#
# Env:
#   SCENE             — one MP3D scene id (e.g. GdvgFV5R1Z5)
#   SCENES            — space-separated scene ids (overrides SCENE)
#   METHODS           — default "ActiveGeom ActiveOpenSem"
#   RUN_TAG           — default run_0
#   RUNS              — explicit run dirs (overrides SCENE/SCENES)
#   NVS_ROOT          — default data/mp3d_sim_nvs_v2
#   PARALLEL_TRAIN=2  — one job per GPU (cuda:GPU0 + cuda:GPU1); default 1 = single GPU
#   PARALLEL_VALIDATE=2
#   GPU=0             — single-GPU mode (legacy)
#   GPU0=0 GPU1=1     — physical ids for 2-GPU pool
#   SKIP_DONE=1
#   FORCE_RETRAIN=0
#   SKIP_VALIDATE=0
#   SKIP_TRAIN=0                 validate-only when 1
#   MIN_RECOVER_ITER_FRAC=90     best.pt accepted only if best_iteration >= 90% of NUM_ITERS
#   LOG_DIR=logs/lang_field_mp3d
#
# On kill/OOM: failed job is logged; best.pt → lang_field.pt only if >= MIN_RECOVER_ITER_FRAC,
# then the queue continues with the next level/run (no global abort).
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${PROJ_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJ_DIR}"

GPU="${GPU:-0}"
GPU0="${GPU0:-${GPU}}"
GPU1="${GPU1:-1}"
GPUS=("${GPU0}" "${GPU1}")
DEVICE="cuda:0"
PARALLEL_TRAIN="${PARALLEL_TRAIN:-1}"
PARALLEL_VALIDATE="${PARALLEL_VALIDATE:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
NUM_ITERS="${NUM_ITERS:-12000}"
# Recover best.pt → lang_field.pt only if training reached at least this fraction (avoid 500/12000 false "done").
MIN_RECOVER_ITER_FRAC="${MIN_RECOVER_ITER_FRAC:-90}"
TRAIN_DOWNSCALE="${TRAIN_DOWNSCALE:-0.5}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-64}"
VQ_LAYER_NUM="${VQ_LAYER_NUM:-1}"
TOPK="${TOPK:-4}"
RENDER_CHECKPOINT="${RENDER_CHECKPOINT:-auto}"
LOG_EVERY="${LOG_EVERY:-500}"
NVS_ROOT="${NVS_ROOT:-data/mp3d_sim_nvs_v2}"
CLASS_INFO="${CLASS_INFO:-configs/MP3D/class_info_file.json}"
RUN_TAG="${RUN_TAG:-run_0}"
METHODS="${METHODS:-ActiveGeom ActiveOpenSem}"

DEFAULT_SCENES=(GdvgFV5R1Z5 gZ6f7yhEvPG)

_build_runs_for_scenes() {
  local scene method
  for scene in "$@"; do
    for method in ${METHODS}; do
      echo "results/MP3D/${scene}/${method}/${RUN_TAG}"
    done
  done
}

_resolve_run_list() {
  if [[ -n "${RUNS:-}" ]]; then
    # shellcheck disable=SC2206
    RUN_LIST=(${RUNS})
    return
  fi

  local -a scenes=()
  if [[ -n "${SCENES:-}" ]]; then
    # shellcheck disable=SC2206
    scenes=(${SCENES})
  elif [[ -n "${SCENE:-}" ]]; then
    scenes=("${SCENE}")
  else
    scenes=("${DEFAULT_SCENES[@]}")
  fi

  RUN_LIST=()
  local rel
  while IFS= read -r rel; do
    [[ -n "${rel}" ]] && RUN_LIST+=("${rel}")
  done < <(_build_runs_for_scenes "${scenes[@]}")
}

_resolve_run_list

if [[ "${#RUN_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: empty RUN_LIST — set SCENE, SCENES, or RUNS" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/lang_field_mp3d}"
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

_scene_from_run() {
  local rel="$1"
  local abs scene method
  abs="$(_run_abs "${rel}")"
  scene="$(basename "$(dirname "$(dirname "${abs}")")")"
  method="$(basename "$(dirname "${abs}")")"
  echo "${scene}|${method}"
}

_run_label() {
  local rel="$1"
  local abs scene method run_name
  abs="$(_run_abs "${rel}")"
  scene="$(basename "$(dirname "$(dirname "${abs}")")")"
  method="$(basename "$(dirname "${abs}")")"
  run_name="$(basename "${abs}")"
  echo "${scene}_${method}_${run_name}"
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
  local rel_from_rd label_default
  rel_from_rd="$(echo "${rd}" | sed "s|^${PROJ_DIR}/||")"
  label_default="$(_run_label "${rel_from_rd}")"
  local label="${3:-${label_default}}"
  local dir="${rd}/lang_field_${level}k${CODEBOOK_SIZE}_l${VQ_LAYER_NUM}"
  local best="${dir}/best.pt"
  local final="${dir}/lang_field.pt"
  local best_it min_it

  if [[ -f "${final}" ]]; then
    return 0
  fi
  if [[ ! -f "${best}" ]]; then
    return 1
  fi

  best_it="$(_best_iteration "${best}")"
  min_it="$(_min_recover_iters)"
  if [[ "${best_it}" -lt "${min_it}" ]]; then
    log "SKIP RECOVER ${label}/${level}: best.pt @ iter ${best_it} < ${min_it} (${MIN_RECOVER_ITER_FRAC}% of ${NUM_ITERS}) — needs retrain"
    return 1
  fi

  log "RECOVER ${label}/${level}: best.pt (iter ${best_it}) → lang_field.pt"
  cp -a "${best}" "${final}"
  return 0
}

_level_ready() {
  local rd="$1"
  local level="$2"
  _level_done "${rd}" "${level}"
}

_all_levels_done() {
  local rd="$1"
  local lvl
  for lvl in s m l; do
    _level_ready "${rd}" "${lvl}" || return 1
  done
  return 0
}

_info_semantic_mp3d() {
  local out="${LOG_DIR}/mp3d_info_semantic.json"
  if [[ -f "${out}" ]]; then
    echo "${out}"
    return 0
  fi
  if [[ ! -f "${PROJ_DIR}/${CLASS_INFO}" ]]; then
    log "ERROR: missing ${CLASS_INFO} for MP3D validation"
    return 1
  fi
  "${PY}" - "${PROJ_DIR}/${CLASS_INFO}" "${out}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = json.loads(src.read_text(encoding="utf-8"))
classes = []
for key in sorted(data.keys(), key=lambda k: int(k)):
    entry = data[key]
    classes.append({"id": int(key), "name": str(entry["name"]).strip()})
payload = {"classes": classes}
dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(dst)
PY
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

_check_validate_prereqs() {
  local scene="$1"
  local traj="${PROJ_DIR}/${NVS_ROOT}/${scene}/traj.txt"
  local sem_dir="${PROJ_DIR}/${NVS_ROOT}/${scene}/results_habitat/semantic"
  local missing=()

  [[ -f "${traj}" ]] || missing+=("${NVS_ROOT}/${scene}/traj.txt")
  if [[ ! -d "${sem_dir}" ]] || [[ -z "$(ls -A "${sem_dir}"/semantic_map_*.npy 2>/dev/null)" ]]; then
    missing+=("${NVS_ROOT}/${scene}/results_habitat/semantic/semantic_map_*.npy")
  fi

  if [[ "${#missing[@]}" -gt 0 ]]; then
    log "ERROR validate ${scene}: missing ${missing[*]}"
    return 1
  fi
  return 0
}

_train_level() {
  local rel="$1"
  local level="$2"
  local gpu="${3:-${GPU0}}"
  local label log_file
  label="$(_run_label "${rel}")"
  log_file="${LOG_DIR}/train_${label}_${level}_gpu${gpu}.log"

  log "TRAIN ${label}/${level} on cuda:${gpu} → ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  bash "${PROJ_DIR}/scripts/aov-gs/03_train_gaussian_lang_field.sh" \
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

  for rel in "${RUN_LIST[@]}"; do
    rd="$(_run_abs "${rel}")"
    label="$(_run_label "${rel}")"

    if [[ "${FORCE_RETRAIN}" == "1" ]]; then
      log "FORCE_RETRAIN=1 — removing lang_field_* for ${label}"
      rm -rf "${rd}"/lang_field_*
    fi

    for lvl in s m l; do
      if _level_ready "${rd}" "${lvl}"; then
        if [[ "${SKIP_DONE}" == "1" ]]; then
          log "SKIP ${label}/${lvl}: checkpoint exists"
          continue
        fi
      fi
      queue+=("${rel}|${lvl}")
    done
  done

  if [[ "${#queue[@]}" -eq 0 ]]; then
    log "TRAIN: all levels done for all runs"
    return 0
  fi

  log "TRAIN pool: ${#queue[@]} jobs, parallel=${PARALLEL_TRAIN} (GPUs: ${GPUS[*]})"
  for item in "${queue[@]}"; do
    log "  queued ${item%|*}/$(echo "${item}" | cut -d'|' -f2)"
  done

  if [[ "${PARALLEL_TRAIN}" -le 1 ]]; then
    TRAIN_QUEUE=("${queue[@]}")
    _run_train_pool_slots 1 "${GPU0}"
    return $?
  fi

  TRAIN_QUEUE=("${queue[@]}")
  _run_train_pool_slots 2 "${GPUS[@]}"
  return $?
}

_run_train_pool_slots() {
  local num_slots="$1"
  shift
  local -a slot_gpus=("$@")
  local qi=0 fail=0
  local -a failed_jobs=()
  local -a queue=()

  # Re-build queue from outer scope — pass via global QUEUE_ITEMS set before call
  # shellcheck disable=SC2206
  queue=("${TRAIN_QUEUE[@]}")

  declare -a SLOT_PIDS=()
  declare -a SLOT_RELS=()
  declare -a SLOT_LEVELS=()
  local s
  for ((s = 0; s < num_slots; s++)); do
    SLOT_PIDS+=("")
    SLOT_RELS+=("")
    SLOT_LEVELS+=("")
  done

  _reap_slot() {
    local i pid rc rd label
    for ((i = 0; i < num_slots; i++)); do
      pid="${SLOT_PIDS[$i]}"
      [[ -n "${pid}" ]] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        continue
      fi
      set +e
      wait "${pid}"
      rc=$?
      set -e
      label="$(_run_label "${SLOT_RELS[$i]}")"
      rd="$(_run_abs "${SLOT_RELS[$i]}")"
      local lvl="${SLOT_LEVELS[$i]}"
      local gpu="${slot_gpus[$i]}"
      if [[ "${rc}" -ne 0 ]]; then
        log "FAIL ${label}/${lvl} gpu${gpu} (rc=${rc}) — tail ${LOG_DIR}/train_${label}_${lvl}_gpu${gpu}.log"
        tail -30 "${LOG_DIR}/train_${label}_${lvl}_gpu${gpu}.log" 2>/dev/null || true
        if [[ "${rc}" -eq 137 || "${rc}" -eq 143 ]]; then
          log "  hint: rc=${rc} often means OOM kill (137) or SIGTERM (143)"
        fi
        if _recover_level_from_best "${rd}" "${lvl}" "${label}"; then
          log "RECOVERED ${label}/${lvl} from best.pt after abort"
        else
          failed_jobs+=("${label}/${lvl}")
          fail=1
        fi
      else
        if ! _level_ready "${rd}" "${lvl}"; then
          log "WARN ${label}/${lvl} exited 0 but lang_field.pt missing — try best.pt"
          _recover_level_from_best "${rd}" "${lvl}" "${label}" \
            || { failed_jobs+=("${label}/${lvl}"); fail=1; }
        else
          log "DONE ${label}/${lvl} gpu${gpu}"
        fi
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
    local rel="${item%|*}"
    local lvl="${item#*|}"
    qi=$((qi + 1))
    _train_level "${rel}" "${lvl}" "${slot_gpus[$slot]}" &
    SLOT_PIDS[$slot]=$!
    SLOT_RELS[$slot]="${rel}"
    SLOT_LEVELS[$slot]="${lvl}"
    log "START $(_run_label "${rel}")/${lvl} gpu${slot_gpus[$slot]} pid=${SLOT_PIDS[$slot]}"
    return 0
  }

  _any_slot_running() {
    local _s
    for ((_s = 0; _s < num_slots; _s++)); do
      [[ -n "${SLOT_PIDS[$_s]}" ]] && return 0
    done
    return 1
  }

  while [[ "${qi}" -lt ${#queue[@]} ]] || _any_slot_running; do
    _reap_slot
    local slot
    for ((slot = 0; slot < num_slots; slot++)); do
      _start_slot "${slot}" || true
    done
    _any_slot_running && sleep 3
  done

  if [[ "${#failed_jobs[@]}" -gt 0 ]]; then
    log "TRAIN failed jobs (${#failed_jobs[@]}): ${failed_jobs[*]}"
  fi
  return "${fail}"
}

_validate_run() {
  local rel="$1"
  local gpu="${2:-${GPU0}}"
  local rd label parsed scene method traj out summary log_file info_sem
  rd="$(_run_abs "${rel}")"
  label="$(_run_label "${rel}")"
  parsed="$(_scene_from_run "${rel}")"
  scene="${parsed%%|*}"
  method="${parsed#*|}"
  traj="${PROJ_DIR}/${NVS_ROOT}/${scene}/traj.txt"
  out="${rd}/lang_field_traj_eval"
  summary="${out}/miou_summary.txt"
  log_file="${LOG_DIR}/validate_${label}_gpu${gpu}.log"

  log "VALIDATE ${label} (${scene}/${method}) on cuda:${gpu} → ${log_file}"

  if ! _check_validate_prereqs "${scene}"; then
    return 1
  fi

  if ! _all_levels_done "${rd}"; then
    log "ERROR ${label}: not all lang_field levels ready"
    return 1
  fi

  if [[ "${SKIP_DONE}" == "1" && -f "${summary}" ]]; then
    log "SKIP ${label}: ${summary} exists"
    grep "Overall mIoU" "${summary}" 2>/dev/null || true
    return 0
  fi

  info_sem="$(_info_semantic_mp3d)" || return 1

  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PY}" scripts/validate_lang_field_traj.py \
    --scene "${scene}" \
    --result_dir "${rd}" \
    --traj_txt "${traj}" \
    --out_dir "${out}" \
    --device "${DEVICE}" \
    --class_info_file "${PROJ_DIR}/${CLASS_INFO}" \
    --info_semantic "${info_sem}" >"${log_file}" 2>&1
  local rc=$?
  set -e

  if [[ "${rc}" -eq 0 && -f "${summary}" ]]; then
    log "OK ${label} gpu${gpu}"
    grep "Overall mIoU" "${summary}" 2>/dev/null || true
    return 0
  fi

  log "FAIL ${label} validate gpu${gpu} (rc=${rc})"
  tail -25 "${log_file}" 2>/dev/null || true
  return 1
}

_run_validate_pool_slots() {
  local num_slots="$1"
  shift
  local -a slot_gpus=("$@")
  local -a todo=("${VALIDATE_TODO[@]}")
  local qi=0 fail=0
  local -a failed_jobs=()

  declare -a SLOT_PIDS=()
  declare -a SLOT_RELS=()
  local s
  for ((s = 0; s < num_slots; s++)); do
    SLOT_PIDS+=("")
    SLOT_RELS+=("")
  done

  _reap_slot() {
    local i pid rc label gpu
    for ((i = 0; i < num_slots; i++)); do
      pid="${SLOT_PIDS[$i]}"
      [[ -n "${pid}" ]] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        continue
      fi
      set +e
      wait "${pid}"
      rc=$?
      set -e
      label="$(_run_label "${SLOT_RELS[$i]}")"
      gpu="${slot_gpus[$i]}"
      if [[ "${rc}" -ne 0 ]]; then
        log "FAIL validate ${label} gpu${gpu} (rc=${rc})"
        failed_jobs+=("${label}")
        fail=1
      else
        log "DONE validate ${label} gpu${gpu}"
      fi
      SLOT_PIDS[$i]=""
      SLOT_RELS[$i]=""
    done
  }

  _start_slot() {
    local slot="$1"
    [[ -z "${SLOT_PIDS[$slot]}" ]] || return 1
    [[ "${qi}" -lt ${#todo[@]} ]] || return 1

    local rel="${todo[$qi]}"
    qi=$((qi + 1))
    _validate_run "${rel}" "${slot_gpus[$slot]}" &
    SLOT_PIDS[$slot]=$!
    SLOT_RELS[$slot]="${rel}"
    log "START validate $(_run_label "${rel}") gpu${slot_gpus[$slot]} pid=${SLOT_PIDS[$slot]}"
    return 0
  }

  _any_slot_running() {
    local _s
    for ((_s = 0; _s < num_slots; _s++)); do
      [[ -n "${SLOT_PIDS[$_s]}" ]] && return 0
    done
    return 1
  }

  while [[ "${qi}" -lt ${#todo[@]} ]] || _any_slot_running; do
    _reap_slot
    local slot
    for ((slot = 0; slot < num_slots; slot++)); do
      _start_slot "${slot}" || true
    done
    _any_slot_running && sleep 3
  done

  if [[ "${#failed_jobs[@]}" -gt 0 ]]; then
    log "VALIDATE failed runs (${#failed_jobs[@]}): ${failed_jobs[*]}"
  fi
  return "${fail}"
}

_run_validate_parallel() {
  local rel
  VALIDATE_TODO=()
  for rel in "${RUN_LIST[@]}"; do
    VALIDATE_TODO+=("${rel}")
  done

  if [[ "${#VALIDATE_TODO[@]}" -eq 0 ]]; then
    return 0
  fi

  log "VALIDATE pool: ${#VALIDATE_TODO[@]} runs, parallel=${PARALLEL_VALIDATE} (GPUs: ${GPUS[*]})"

  if [[ "${PARALLEL_VALIDATE}" -le 1 ]]; then
    _run_validate_pool_slots 1 "${GPU0}"
    return $?
  fi

  _run_validate_pool_slots 2 "${GPUS[@]}"
}

log "=== MP3D lang-field pipeline ==="
if [[ -n "${SCENES:-}" ]]; then
  log "  Scenes filter   : ${SCENES}"
elif [[ -n "${SCENE:-}" ]]; then
  log "  Scene           : ${SCENE}"
else
  log "  Scenes          : all default (${DEFAULT_SCENES[*]})"
fi
log "  Methods         : ${METHODS}"
log "  Run tag         : ${RUN_TAG}"
log "  Runs            : ${RUN_LIST[*]}"
log "  NVS root        : ${NVS_ROOT}"
log "  GPU pool        : cuda:${GPU0}, cuda:${GPU1}"
log "  Parallel train  : ${PARALLEL_TRAIN}"
log "  Parallel valid  : ${PARALLEL_VALIDATE}"
log "  NUM_ITERS       : ${NUM_ITERS}"
log "  TRAIN_DOWNSCALE : ${TRAIN_DOWNSCALE}"
log "  SKIP_TRAIN      : ${SKIP_TRAIN}"
log "  MIN_RECOVER     : ${MIN_RECOVER_ITER_FRAC}% of ${NUM_ITERS} (= $(_min_recover_iters) iters)"
log "  Log dir         : ${LOG_DIR}"

FAIL=0
for rel in "${RUN_LIST[@]}"; do
  _check_prereqs "${rel}" "$(_run_abs "${rel}")" || FAIL=1
done

if [[ "${FAIL}" -ne 0 ]]; then
  log "Prerequisite check failed."
  exit 1
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  _run_train_pool || FAIL=1
else
  log "SKIP_TRAIN=1 — training phase skipped"
fi

if [[ "${FAIL}" -eq 0 && "${SKIP_VALIDATE}" != "1" ]]; then
  _run_validate_parallel || FAIL=1
elif [[ "${SKIP_VALIDATE}" == "1" ]]; then
  log "SKIP_VALIDATE=1 — validation phase skipped"
fi

if [[ "${FAIL}" -eq 0 ]]; then
  log "Pipeline finished successfully."
else
  log "Pipeline finished with errors."
  exit 1
fi
