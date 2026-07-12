#!/usr/bin/env bash
##################################################
# ScanNet scene0050_02 — validate language fields (NVS traj mIoU).
#
# Runs (parallel, 1 job = 1 GPU):
#   cuda:GPU0 → ActiveOpenSem/run_0
#   cuda:GPU1 → ActiveOpenSem_replica_nbv/run_0
#
# Required in each run_dir:
#   lang_field_sk64_l1/lang_field.pt
#   lang_field_mk64_l1/lang_field.pt
#   lang_field_lk64_l1/lang_field.pt
#   splatam/final/params*.npz
#
# External data:
#   data/scannet_sim_nvs/scene0050_02/traj.txt
#   data/scannet_sim_nvs/scene0050_02/results_habitat/semantic/semantic*.npy
#   data/ScanNet/scene0050_02/traj.txt
#   configs/ScanNet/nyu40_info_semantic.json  (NYU40 GT ids, NOT class_info_file.json)
#
# Usage (from repo root):
#   bash scripts/aov-gs/validate_scannet0050_lang_field.sh
#
# Background:
#   mkdir -p logs/scannet0050_lang_field_validate
#   nohup bash scripts/aov-gs/validate_scannet0050_lang_field.sh \
#     > logs/scannet0050_lang_field_validate/orchestrator.log 2>&1 &
#
# Env:
#   SCENE=scene0050_02  RUN_TAG=run_0
#   GPU0=0  GPU1=1
#   SKIP_DONE=1  FORCE=0
#   CODEBOOK_SIZE=64  VQ_LAYER_NUM=1
#   REGENERATE_SEMANTICS=auto|1|0  (auto: regen if GT masks are all-zero)
#   NVS_ROOT=data/scannet_sim_nvs
##################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

SCENE="${SCENE:-scene0050_02}"
RUN_TAG="${RUN_TAG:-run_0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DEVICE="cuda:0"

NVS_ROOT="${NVS_ROOT:-data/scannet_sim_nvs}"
CLASS_INFO="${CLASS_INFO:-configs/ScanNet/class_info_file.json}"
TRAIN_TRAJ="${TRAIN_TRAJ:-data/ScanNet/${SCENE}/traj.txt}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-64}"
VQ_LAYER_NUM="${VQ_LAYER_NUM:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
FORCE="${FORCE:-0}"
REGENERATE_SEMANTICS="${REGENERATE_SEMANTICS:-auto}"

LOG_DIR="${LOG_DIR:-${PROJ_DIR}/logs/scannet0050_lang_field_validate}"
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
export PROJ_DIR

log() { echo "[$(date -Is)] $*" >&2; }

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

_all_levels_done() {
  local rd="$1"
  local lvl
  for lvl in s m l; do
    _level_done "${rd}" "${lvl}" || return 1
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
    log "ERROR: missing ${missing[*]}"
    return 1
  fi
  return 0
}

_nvs_sem_dir() {
  echo "${PROJ_DIR}/${NVS_ROOT}/${SCENE}/results_habitat/semantic"
}

_semantics_have_labels() {
  local sem_dir
  sem_dir="$(_nvs_sem_dir)"
  "${PY}" - "${sem_dir}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["PROJ_DIR"])
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts"))
import lang_field_eval_utils as lfu

sem_dir = Path(sys.argv[1])
sys.exit(0 if lfu.nvs_semantics_have_labels(sem_dir) else 1)
PY
}

_regenerate_nvs_semantics() {
  local log_file="${LOG_DIR}/regenerate_${SCENE}_semantics.log"
  log "Regenerating NVS GT semantics (scene_id=<scene>_semantic.ply) -> ${log_file}"
  log "  (hires.glb + semantic.ply stage often yields all-zero masks; this fixes it)"

  if [[ -f "${PROJ_DIR}/docker/ensure_habitat_egl.sh" ]]; then
    bash "${PROJ_DIR}/docker/ensure_habitat_egl.sh" >>"${log_file}" 2>&1 || {
      log "FAIL: Habitat EGL bootstrap — see ${log_file}"
      return 1
    }
  fi

  set +e
  (
    if [[ -f "${PROJ_DIR}/docker/nvidia_habitat_env.sh" ]]; then
      # shellcheck source=/dev/null
      source "${PROJ_DIR}/docker/nvidia_habitat_env.sh"
    fi
    export CUDA_VISIBLE_DEVICES="${GPU0}"
    export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
    "${PY}" scripts/data/generate_scannet_semantics_from_poses.py \
      --scenes "${SCENE}" \
      --overwrite
  ) >>"${log_file}" 2>&1
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    log "FAIL semantic regeneration (rc=${rc})"
    tail -40 "${log_file}" 2>/dev/null || true
    return "${rc}"
  fi
  if ! _semantics_have_labels; then
    log "ERROR: semantics still all-zero after regeneration"
    return 1
  fi
  log "OK: NVS semantics regenerated"
  return 0
}

_ensure_nvs_semantics() {
  local sem_dir
  sem_dir="$(_nvs_sem_dir)"
  if _semantics_have_labels; then
    log "NVS semantics OK: ${sem_dir}"
    return 0
  fi
  log "WARN: NVS semantic masks are missing or all-zero under ${sem_dir}"
  case "${REGENERATE_SEMANTICS}" in
    0|false|no)
      log "ERROR: set REGENERATE_SEMANTICS=auto|1 or run:"
      log "  python scripts/data/generate_scannet_semantics_from_poses.py --scenes ${SCENE} --overwrite"
      return 1
      ;;
  esac
  _regenerate_nvs_semantics
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
  info_sem="${INFO_SEMANTIC_PATH}"

  log "VALIDATE ${label} on cuda:${gpu} -> ${log_file}"

  if ! _all_levels_done "${rd}"; then
    log "ERROR ${label}: need lang_field_{s,m,l}k${CODEBOOK_SIZE}_l${VQ_LAYER_NUM}/lang_field.pt"
    return 1
  fi

  if [[ "${FORCE}" != "1" && "${SKIP_DONE}" == "1" && -f "${summary}" ]]; then
    log "SKIP ${label}: ${summary} exists"
    grep "Overall mIoU" "${summary}" 2>/dev/null || true
    return 0
  fi

  if [[ "${FORCE}" == "1" ]]; then
    log "FORCE=1 — removing ${out}"
    rm -rf "${out}"
  fi

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
  tail -40 "${log_file}" 2>/dev/null || true
  return 1
}

# Fixed GPU assignment: one method per card.
RUN_GPU0="results/ScanNet/${SCENE}/ActiveOpenSem/${RUN_TAG}"
RUN_GPU1="results/ScanNet/${SCENE}/ActiveOpenSem_replica_nbv/${RUN_TAG}"

log "=== ScanNet ${SCENE} lang-field validation ==="
log "  cuda:${GPU0} -> ${RUN_GPU0}"
log "  cuda:${GPU1} -> ${RUN_GPU1}"
log "  NVS traj: ${NVS_ROOT}/${SCENE}/traj.txt"
log "  SKIP_DONE=${SKIP_DONE}  FORCE=${FORCE}  REGENERATE_SEMANTICS=${REGENERATE_SEMANTICS}"
log "  Log dir: ${LOG_DIR}"

if ! _check_validate_prereqs; then
  exit 1
fi

if ! _ensure_nvs_semantics; then
  exit 1
fi

INFO_SEMANTIC_PATH="$(_info_semantic_scannet)" || exit 1
log "  info_semantic: ${INFO_SEMANTIC_PATH}"

for rel in "${RUN_GPU0}" "${RUN_GPU1}"; do
  rd="$(_run_abs "${rel}")"
  if [[ ! -d "${rd}" ]]; then
    log "ERROR: run dir missing: ${rd}"
    exit 1
  fi
  if ! _all_levels_done "${rd}"; then
    log "ERROR: lang fields not ready in ${rel}"
    exit 1
  fi
done

FAIL=0
PID0="" PID1=""

_validate_run "${RUN_GPU0}" "${GPU0}" &
PID0=$!
_validate_run "${RUN_GPU1}" "${GPU1}" &
PID1=$!

set +e
wait "${PID0}"
RC0=$?
wait "${PID1}"
RC1=$?
set -e

[[ "${RC0}" -eq 0 ]] || FAIL=1
[[ "${RC1}" -eq 0 ]] || FAIL=1

log "--- Results ---"
for rel in "${RUN_GPU0}" "${RUN_GPU1}"; do
  summary="$(_run_abs "${rel}")/lang_field_traj_eval/miou_summary.txt"
  if [[ -f "${summary}" ]]; then
    log "  ${rel}:"
    grep "Overall mIoU" "${summary}" 2>/dev/null | sed 's/^/    /' || true
  else
    log "  ${rel}: (no miou_summary.txt)"
  fi
done

if [[ "${FAIL}" -eq 0 ]]; then
  log "Done."
  exit 0
fi

log "Finished with errors."
exit 1
