#!/usr/bin/env bash
set -u

# Run ActiveOpenSem on MP3D scenes in order:
#   1) first scene from scripts/data/scan_id.txt
#   2) all remaining scenes
# Continue even if some scenes fail.
#
# Per-scene outputs:
#   logs/mp3d_activeopensem/<scene>.log
#   logs/mp3d_activeopensem/<scene>_vram.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENES_FILE="${ROOT_DIR}/scripts/data/scan_id.txt"
LOG_DIR="${ROOT_DIR}/logs/mp3d_activeopensem"
ENABLE_VIS="${ENABLE_VIS:-0}"
GPU_ID="${GPU_ID:-0}"
NUM_RUN="${NUM_RUN:-1}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SCENES_FILE}" ]]; then
  echo "[ERROR] Scenes file not found: ${SCENES_FILE}"
  exit 1
fi

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/' "${SCENES_FILE}")
if [[ "${#SCENES[@]}" -eq 0 ]]; then
  echo "[ERROR] No scenes found in ${SCENES_FILE}"
  exit 1
fi

run_scene() {
  local scene="$1"
  local scene_log="${LOG_DIR}/${scene}.log"
  local vram_log="${LOG_DIR}/${scene}_vram.csv"

  echo "===== START ${scene} $(date '+%F %T') =====" | tee -a "${scene_log}"

  (
    echo "timestamp,index,memory.used,memory.total,utilization.gpu"
    while true; do
      nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
      sleep 5
    done
  ) > "${vram_log}" &
  local vram_pid=$!

  (
    cd "${ROOT_DIR}" || exit 1
    bash scripts/activesgm/run_mp3d.sh "${scene}" "${NUM_RUN}" ActiveOpenSem "${ENABLE_VIS}" "${GPU_ID}"
  ) >> "${scene_log}" 2>&1
  local code=$?

  kill "${vram_pid}" 2>/dev/null || true
  wait "${vram_pid}" 2>/dev/null || true

  echo "===== END ${scene} exit=${code} $(date '+%F %T') =====" | tee -a "${scene_log}"
  return "${code}"
}

first_scene="${SCENES[0]}"
run_scene "${first_scene}" || true

for scene in "${SCENES[@]:1}"; do
  run_scene "${scene}" || true
done

echo "[DONE] Finished ActiveOpenSem batch over ${#SCENES[@]} scenes."
