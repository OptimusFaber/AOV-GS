#!/bin/bash
##################################################
# ActiveOpenSem / ActiveGeom on MP3D (Habitat).
#
# Results (auto run_N):
#   Passive          -> results/MP3D/{SCENE}/Passive/run_N/
#   ActiveGeom       -> results/MP3D/{SCENE}/ActiveGeom/run_N/
#   ActiveOpenSem    -> results/MP3D/{SCENE}/ActiveOpenSem/run_N/
#
# Usage:
#   bash scripts/aov-gs/01_slam_exploration_mp3d.sh [SCENE] [EXP] [SEED] [ENABLE_VIS] [DEBUG]
#
#   SCENE: GdvgFV5R1Z5 | gZ6f7yhEvPG | HxpKQynjfin | pLe4wQe7qrG | YmJkqBEsHnH | all
#   EXP:   ActiveOpenSem | ActiveOpenSemGeom | ActiveOpenSemPassive | ActiveOpenSem_base
#   GPU:   export GPU=0  (default; single-GPU setup)
#
# Prerequisites:
#   bash scripts/data/prepare_mp3d_opensem_configs.py   # once, generates configs
#   data/MP3D/ + data/mp3d_sim_nvs_v2/{SCENE}/          # dataset + NVS traj
#
# Examples:
#   bash scripts/aov-gs/01_slam_exploration_mp3d.sh GdvgFV5R1Z5 ActiveOpenSem
#   bash scripts/aov-gs/01_slam_exploration_mp3d.sh all ActiveOpenSemGeom 0 0
##################################################

set -e

PROJ_DIR_EARLY="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f /.dockerenv ]]; then
    bash "${PROJ_DIR_EARLY}/docker/ensure_habitat_ptex.sh" 2>/dev/null || true
fi

SCENE=${1:-GdvgFV5R1Z5}
EXP=${2:-ActiveOpenSem}
SEED=${3:-0}
ENABLE_VIS=${4:-0}
DEBUG=${5:-0}

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_DIR"

ALL_SCENES=(GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH)

run_one() {
    local scene="$1"
    local cfg="${PROJ_DIR}/configs/MP3D/${scene}/${EXP}.py"

    if [[ ! -f "$cfg" ]]; then
        echo "Missing $cfg — run: python scripts/data/prepare_mp3d_opensem_configs.py" >&2
        exit 1
    fi

    if [[ ! -f "${PROJ_DIR}/data/MP3D/v1/tasks/mp3d/${scene}/${scene}.glb" ]]; then
        echo "Missing Habitat mesh for ${scene} under data/MP3D/v1/tasks/mp3d/" >&2
        exit 1
    fi

    echo "=============================================="
    echo "  Dataset    : MP3D"
    echo "  Scene      : $scene"
    echo "  Config     : configs/MP3D/${scene}/${EXP}.py"
    echo "  Seed       : $SEED"
    echo "  GPU        : $CUDA_VISIBLE_DEVICES"
    echo "  Visualize  : $ENABLE_VIS"
    echo "  Debug      : $DEBUG"
    echo "  CorrCLIP   : OFF (--corrclip 0)"
    echo "=============================================="

    local debug_flag=""
    [[ "$DEBUG" == "1" ]] && debug_flag="--debug"

    python src/main/activesgm.py \
        --cfg        "configs/MP3D/${scene}/${EXP}.py" \
        --seed       "$SEED" \
        --enable_vis "$ENABLE_VIS" \
        --corrclip   0 \
        $debug_flag

    echo ""
    echo "=== SLAM finished: MP3D/${scene}/${EXP} ==="
    echo "    Check latest run_* under results/MP3D/${scene}/"
}

if [[ "$SCENE" == "all" ]]; then
    for s in "${ALL_SCENES[@]}"; do
        run_one "$s"
    done
else
    run_one "$SCENE"
fi
