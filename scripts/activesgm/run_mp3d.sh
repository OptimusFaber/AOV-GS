#!/bin/bash
##################################################
### This script is to run the full NARUTO system 
### (active planning and active ray sampling) 
###  on the Replica dataset.
##################################################
set -euo pipefail

# Input arguments
scene=${1:-GdvgFV5R1Z5}
num_run=${2:-1}
# ActiveOpenSemGeom = SplaTAM + active_gs (geometry). ActiveOpenSem = hybrid v3 + SAM.
EXP=${3:-ActiveOpenSemGeom}
ENABLE_VIS=${4:-0}
GPU_ID=${5:-0}

# Result folder name (may differ from config filename).
case "$EXP" in
    ActiveOpenSemGeom) RESULT_EXP=ActiveGeom ;;
    ActiveOpenSemPassive) RESULT_EXP=Passive ;;
    *) RESULT_EXP="$EXP" ;;
esac

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PROJ_DIR=${PWD}
DATASET=MP3D
RESULT_DIR=${PROJ_DIR}/results/

##################################################
### Random Seed
###     also used to initialize agent pose 
###     from indexing the pose in Replica SLAM 
###     trajectory.
##################################################
seeds=(0 500 1000 1500 1999)
seeds=("${seeds[@]:0:$num_run}")

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
# scenes=(room0 room1 room2 office0 office1 office2 office3 office4)
scenes=( GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH )
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

##################################################
### Main
###     Run for selected scenes for N trials
##################################################
for scene in $selected_scenes
do
    for i in "${!seeds[@]}"; do
        seed=${seeds[$i]}

        ### create result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${RESULT_EXP}/run_${i}
        mkdir -p ${result_dir}

        ### run experiment ###
        CFG=configs/${DATASET}/${scene}/${EXP}.py
        if [[ ! -f "$CFG" ]]; then
            echo "[ERROR] Missing $CFG — run: python scripts/data/prepare_mp3d_opensem_configs.py" >&2
            exit 1
        fi
        python src/main/activesgm.py --cfg ${CFG} --seed ${seed} --result_dir ${result_dir} --enable_vis ${ENABLE_VIS} --corrclip 0

        ### 3D Reconstruction evaluation ###
        DASHSCENE=${scene: 0: 0-1}_${scene: 0-1}
        GT_MESH=$PROJ_DIR/data/MP3D/v1/scans/${scene}/mesh.obj
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${RESULT_EXP}/run_${i}

        eval_ckpt_if_exists() {
            local ckpt_path="$1"
            local out_dir="$2"
            if [[ ! -f "$ckpt_path" ]]; then
                echo "[WARN] Skip eval: checkpoint not found: $ckpt_path"
                return 0
            fi
            python src/evaluation/eval_splatam_recon_v2.py \
                --ckpt "$ckpt_path" \
                --gt_mesh "$GT_MESH" \
                --transform_traj "data/mp3d_sim_nvs_v2/${scene}/traj.txt" \
                --result_dir "$out_dir"
        }

        eval_ckpt_if_exists \
            "${result_dir}/splatam/exploration_stage_0/params.npz" \
            "${result_dir}/eval_3d/exploration_stage_0"

        eval_ckpt_if_exists \
            "${result_dir}/splatam/exploration_stage_1/params.npz" \
            "${result_dir}/eval_3d/exploration_stage_1"

        eval_ckpt_if_exists \
            "${result_dir}/splatam/final/params.npz" \
            "${result_dir}/eval_3d/final"

    done
done
