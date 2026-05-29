#!/bin/bash
# Shared GPU helpers for AOV-GS shell scripts.
#
# resolve_train_device DEVICE_VAR_NAME
#   For language-field training: maps physical cuda:N (N>0) to CUDA_VISIBLE_DEVICES=N
#   and logical cuda:0 (required by diff_gaussian_rasterization).
#   Sets PHYSICAL_GPU when remapping occurs.

resolve_train_device() {
    local -n _dev_ref=$1
    PHYSICAL_GPU=""
    if [[ "$_dev_ref" =~ ^cuda:([0-9]+)$ ]]; then
        PHYSICAL_GPU="${BASH_REMATCH[1]}"
        if [[ "$PHYSICAL_GPU" != "0" ]]; then
            export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
            _dev_ref=cuda:0
        fi
    fi
}

print_train_device() {
    local device="$1"
    if [[ -n "${PHYSICAL_GPU:-}" && "$PHYSICAL_GPU" != "0" ]]; then
        echo "  Device       : cuda:0 (physical GPU ${PHYSICAL_GPU}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
    else
        echo "  Device       : ${device}"
    fi
}
