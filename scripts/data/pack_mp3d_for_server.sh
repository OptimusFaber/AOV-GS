#!/bin/bash
# Pack MP3D data for transfer to a remote server.
#
# Includes only the 5 benchmark scenes used by ActiveSGM/AOV-GS:
#   GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH
#
# Contents (~10 GB uncompressed):
#   MP3D/v1/scans/{scene}/          — GT mesh + house_segmentations
#   MP3D/v1/tasks/mp3d/{scene}/       — Habitat .glb + semantic meshes
#   mp3d_sim_nvs_v2/{scene}/          — NVS eval trajectories (optional)
#
# Redundant *.zip archives in scans/ are skipped when extracted folders exist.
#
# Usage:
#   bash scripts/data/pack_mp3d_for_server.sh [OUTPUT_ARCHIVE] [DATA_ROOT]
#
# Format is chosen from the archive extension (default: .tar.gz):
#   .tar.gz / .tgz  — gzip  (tar -xzf on server, no extra tools)
#   .tar.zst        — zstd  (needs zstd on server)
#
# Example:
#   bash scripts/data/pack_mp3d_for_server.sh /mnt/data/mp3d_active_sgm_5scenes.tar.gz /mnt/data
#
# On the server:
#   mkdir -p /path/to/data
#   tar -xzf mp3d_active_sgm_5scenes.tar.gz -C /path/to/data
#   ln -sfn /path/to/data/MP3D /path/to/AOV-GS/data/MP3D
#   ln -sfn /path/to/data/mp3d_sim_nvs_v2 /path/to/AOV-GS/data/mp3d_sim_nvs_v2

set -euo pipefail

SCENES=(GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH)
OUTPUT="${1:-/mnt/data/mp3d_active_sgm_5scenes.tar.gz}"
DATA_ROOT="${2:-/mnt/data}"
GZIP_LEVEL="${GZIP_LEVEL:-6}"
ZSTD_LEVEL="${ZSTD_LEVEL:-6}"

if [[ ! -d "${DATA_ROOT}/MP3D/v1" ]]; then
    echo "[ERROR] Missing ${DATA_ROOT}/MP3D/v1" >&2
    exit 1
fi

for scene in "${SCENES[@]}"; do
    if [[ ! -f "${DATA_ROOT}/MP3D/v1/tasks/mp3d/${scene}/${scene}.glb" ]]; then
        echo "[ERROR] Missing Habitat mesh for ${scene}" >&2
        exit 1
    fi
    if [[ ! -f "${DATA_ROOT}/MP3D/v1/scans/${scene}/mesh.obj" ]]; then
        echo "[ERROR] Missing GT mesh for ${scene}" >&2
        exit 1
    fi
done

TAR_PATHS=()
for scene in "${SCENES[@]}"; do
    TAR_PATHS+=("MP3D/v1/scans/${scene}")
    TAR_PATHS+=("MP3D/v1/tasks/mp3d/${scene}")
done

if [[ -d "${DATA_ROOT}/mp3d_sim_nvs_v2" ]]; then
    for scene in "${SCENES[@]}"; do
        if [[ -d "${DATA_ROOT}/mp3d_sim_nvs_v2/${scene}" ]]; then
            TAR_PATHS+=("mp3d_sim_nvs_v2/${scene}")
        fi
    done
fi

case "${OUTPUT}" in
    *.tar.zst) FORMAT="zstd"; COMP_LABEL="zstd level ${ZSTD_LEVEL}" ;;
    *.tar.gz|*.tgz) FORMAT="gzip"; COMP_LABEL="gzip level ${GZIP_LEVEL}" ;;
    *)
        echo "[ERROR] Unsupported archive extension: ${OUTPUT}" >&2
        echo "Use .tar.gz (recommended) or .tar.zst" >&2
        exit 1
        ;;
esac

echo "=============================================="
echo "  Packing MP3D for server transfer"
echo "  Data root : ${DATA_ROOT}"
echo "  Output    : ${OUTPUT}"
echo "  Format    : ${FORMAT} (${COMP_LABEL})"
echo "  Scenes    : ${SCENES[*]}"
echo "=============================================="

du -ch "${TAR_PATHS[@]/#/${DATA_ROOT}/}" 2>/dev/null | tail -1 || true

mkdir -p "$(dirname "${OUTPUT}")"

TAR_EXCLUDES=(
    --exclude='house_segmentations.zip'
    --exclude='matterport_mesh.zip'
    --exclude='poisson_meshes.zip'
)

if [[ "${FORMAT}" == "gzip" ]]; then
    if command -v pigz >/dev/null 2>&1; then
        tar -cf - "${TAR_EXCLUDES[@]}" -C "${DATA_ROOT}" "${TAR_PATHS[@]}" \
            | pigz "-${GZIP_LEVEL}" > "${OUTPUT}"
    else
        tar -czf "${OUTPUT}" "${TAR_EXCLUDES[@]}" -C "${DATA_ROOT}" "${TAR_PATHS[@]}"
    fi
else
    tar -cf - "${TAR_EXCLUDES[@]}" -C "${DATA_ROOT}" "${TAR_PATHS[@]}" \
        | zstd -T0 "-${ZSTD_LEVEL}" -o "${OUTPUT}"
fi

echo ""
echo "Done: ${OUTPUT}"
ls -lh "${OUTPUT}"
