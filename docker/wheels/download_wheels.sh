#!/bin/bash
# Скачать wheels на машине с нормальным интернетом, положить в docker/wheels/, затем docker build.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

wget -c --tries=10 -O pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl \
  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl

wget -c --tries=10 -O torch_scatter-2.1.1+pt113cu117-cp38-cp38-linux_x86_64.whl \
  https://data.pyg.org/whl/torch-1.13.1+cu117/torch_scatter-2.1.1+pt113cu117-cp38-cp38-linux_x86_64.whl

wget -c --tries=10 -O torch_sparse-0.6.17+pt113cu117-cp38-cp38-linux_x86_64.whl \
  https://data.pyg.org/whl/torch-1.13.1+cu117/torch_sparse-0.6.17+pt113cu117-cp38-cp38-linux_x86_64.whl

ls -lh *.whl
echo "OK — теперь: bash docker/build.sh"
