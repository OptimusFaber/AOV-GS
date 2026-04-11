#!/bin/bash
while read p; do
  python ./src/data/download_mp.py -o /mnt/Data4/slam_datasets/MP3D --id "$p" --task_data habitat
#  echo "$p"
done <./scripts/data/scan_id.txt
