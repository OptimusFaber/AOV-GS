#!/bin/bash

# Define the base directory containing the SCENE directories.
BASE_DIR=${1:-"data/replica_v1"}

sudo apt install jq

# Iterate over each SCENE directory within the base directory.
for scene_dir in "$BASE_DIR"/*/; do
  # Define the source JSON file path.
  src_json="$scene_dir/habitat/replica_stage.stage_config.json"
  
  # Define the destination JSON file path.
  dest_json="$scene_dir/habitat/replicaSDK_stage.stage_config.json"
  
  # Check if the source JSON file exists.
  if [[ -f "$src_json" ]]; then
    # Copy the source JSON file to the destination.
    cp "$src_json" "$dest_json"
    
    # Replace the specified content in the destination JSON file.
    jq '.up = [0, 1, 0]' "$dest_json" > tmp.json && mv tmp.json "$dest_json"
    jq '.front = [0, 0, -1]' "$dest_json" > tmp.json && mv tmp.json "$dest_json"
    jq '.semantic_up = [0, 0, 1]' "$dest_json" > tmp.json && mv tmp.json "$dest_json"
    jq '.semantic_front = [0, -1, 0]' "$dest_json" > tmp.json && mv tmp.json "$dest_json"


#    sed -i 's/"up": \[0, 0, 1\]/"up": [0, 1, 0]/' "$dest_json"
#    sed -i 's/"front": \[0, 1, 0\]/"front": [0, 0, -1]/' "$dest_json"
#    sed -i 's/"semantic_up": \[0, 0, 1\]/"semantic_up": [0, 0, 1]/' "$dest_json"
#    sed -i 's/"semantic_front": \[0, 0, 1\]/"semantic_up": [0, 0, 1]/' "$dest_json"
#        "up": [0, 1, 0],
#  "front": [0, 0, -1],
#  "semantic_up": [0, 0, 1],
#  "semantic_front": [0, -1, 0]
    
    echo "Processed $dest_json"
  else
    echo "Source file does not exist: $src_json"
  fi
done
