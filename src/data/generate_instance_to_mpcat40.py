"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import json
import pandas as pd

if __name__ == "__main__":

    category_mapping_file = './configs/MP3D/category_mapping.tsv'

    # Load category mapping file
    mapping_df = pd.read_csv(category_mapping_file, sep="\t")

    # Remap mpcat40==41 to 0, and assign label as 'unknown' for index 0
    mapping_df.loc[mapping_df["mpcat40index"] == 41, "mpcat40index"] = 0

    # label_index → mpcat40
    label_to_mpcat40 = mapping_df.set_index("index")["mpcat40index"].to_dict()

    MP3D_scenes = ["GdvgFV5R1Z5","gZ6f7yhEvPG","HxpKQynjfin","pLe4wQe7qrG","YmJkqBEsHnH"]
    for scene in MP3D_scenes:

        scene_seg_file = f'./data/MP3D/v1/scans/{scene}/{scene}/house_segmentations/{scene}.semseg.json'
        # Load semseg JSON
        with open(scene_seg_file, "r") as f:
            semseg_data = json.load(f)

        # instance_id → mpcat40
        instance_to_mpcat40 = {
            str(group["id"]): int(label_to_mpcat40.get(group["label_index"], 0))
            for group in semseg_data["segGroups"]
        }

        # Save to a new JSON file
        instance_to_mpcat40_file = f'./configs/MP3D/{scene}/instance_to_mpcat40.json'
        with open(instance_to_mpcat40_file, "w") as f:
            json.dump(instance_to_mpcat40, f, indent=2)