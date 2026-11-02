#!/bin/bash
# Download NIH ChestX-ray14 (public, NIH Clinical Center) from the Hugging Face mirror.
# Usage: bash applications/chest_xray/download_data.sh   (set NIH_CXR_DIR to save elsewhere)
set -u
D="${NIH_CXR_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/data/nih_cxr14/raw}"
mkdir -p "$D" && cd "$D"
B=https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data
for f in Data_Entry_2017_v2020.csv BBox_List_2017.csv test_list.txt train_val_list.txt; do
  curl -sL --retry 5 -o "$f" "$B/$f"
done
for i in $(seq -w 1 12); do
  f=images_0$i.zip
  curl -sL --retry 5 -C - -o "$f" "$B/images/$f" || echo "FAIL $f"
  ls -l "$f"
done
curl -s "https://huggingface.co/api/datasets/alkzar90/NIH-Chest-X-ray-dataset/tree/main/data/images" > lfs_listing.json
echo DOWNLOAD_DONE
