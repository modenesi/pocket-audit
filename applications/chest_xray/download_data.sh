#!/bin/bash
# Download NIH ChestX-ray14 (public, NIH Clinical Center) from the Hugging Face mirror and check every file against
# the SHA-256 of the version used in the paper (expected_sha256.txt, next to this script). If a check fails, the
# mirror has changed; download the same files from the NIH original instead (see data/nih_cxr14/README.md).
# Usage: bash applications/chest_xray/download_data.sh   (set NIH_CXR_DIR to save elsewhere)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
D="${NIH_CXR_DIR:-$HERE/../../data/nih_cxr14/raw}"
mkdir -p "$D" && cd "$D"
B=https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data
ok() { grep "  $1\$" "$HERE/expected_sha256.txt" | sha256sum -c --quiet - > /dev/null 2>&1; }
for f in Data_Entry_2017_v2020.csv BBox_List_2017.csv test_list.txt train_val_list.txt; do
  ok "$f" || curl -fsSL --retry 5 -o "$f" "$B/$f"
done
for i in $(seq -w 1 12); do
  f=images_0$i.zip
  ok "$f" && continue                         # already downloaded and verified
  curl -fSL --retry 5 -C - -o "$f" "$B/images/$f"
done
sha256sum -c "$HERE/expected_sha256.txt"
echo "Download complete and verified: $D"
