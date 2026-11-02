# NIH ChestX-ray14

112,120 frontal chest X-rays from 30,805 patients, released by the NIH Clinical Center, each with 14 disease labels
mined from its radiology report (Wang et al., "ChestX-ray8", CVPR 2017). About 43 GB.

**Access.** Public. Cite the paper and acknowledge the NIH Clinical Center. Original release:
https://nihcc.app.box.com/v/ChestXray-NIHCC

**Get it.** The script downloads the same files from a Hugging Face mirror
(https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset), which is faster than the original site:

```bash
bash applications/chest_xray/download_data.sh    # into data/nih_cxr14/raw/
```

To keep the images elsewhere (an external disk, say), set `NIH_CXR_DIR` for the download and `NIH_CXR14_DIR` (the
folder holding `raw/` and `derived/`) for the audit.

**Layout.**

```
nih_cxr14/
  raw/        images_001.zip ... images_012.zip, Data_Entry_2017_v2020.csv, BBox_List_2017.csv,
              train_val_list.txt, test_list.txt
  derived/    cxr_outputs.npz, cxr_pc_features.npz   (written by applications/chest_xray/inference.py)
```

The images are read straight from the zip files; there is no need to unzip them.

**Used by.** The chest X-ray application (`applications/chest_xray/`).
