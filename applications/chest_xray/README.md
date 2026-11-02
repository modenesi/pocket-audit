# Chest X-rays (NIH ChestX-ray14)

NIH ChestX-ray14 holds 112,120 frontal chest X-rays from 30,805 patients, each with 14 disease labels mined from
its radiology report (Wang et al., CVPR 2017).

**What we audit.** Two public DenseNet-121 classifiers from TorchXRayVision, trained on CheXpert and on MIMIC-CXR
(neither saw NIH images), predicting pleural effusion. We use each patient's first image, so patients are
independent. Before auditing, each classifier's effusion logit is mapped to a probability by an increasing
(isotonic) regression fit on development images and clipped to [0.001, 0.999]; the audited objects are these
recalibrated pipelines. Pockets are formed from age, view position (bedside AP or standard PA) and label-free image
features from a third model, trained on PadChest. Contrast: female vs. male.

**Data.** Public. About 43 GB. See [`data/nih_cxr14/README.md`](../../data/nih_cxr14/README.md):

```bash
bash applications/chest_xray/download_data.sh     # to data/nih_cxr14/raw/, checks every file's SHA-256
```

If you keep the data elsewhere, set `NIH_CXR14_DIR` to the folder that holds `raw/` and `derived/`.

**Step 1: model outputs and image features.** This step needs PyTorch and TorchXRayVision, which the audit itself
does not, so use a separate environment (the paper used torch 2.7.0 and torchvision 0.22.0, CUDA 11.8 builds, and
torchxrayvision 1.3.4). It reads no label.

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchxrayvision==1.3.4 scikit-image pyarrow
D="${NIH_CXR14_DIR:-data/nih_cxr14}"
python applications/chest_xray/inference.py --data-dir $D --out $D/derived/cxr_outputs.parquet --workers 16
python applications/chest_xray/inference.py --data-dir $D --out $D/derived/cxr_pc_features.parquet --encoder pc --workers 12
python applications/chest_xray/inference.py --data-dir $D --out $D/derived/cxr_outputs.parquet --to-npz
python applications/chest_xray/inference.py --data-dir $D --out $D/derived/cxr_pc_features.parquet --to-npz
```

TorchXRayVision downloads the model weights to `~/.torchxrayvision/` on first use.

**Step 2: the audit**, in the main environment:

```bash
python applications/chest_xray/audit_cxr.py discover --run-dir results/chest_xray --n-jobs 24
python applications/chest_xray/audit_cxr.py describe --run-dir results/chest_xray
python applications/chest_xray/audit_cxr.py validate --run-dir results/chest_xray --n-jobs 28
python applications/chest_xray/audit_cxr.py audit    --run-dir results/chest_xray
mkdir -p results/tables
python applications/chest_xray/tables.py --run-dir results/chest_xray --out results/tables
```

The effusion labels of audit images are parsed only in the `audit` stage, which can run once per run directory: it
leaves a lock file, `AUDIT_RESERVED`.
