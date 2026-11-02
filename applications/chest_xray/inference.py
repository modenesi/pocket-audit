"""Chest X-ray application (NIH ChestX-ray14): model outputs and label-free image features, one row per image. Reads no label.

Run in a separate imaging environment (torch 2.7, torchxrayvision 1.3.4; see applications/chest_xray/README.md).
Images are read straight from the 12 NIH zip archives:

    python applications/chest_xray/inference.py --data-dir data/nih_cxr14 \
        --out data/nih_cxr14/derived/cxr_outputs.parquet

Per image: the Effusion logit of two pretrained TorchXRayVision DenseNet-121 models never trained on NIH
(``densenet121-res224-chex``, CheXpert; ``densenet121-res224-mimic_ch``, MIMIC-CXR), with the library's operating-
point normalization switched off, and the 512 channel means of the label-free ResNet autoencoder ``101-elastic``
(trained to reconstruct images from NIH, PadChest, RSNA, MIMIC-CXR and CheXpert; no labels). Preprocessing is the
library's: grey scale, normalize to [-1024, 1024], centre crop, resize to 224.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class ZipImages(torch.utils.data.Dataset):
    def __init__(self, names: list[str], where: dict[str, str]):
        self.names, self.where, self.zips = names, where, {}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        import torchxrayvision as xrv
        from PIL import Image
        name = self.names[i]
        zp = self.where[name]
        if zp not in self.zips:                      # opened lazily, once per worker process
            self.zips[zp] = zipfile.ZipFile(zp)
        with self.zips[zp].open(f"images/{name}") as fh:
            img = np.array(Image.open(fh).convert("L"), dtype=np.float32)
        img = xrv.datasets.normalize(img, 255)[None]
        img = xrv.datasets.XRayCenterCrop()(img)
        img = xrv.datasets.XRayResizer(224, engine="cv2" if _has_cv2() else "skimage")(img)
        return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)), i


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def to_npz(parquet: Path) -> Path:
    """Re-save a parquet output as .npz (image ids + float32 matrix): the audit env's older PyArrow cannot read
    these parquet files. Records both hashes."""
    d = pd.read_parquet(parquet)
    cols = [c for c in d.columns if c != "image"]
    out = parquet.with_suffix(".npz")
    np.savez_compressed(out, image=d["image"].to_numpy().astype(str), columns=np.array(cols),
                        values=d[cols].to_numpy(np.float32))
    meta_p = parquet.with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta["npz"] = {"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "columns": len(cols)}
    meta_p.write_text(json.dumps(meta, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    ap.add_argument("--encoder", choices=["ae", "pc"], default="ae",
                    help="ae: the outputs + autoencoder features (first pass); pc: only the 1024 pooled penultimate "
                         "features of densenet121-res224-pc (PadChest-trained, NIH-free), for pocket forming")
    ap.add_argument("--to-npz", action="store_true", help="only convert the existing --out parquet to .npz")
    args = ap.parse_args()
    if args.to_npz:
        print(to_npz(args.out))
        return
    import torchxrayvision as xrv
    raw = args.data_dir / "raw"
    names = pd.read_csv(raw / "Data_Entry_2017_v2020.csv", usecols=["Image Index"])["Image Index"].tolist()
    where = {}
    for zp in sorted(raw.glob("images_0*.zip")):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.endswith(".png"):
                    where[n.split("/")[-1]] = str(zp)
    missing = [n for n in names if n not in where]
    if missing:
        raise SystemExit(f"{len(missing)} listed images are not in the archives, e.g. {missing[:3]}")
    if args.limit:
        names = names[:args.limit]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    if args.encoder == "pc":
        return _pc_features(args, names, where, dev)
    for key, w in (("chex", "densenet121-res224-chex"), ("mimic", "densenet121-res224-mimic_ch")):
        m = xrv.models.DenseNet(weights=w).to(dev).eval()
        m.op_threshs = None                        # raw outputs: no operating-point normalization
        m.apply_sigmoid = False                    # keep logits
        models[key] = (m, m.pathologies.index("Effusion"))
    ae = xrv.autoencoders.ResNetAE(weights="101-elastic").to(dev).eval()
    dl = torch.utils.data.DataLoader(ZipImages(names, where), batch_size=args.batch, num_workers=args.workers,
                                     shuffle=False, pin_memory=True, persistent_workers=False)
    logit = {k: np.full(len(names), np.nan, np.float32) for k in models}
    feats = np.full((len(names), 512), np.nan, np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b, (x, idx) in enumerate(dl):
            x = x.to(dev, non_blocking=True)
            for k, (m, j) in models.items():
                logit[k][idx.numpy()] = m(x)[:, j].float().cpu().numpy()
            feats[idx.numpy()] = ae.encode(x).mean(dim=(2, 3)).float().cpu().numpy()
            if b % 50 == 0:
                print(f"{(b + 1) * args.batch}/{len(names)} images, {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame({"image": names, "logit_chex": logit["chex"], "logit_mimic": logit["mimic"]})
    out = pd.concat([out, pd.DataFrame(feats, columns=[f"ae{j:03d}" for j in range(512)])], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    h = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta = dict(argv=sys.argv, n_images=len(names), seconds=round(time.time() - t0), sha256=h,
                torch=torch.__version__, torchxrayvision=xrv.__version__, device=str(dev),
                nan_rows=int(out[["logit_chex", "logit_mimic"]].isna().any(axis=1).sum()))
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def _pc_features(args, names, where, dev):
    import torchxrayvision as xrv
    enc = xrv.models.DenseNet(weights="densenet121-res224-pc").to(dev).eval()
    dl = torch.utils.data.DataLoader(ZipImages(names, where), batch_size=args.batch, num_workers=args.workers,
                                     shuffle=False, pin_memory=True)
    feats = np.full((len(names), 1024), np.nan, np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b, (x, idx) in enumerate(dl):
            feats[idx.numpy()] = enc.features2(x.to(dev, non_blocking=True)).float().cpu().numpy()
            if b % 100 == 0:
                print(f"{(b + 1) * args.batch}/{len(names)} images, {time.time() - t0:.0f}s", flush=True)
    out = pd.concat([pd.DataFrame({"image": names}),
                     pd.DataFrame(feats, columns=[f"pc{j:04d}" for j in range(1024)])], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    meta = dict(argv=sys.argv, n_images=len(names), seconds=round(time.time() - t0),
                sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(), torch=torch.__version__,
                torchxrayvision=xrv.__version__, encoder="densenet121-res224-pc features2 (1024)",
                nan_rows=int(out.isna().any(axis=1).sum()))
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
