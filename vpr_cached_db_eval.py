#!/usr/bin/env python3
"""
Fast VPR evaluation for SVOX subsets (night/sun) by reusing cached database descriptors.

Motivation:
- Running `VPR-methods-evaluation/main.py` on SVOX night/sun subsets is slow because it recomputes
  gallery (database) descriptors for ~17k images each time.
- We already have `database_features` and `database_utms` saved in a previous full SVOX run's `z_data.torch`.

This script:
1) Loads cached `database_features` (+ optionally `database_utms`) from a `z_data.torch`
2) Extracts query descriptors for a new queries folder using the same model+transforms as VPR eval
3) Searches nearest neighbors with FAISS (L2) and saves:
   - preds/ .txt files (same format as VPR eval)
   - z_data.torch (predictions, distances, query_features, plus reused database_features)

Usage example:
python vpr_cached_db_eval.py ^
  --cached-z-data log_svox_test/2025-12-18_16-01-59/z_data.torch ^
  --database-folder data/svox/images/test/gallery ^
  --queries-folder data/svox/images/test/queries_night ^
  --log-dir logs/log_svox_night_test_cached ^
  --device cuda ^
  --image-size 512 512 ^
  --num-preds-to-save 20
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import faiss

# Ensure local VPR-methods-evaluation is importable
import sys
sys.path.insert(0, str(Path(__file__).parent / "VPR-methods-evaluation"))

from test_dataset import TestDataset
from vpr_models import get_model
from visualizations import save_preds


class QueryOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, queries_paths: list[str], transform):
        self.queries_paths = queries_paths
        self.transform = transform

    def __len__(self):
        return len(self.queries_paths)

    def __getitem__(self, idx):
        from PIL import Image
        p = self.queries_paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), idx


def extract_query_features(model, dataloader, device: torch.device) -> np.ndarray:
    model.eval()
    feats = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            out = model(images)
            if isinstance(out, tuple):
                out = out[0]
            out = F.normalize(out, p=2, dim=1)
            feats.append(out.cpu())
    feats = torch.cat(feats, dim=0)
    return feats.numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Cached-db VPR evaluation for SVOX subsets")
    ap.add_argument("--cached-z-data", type=str, required=True)
    ap.add_argument("--database-folder", type=str, required=True)
    ap.add_argument("--queries-folder", type=str, required=True)
    ap.add_argument("--log-dir", type=str, required=True)

    ap.add_argument("--method", type=str, default="cosplace")
    ap.add_argument("--backbone", type=str, default="ResNet18")
    ap.add_argument("--descriptors-dimension", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--num-preds-to-save", type=int, default=20)
    ap.add_argument("--positive-dist-threshold", type=int, default=25)
    args = ap.parse_args()

    cached = torch.load(args.cached_z_data, weights_only=False)
    database_features = np.asarray(cached["database_features"]).astype("float32")
    database_utms = cached.get("database_utms", None)

    # Build dataset for labels + writing preds
    eval_ds = TestDataset(
        database_folder=args.database_folder,
        queries_folder=args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=True,
    )

    # Query-only loader using same transform
    q_ds = QueryOnlyDataset(eval_ds.queries_paths, eval_ds.transform)
    q_loader = torch.utils.data.DataLoader(
        q_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = get_model(args.method, args.backbone, args.descriptors_dimension).to(device)
    model.eval()

    start = datetime.now()
    query_features = extract_query_features(model, q_loader, device)

    # FAISS search
    dim = database_features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(database_features)
    k = min(args.num_preds_to_save, database_features.shape[0])
    distances, predictions = index.search(query_features.astype("float32"), k)

    # Output dir (timestamped like main.py)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.log_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save preds + z_data
    save_preds(
        predictions=predictions,
        eval_ds=eval_ds,
        log_dir=out_dir,
        save_only_wrong_preds=False,
        use_labels=True,
    )

    z_data = {
        "database_utms": eval_ds.database_utms if database_utms is None else database_utms,
        "queries_utms": eval_ds.queries_utms,
        "predictions": predictions,
        "distances": distances,
        "database_features": database_features,
        "query_features": query_features,
    }
    torch.save(z_data, out_dir / "z_data.torch")

    runtime = (datetime.now() - start).total_seconds()
    print(f"Saved: {out_dir}")
    print(f"  preds/: {out_dir / 'preds'}")
    print(f"  z_data: {out_dir / 'z_data.torch'}")
    print(f"Runtime: {runtime/60:.2f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


