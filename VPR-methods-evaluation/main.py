#!/usr/bin/env python3
"""
Main script for VPR evaluation.
This script evaluates VPR models on test datasets and computes recall metrics.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import faiss
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from parser import parse_arguments
from test_dataset import TestDataset
from vpr_models import get_model
from visualizations import save_preds

# Setup logging
def setup_logging(log_dir):
    log_dir = Path(log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / timestamp
    log_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_path / 'info.log'),
            logging.StreamHandler()
        ]
    )
    return log_path

def extract_features(model, dataloader, device):
    """Extract features from images using the VPR model."""
    model.eval()
    features = []
    indices = []
    
    with torch.no_grad():
        for images, idxs in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = F.normalize(feats, p=2, dim=1)
            features.append(feats.cpu())
            indices.extend(idxs.tolist())
    
    features = torch.cat(features, dim=0)
    return features.numpy(), indices

def compute_recall(predictions, positives_per_query, recall_values):
    """Compute recall@N metrics."""
    recalls = np.zeros(len(recall_values))
    
    for query_idx, preds in enumerate(predictions):
        positives = positives_per_query[query_idx]
        for i, n in enumerate(recall_values):
            if n <= len(preds) and any(pred in positives for pred in preds[:n]):
                recalls[i:] += 1
                break
    
    recalls = recalls / len(predictions) * 100.0
    return recalls

def main():
    args = parse_arguments()
    
    # Setup logging
    log_path = setup_logging(args.log_dir)
    logging.info(f"{Path(__file__).name} {' '.join(sys.argv[1:])}")
    logging.info(f"Arguments: {args}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create dataset
    logging.info(f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}")
    logging.info(f"The outputs are being saved in {log_path}")
    
    eval_ds = TestDataset(
        database_folder=args.database_folder,
        queries_folder=args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels
    )
    
    logging.info(f"Testing on < #queries: {eval_ds.num_queries}; #database: {eval_ds.num_database} >")
    
    # Create dataloader
    # Windows compatibility: set num_workers to 0 on Windows
    if os.name == 'nt':
        if args.num_workers > 0:
            logging.info(f"Windows detected: Setting num_workers to 0 (was {args.num_workers}) to avoid multiprocessing issues (socket.send() errors). This may be slower but more stable.")
            args.num_workers = 0
    
    dataloader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Get model
    model = get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.to(device)
    model.eval()
    
    # GPU memory check and batch size adjustment
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved_memory = torch.cuda.memory_reserved(0) / 1e9
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9
        available_memory = total_memory - reserved_memory
        
        logging.info(f"GPU Memory Status - Total: {total_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB, Allocated: {allocated_memory:.2f} GB, Available: {available_memory:.2f} GB")
        
        if available_memory < 4.0:
            new_batch_size = max(1, args.batch_size // 4)
            if new_batch_size < args.batch_size:
                logging.info(f"Low GPU memory ({available_memory:.2f} GB available). Reducing batch size from {args.batch_size} to {new_batch_size}")
                args.batch_size = new_batch_size
                dataloader = torch.utils.data.DataLoader(
                    eval_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True
                )
        elif available_memory < 6.0:
            new_batch_size = max(1, args.batch_size // 2)
            if new_batch_size < args.batch_size:
                logging.info(f"Moderate GPU memory ({available_memory:.2f} GB available). Reducing batch size from {args.batch_size} to {new_batch_size}")
                args.batch_size = new_batch_size
                dataloader = torch.utils.data.DataLoader(
                    eval_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True
                )
    
    # Extract features
    start_time = datetime.now()
    features, indices = extract_features(model, dataloader, device)
    
    # Split features into database and queries
    database_features = features[:eval_ds.num_database]
    query_features = features[eval_ds.num_database:]
    
    # Build FAISS index
    dimension = database_features.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(database_features.astype('float32'))
    
    # Search for nearest neighbors
    k = max(args.recall_values) if args.recall_values else 100
    k = min(k, eval_ds.num_database)
    
    distances, predictions = index.search(query_features.astype('float32'), k)
    
    # Compute recall if labels are available
    if args.use_labels:
        positives_per_query = eval_ds.get_positives()
        recalls = compute_recall(predictions, positives_per_query, args.recall_values)
        recall_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        logging.info(recall_str)
    
    # Save predictions if requested
    if args.num_preds_to_save > 0:
        logging.info("Saving final predictions")
        save_preds(
            predictions=predictions,
            eval_ds=eval_ds,
            log_dir=log_path,
            save_only_wrong_preds=args.save_only_wrong_preds,
            use_labels=args.use_labels
        )
    
    # Save data for uncertainty estimation if requested
    if args.save_for_uncertainty:
        z_data = {
            'database_utms': eval_ds.database_utms if args.use_labels else None,
            'queries_utms': eval_ds.queries_utms if args.use_labels else None,
            'predictions': predictions,
            'distances': distances,
            'database_features': database_features,
            'query_features': query_features,
        }
        torch.save(z_data, log_path / 'z_data.torch')
        logging.info(f"Saved z_data.torch to {log_path / 'z_data.torch'}")
    
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    logging.info(f"Total runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")

if __name__ == "__main__":
    main()

