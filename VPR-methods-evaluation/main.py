import parser
import sys
import os
from datetime import datetime
from pathlib import Path
import time
import gc
import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import wandb
import visualizations
import vpr_models
from test_dataset import TestDataset

# Import temp directory setup utility
sys.path.insert(0, str(Path(__file__).parent.parent))
from setup_temp_dir import setup_project_temp_directory

# Fix for Windows multiprocessing issues
# Set multiprocessing start method early to avoid socket.send() errors
if sys.platform == "win32":
    import multiprocessing
    # Explicitly set spawn method (default on Windows, but being explicit helps)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass


def main(args):
    # Set up project-local temporary directory first
    setup_project_temp_directory()
    
    start_time = datetime.now()
    start_time_seconds = time.time()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    # Initialize wandb
    wandb.init(
        project="visual-place-recognition",
        name=f"{args.method}_{args.backbone}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            "method": args.method,
            "backbone": args.backbone,
            "descriptors_dimension": args.descriptors_dimension,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "image_size": args.image_size,
            "positive_dist_threshold": args.positive_dist_threshold,
            "recall_values": args.recall_values,
            "log_dir": str(log_dir),
            "use_fp16": args.use_fp16,
        }
    )

    # Set PyTorch CUDA memory allocator to reduce fragmentation (if CUDA is available)
    if args.device == "cuda" and torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Aggressively clear any existing CUDA cache from previous runs
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated(0) / 1024**3
        initial_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"Initial GPU Memory - Allocated: {initial_memory:.2f} GB, "
                   f"Reserved: {initial_reserved:.2f} GB")
        if initial_reserved > 1.0:  # If more than 1GB is already reserved, warn user
            logger.warning(f"WARNING: {initial_reserved:.2f} GB already reserved on GPU. "
                          "Consider restarting Python to clear memory from previous runs.")

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)
    
    # Enable FP16 (half precision) if requested and CUDA is available
    use_amp = args.use_fp16 and args.device == "cuda" and torch.cuda.is_available()
    if use_amp:
        # Check if GPU supports FP16 (compute capability >= 7.0)
        compute_capability = torch.cuda.get_device_capability(0)
        if compute_capability[0] >= 7:
            logger.info(f"✅ FP16 (Mixed Precision) enabled. GPU compute capability: {compute_capability[0]}.{compute_capability[1]}")
            logger.info("   This will reduce memory usage by ~50% and may speed up inference.")
            logger.info("   Recommended for memory-intensive models like NetVLAD.")
        else:
            logger.warning(f"⚠️  GPU compute capability ({compute_capability[0]}.{compute_capability[1]}) may not fully support FP16. "
                          "FP16 will still be used but may be slower. Consider using FP32 instead.")
    elif args.use_fp16 and args.device != "cuda":
        logger.warning("FP16 requested but device is not CUDA. FP16 will be disabled.")
        use_amp = False
    elif args.use_fp16 and not torch.cuda.is_available():
        logger.warning("FP16 requested but CUDA is not available. FP16 will be disabled.")
        use_amp = False
    
    # Log memory after model loading
    if args.device == "cuda" and torch.cuda.is_available():
        model_memory = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"GPU Memory after model load: {model_memory:.2f} GB")

    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    logger.info(f"Testing on {test_ds}")

    # Fix for Windows: multiprocessing with num_workers > 0 can cause socket.send() errors
    # Windows doesn't support fork(), so we need to use spawn which can cause hangs and socket errors
    if sys.platform == "win32":
        if args.num_workers > 0:
            logger.warning(
                f"Windows detected: Setting num_workers to 0 (was {args.num_workers}) "
                "to avoid multiprocessing issues (socket.send() errors). This may be slower but more stable."
            )
        num_workers = 0
        # Additional Windows-specific DataLoader settings
        pin_memory = False  # Disable pin_memory on Windows to avoid issues
        persistent_workers = False  # Disable persistent workers on Windows
    else:
        num_workers = args.num_workers
        pin_memory = True if args.device == "cuda" else False
        persistent_workers = True if num_workers > 0 else False

    # Adaptive batch size reduction based on available memory
    effective_batch_size = args.batch_size
    available_memory_gb = 8.0  # Default, will be updated if CUDA is available
    if args.device == "cuda" and torch.cuda.is_available():
        # Get actual available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        available_memory_gb = total_memory - reserved_memory
        
        logger.info(f"GPU Memory Status - Total: {total_memory:.2f} GB, "
                   f"Reserved: {reserved_memory:.2f} GB, "
                   f"Allocated: {allocated_memory:.2f} GB, "
                   f"Available: {available_memory_gb:.2f} GB")
        
        # Force batch_size=1 if memory is critically low (< 1.5 GB free)
        if available_memory_gb < 1.5:
            effective_batch_size = 1
            logger.warning(f"CRITICAL: Very low GPU memory ({available_memory_gb:.2f} GB available). "
                          f"FORCING batch_size=1 (was {args.batch_size}). "
                          "Consider clearing GPU memory or restarting Python.")
        elif available_memory_gb < 2.5 and args.batch_size > 4:
            effective_batch_size = min(4, args.batch_size // 2)
            logger.warning(f"Low GPU memory ({available_memory_gb:.2f} GB available). "
                          f"Reducing batch size from {args.batch_size} to {effective_batch_size}")
        elif available_memory_gb < 4.0 and args.batch_size > 8:
            effective_batch_size = min(8, args.batch_size // 2)
            logger.info(f"Moderate GPU memory ({available_memory_gb:.2f} GB available). "
                       f"Reducing batch size from {args.batch_size} to {effective_batch_size}")
    
    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, 
            num_workers=num_workers, 
            batch_size=effective_batch_size,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        
        for batch_idx, (images, indices) in enumerate(tqdm(database_dataloader)):
            try:
                # Move images to device
                images_gpu = images.to(args.device, non_blocking=True)
                
                # Extract descriptors with optional FP16 (mixed precision)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    descriptors = model(images_gpu)
                
                # Move to CPU immediately and convert to float32 for numpy compatibility
                # FP16 tensors need explicit conversion to float32 before numpy
                if use_amp and descriptors.dtype == torch.float16:
                    descriptors = descriptors.float()
                descriptors = descriptors.cpu().numpy()
                
                # Store descriptors
                all_descriptors[indices.numpy(), :] = descriptors
                
                # Aggressively delete GPU tensors
                del images_gpu, descriptors, images
                
                # Clear cache after EVERY batch if memory is low, otherwise every 5 batches
                if args.device == "cuda" and torch.cuda.is_available():
                    if available_memory_gb < 2.0 or (batch_idx + 1) % 5 == 0:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure all operations complete
                    
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Out of memory at batch {batch_idx}. Current batch size: {effective_batch_size}")
                # Clear everything possible
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                
                if effective_batch_size > 1:
                    effective_batch_size = max(1, effective_batch_size // 2)
                    logger.warning(f"Reducing batch size to {effective_batch_size} and retrying from batch {batch_idx}...")
                    # Recreate dataloader with smaller batch size
                    database_dataloader = DataLoader(
                        dataset=database_subset_ds, 
                        num_workers=num_workers, 
                        batch_size=effective_batch_size,
                        pin_memory=pin_memory,
                        persistent_workers=persistent_workers,
                        prefetch_factor=2 if num_workers > 0 else None
                    )
                    # Continue from where we left off (skip already processed batches)
                    # Note: This is a simple retry - for production, implement proper resume logic
                    logger.error("Cannot automatically resume. Please restart with --batch_size 1")
                    raise RuntimeError(f"Out of memory. Please restart with --batch_size {effective_batch_size} or clear GPU memory first.")
                else:
                    logger.error("Already using batch_size=1. GPU memory is critically low.")
                    logger.error("SOLUTION: Run 'python clear_gpu_memory.py' or restart Python/IDE to clear GPU memory.")
                    raise

        # Clear cache after database extraction
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug(f"After database extraction - GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds, 
            num_workers=num_workers, 
            batch_size=1,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if num_workers > 0 else None
        )
        for images, indices in tqdm(queries_dataloader):
            try:
                images_gpu = images.to(args.device, non_blocking=True)
                # Extract descriptors with optional FP16 (mixed precision)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    descriptors = model(images_gpu)
                # Move to CPU and convert to float32 if needed
                if use_amp and descriptors.dtype == torch.float16:
                    descriptors = descriptors.float()
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors
                del images_gpu, descriptors, images
                # Clear cache after every query if memory is low
                if args.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as e:
                logger.error("Out of memory during query extraction.")
                torch.cuda.empty_cache()
                gc.collect()
                logger.error("SOLUTION: Run 'python clear_gpu_memory.py' or restart Python/IDE to clear GPU memory.")
                raise

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Clear GPU memory after descriptor extraction is complete
    if args.device == "cuda" and torch.cuda.is_available():
        # Move model to CPU to free GPU memory for FAISS operations
        model_cpu = model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug(f"After descriptor extraction - GPU Memory - Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB, "
                    f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        # Model is no longer needed, but keep reference for cleanup later
        model = model_cpu

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    gc.collect()

    logger.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

    # For each query, check if the predictions are correct
    if args.use_labels:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by num_queries and multiply by 100, so the recalls are in percentages
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        logger.info(recalls_str)

    # Calculate total runtime
    end_time_seconds = time.time()
    total_time_seconds = end_time_seconds - start_time_seconds
    total_time_minutes = total_time_seconds / 60
    total_time_hours = total_time_minutes / 60

    # Log metrics to wandb
    metrics = {
        "total_time_seconds": total_time_seconds,
        "total_time_minutes": total_time_minutes,
        "total_time_hours": total_time_hours,
        "num_queries": test_ds.num_queries,
        "num_database": test_ds.num_database,
    }
    
    # Add recall metrics if labels were used
    if args.use_labels:
        for val, rec in zip(args.recall_values, recalls):
            metrics[f"recall@{val}"] = rec
    
    wandb.log(metrics)
    logger.info(f"Total runtime: {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)")
    
    wandb.finish()

    # Final cleanup: clear GPU memory
    if args.device == "cuda" and torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("GPU memory cleared")

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels
        )

    if args.save_for_uncertainty:
        z_data = {}
        z_data['database_utms'] = test_ds.database_utms
        z_data['positives_per_query'] = positives_per_query
        z_data['predictions'] = predictions
        z_data['distances'] = distances

        torch.save(z_data, log_dir / "z_data.torch")

if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
