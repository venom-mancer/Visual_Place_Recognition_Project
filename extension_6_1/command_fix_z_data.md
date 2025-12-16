# Command to Fix z_data.torch Issue (Without Wandb)

## Problem
The `z_data.torch` file is missing because the VPR evaluation was run without the `--save_for_uncertainty` flag.

## Solution
Re-run the VPR evaluation with the `--save_for_uncertainty` flag, but disable Wandb logging.

## Command (Windows PowerShell)

```powershell
$env:WANDB_MODE="disabled"; .venv\Scripts\python VPR-methods-evaluation\main.py --num_workers 8 --batch_size 32 --log_dir log_svox_train --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 --image_size 512 512 --database_folder data/svox/images/train/gallery --queries_folder data/svox/images/train/queries --num_preds_to_save 20 --recall_values 1 5 10 20 --save_for_uncertainty --device cuda
```

## Command (Windows CMD)

```cmd
set WANDB_MODE=disabled && .venv\Scripts\python VPR-methods-evaluation\main.py --num_workers 8 --batch_size 32 --log_dir log_svox_train --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 --image_size 512 512 --database_folder data/svox/images/train/gallery --queries_folder data/svox/images/train/queries --num_preds_to_save 20 --recall_values 1 5 10 20 --save_for_uncertainty --device cuda
```

## Command (Git Bash / Linux)

```bash
WANDB_MODE=disabled .venv/Scripts/python VPR-methods-evaluation/main.py --num_workers 8 --batch_size 32 --log_dir log_svox_train --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 --image_size 512 512 --database_folder data/svox/images/train/gallery --queries_folder data/svox/images/train/queries --num_preds_to_save 20 --recall_values 1 5 10 20 --save_for_uncertainty --device cuda
```

## What This Does

1. **Sets `WANDB_MODE=disabled`**: Prevents Wandb from logging anything (runs in offline/disabled mode)
2. **Runs VPR evaluation**: Extracts descriptors and performs retrieval
3. **Saves z_data.torch**: The `--save_for_uncertainty` flag creates the required file with:
   - `database_utms`: Database image poses
   - `predictions`: Retrieval prediction indices
   - `distances`: Descriptor distances

## Expected Output Location

After running, the file will be created at:
```
logs/log_svox_train/YYYY-MM-DD_HH-MM-SS/z_data.torch
```

Note: The timestamp in the path will be different from your previous run, so you'll need to update the path in the Stage 1 command accordingly.

## After Running

Once `z_data.torch` is created, you can proceed with Stage 1 feature extraction using the new log directory path.

