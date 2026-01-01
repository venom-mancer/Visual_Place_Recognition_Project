"""
Run image matching only for hard queries (adaptive approach).
Reads hard query indices from a text file and only processes those queries.
"""

import os
import sys
import argparse
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from util import read_file_preds
from setup_temp_dir import setup_project_temp_directory

# Insert image-matching-models at the beginning of sys.path
image_matching_path = str(Path(__file__).parent.joinpath("image-matching-models"))
if image_matching_path not in sys.path:
    sys.path.insert(0, image_matching_path)

from matching import get_matcher, available_models
from matching.utils import get_default_device


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run image matching only for hard queries")
    
    parser.add_argument("--preds-dir", type=str, required=True, help="directory with predictions of a VPR model")
    parser.add_argument("--hard-queries-list", type=str, required=True, help="text file with hard query indices (one per line)")
    parser.add_argument("--out-dir", type=str, required=True, help="output directory of image matching results")
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--im-size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to match")
    
    return parser.parse_args()


def load_hard_query_indices(hard_queries_file):
    """Load hard query indices from text file."""
    with open(hard_queries_file, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    return set(indices)


def main(args):
    # Set up project-local temporary directory first
    temp_dir = setup_project_temp_directory()
    print(f"Temporary files will be stored in: {temp_dir}")
    
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    num_preds = args.num_preds
    matcher = get_matcher(matcher_name, device=device)
    preds_folder = args.preds_dir
    
    # Load hard query indices
    hard_query_indices = load_hard_query_indices(args.hard_queries_list)
    print(f"Loaded {len(hard_query_indices)} hard query indices from {args.hard_queries_list}")
    
    output_folder = Path(args.out_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    
    total_queries = len(txt_files)
    hard_queries_to_process = [i for i in range(total_queries) if i in hard_query_indices]
    
    print(f"Total queries: {total_queries}")
    print(f"Hard queries to process: {len(hard_queries_to_process)}")
    print(f"Easy queries to skip: {total_queries - len(hard_queries_to_process)}")
    print(f"Expected time savings: {100 * (total_queries - len(hard_queries_to_process)) / total_queries:.1f}%")
    print()
    
    # Process only hard queries
    processed = 0
    skipped = 0
    
    for idx, txt_file in enumerate(tqdm(txt_files, desc="Processing queries")):
        if idx not in hard_query_indices:
            # Easy query: create empty file for compatibility
            q_num = Path(txt_file).stem
            out_file = output_folder.joinpath(f"{q_num}.torch")
            if not out_file.exists():
                torch.save([], out_file)  # Empty list for easy queries
            skipped += 1
            continue
        
        # Hard query: do actual image matching
        q_num = Path(txt_file).stem
        out_file = output_folder.joinpath(f"{q_num}.torch")
        if out_file.exists():
            continue
        
        results = []
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=img_size)
        for pred_path in pred_paths[:num_preds]:
            img1 = matcher.load_image(pred_path, resize=img_size)
            result = matcher(deepcopy(img0), img1)
            result["all_desc0"] = result["all_desc1"] = None
            results.append(result)
        torch.save(results, out_file)
        processed += 1
    
    print(f"\nCompleted:")
    print(f"  Hard queries processed (image matching): {processed}")
    print(f"  Easy queries skipped (empty files): {skipped}")
    print(f"  Output directory: {output_folder}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

