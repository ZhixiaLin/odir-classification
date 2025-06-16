"""
Data cleaning script for managing misclassified images.
"""

import os
import sys
import argparse
import random
import shutil
import logging
from pathlib import Path
from tqdm import tqdm

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging configuration."""
    # Define log directory
    log_dir = os.path.join('data', 'odir4', 'misclassified')
    os.makedirs(log_dir, exist_ok=True)
    
    # Build log file path
    log_file_path = os.path.join(log_dir, 'clean_misclassified.log')
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Add stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logging.info("\n--- Starting new script run ---")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Manage misclassified images based on the misclassified_images.txt file.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--action', 
        type=str, 
        required=True, 
        choices=['move_all', 'move_percentage', 'clean_quarantine'],
        help="""Action to perform:
- move_all:           Move all misclassified images to the quarantine directory.
- move_percentage:    Move a percentage of misclassified images, keeping some in the original dataset.
- clean_quarantine:   Delete all images from the quarantine directory."""
    )
    
    parser.add_argument(
        '--keep_percentage', 
        type=int, 
        help='(Required for move_percentage) Percentage of misclassified images to KEEP in the original dataset (0-100).'
    )
    
    parser.add_argument(
        '--dry_run', 
        action='store_true', 
        help='Show what would be done without actually modifying any files.'
    )
    
    return parser.parse_args()

def normalize_path(path):
    """Normalize path for cross-platform compatibility."""
    return os.path.normpath(path)

def main():
    """Main function."""
    setup_logging()
    args = parse_args()
    
    # Validate arguments
    if args.action == 'move_percentage' and args.keep_percentage is None:
        logging.error("Error: Must specify --keep_percentage with the 'move_percentage' action.")
        return
        
    if args.keep_percentage is not None and not 0 <= args.keep_percentage <= 100:
        logging.error("Error: --keep_percentage must be between 0 and 100.")
        return
    
    # Define paths
    quarantine_dir = os.path.join('data', 'odir4', 'misclassified', 'misclassified_images')
    misclassified_file = os.path.join('data', 'odir4', 'misclassified', 'misclassified_images.txt')
    
    # Handle clean_quarantine action
    if args.action == 'clean_quarantine':
        logging.info(f"Action: Clean quarantine directory at '{quarantine_dir}'")
        if not os.path.exists(quarantine_dir):
            logging.info("Quarantine directory does not exist. Nothing to clean.")
            return
        
        files_to_delete = [os.path.join(root, file) 
                          for root, _, files in os.walk(quarantine_dir) 
                          for file in files]
        
        if not files_to_delete:
            logging.info("Quarantine directory is already empty.")
            return
        
        logging.info(f"Found {len(files_to_delete)} files to delete.")
        if args.dry_run:
            logging.info("[Dry Run] Would delete the files listed above.")
        else:
            for file_path in tqdm(files_to_delete, desc="Deleting files"):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}: {e}")
            
            # Clean up empty subdirectories
            for root, dirs, _ in os.walk(quarantine_dir, topdown=False):
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass  # Directory not empty, skip
            
            logging.info("Quarantine directory cleaned.")
        return
    
    # Read misclassified file
    try:
        with open(misclassified_file, 'r') as f:
            misclassified_lines = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        logging.error(f"Error: Misclassified file not found at '{misclassified_file}'")
        return
    
    if not misclassified_lines:
        logging.info("Misclassified file is empty. No images to move.")
        return
    
    # Group images by their actual class
    images_by_class = {}
    for line in misclassified_lines:
        parts = line.split(' | ')
        if len(parts) >= 2:
            actual_class = parts[1].replace('Actual: ', '')
            if actual_class not in images_by_class:
                images_by_class[actual_class] = []
            images_by_class[actual_class].append(parts[0])
    
    # Determine which images to move
    images_to_move = []
    if args.action == 'move_all':
        logging.info("Action: Move all misclassified images.")
        for class_name, paths in images_by_class.items():
            images_to_move.extend(paths)
    
    elif args.action == 'move_percentage':
        logging.info(f"Action: Move images, keeping {args.keep_percentage}% of misclassified images in original dataset.")
        for class_name, paths in images_by_class.items():
            total_count = len(paths)
            num_to_keep = int(total_count * args.keep_percentage / 100)
            
            # Shuffle to ensure random sampling
            random.shuffle(paths)
            
            # The images to keep are the first 'num_to_keep' after shuffling
            paths_to_move = paths[num_to_keep:]
            images_to_move.extend(paths_to_move)
            
            logging.info(f"Class '{class_name}': Total misclassified: {total_count}. "
                        f"Keeping: {num_to_keep}. Moving: {len(paths_to_move)}.")
    
    # Process the list of images to be moved
    if not images_to_move:
        logging.info("No images selected to be moved.")
        return
    
    logging.info(f"Total images to move: {len(images_to_move)}")
    
    if not args.dry_run:
        os.makedirs(quarantine_dir, exist_ok=True)
        logging.info(f"Ensured quarantine directory exists at '{quarantine_dir}'")
    
    missing_files = []
    processed_count = 0
    
    try:
        with tqdm(total=len(images_to_move), desc="Moving images") as pbar:
            for image_path in images_to_move:
                image_path = normalize_path(image_path)
                
                if not os.path.exists(image_path):
                    missing_files.append(image_path)
                    pbar.update(1)
                    continue
                
                # Determine destination
                try:
                    actual_class = os.path.basename(os.path.dirname(image_path))
                    dest_class_dir = os.path.join(quarantine_dir, actual_class)
                    filename = os.path.basename(image_path)
                    dest_path = os.path.join(dest_class_dir, filename)
                except Exception:
                    logging.error(f"Could not determine destination for {image_path}. Skipping.")
                    pbar.update(1)
                    continue
                
                # Perform action
                if args.dry_run:
                    logging.info(f"[Dry Run] Would move '{image_path}' to '{dest_path}'")
                else:
                    try:
                        os.makedirs(dest_class_dir, exist_ok=True)
                        shutil.move(image_path, dest_path)
                    except Exception as e:
                        logging.error(f"Error moving {image_path}: {e}")
                
                processed_count += 1
                pbar.update(1)
                
    except KeyboardInterrupt:
        logging.warning("\nOperation interrupted by user.")
    
    # Final report
    logging.info("--- Operation Summary ---")
    action_verb = "processed"
    if args.dry_run:
        action_verb = "would be processed"
        
    logging.info(f"Total files that {action_verb}: {processed_count}")
    
    if missing_files:
        logging.warning(f"Total missing source files: {len(missing_files)}")
    
    if not args.dry_run and 'move' in args.action:
        logging.info(f"Images have been moved to: {os.path.abspath(quarantine_dir)}")

if __name__ == '__main__':
    main() 