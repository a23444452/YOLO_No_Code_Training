import yaml
import os

def create_data_yaml(train_path, val_path, class_names_str, output_path="data.yaml"):
    """
    Generates a data.yaml file for YOLO training.
    
    Args:
        train_path (str): Path to training images.
        val_path (str): Path to validation images.
        class_names_str (str): Comma separated class names.
        output_path (str): Where to save the yaml file.
    """
    # Parse class names
    classes = [c.strip() for c in class_names_str.split(',') if c.strip()]
    names_dict = {i: name for i, name in enumerate(classes)}
    
    # If val_path is missing, use train_path (not ideal but works for running)
    if not val_path:
        val_path = train_path

    data = {
        'path': os.path.abspath(os.path.dirname(output_path)), # Base path (optional, but good for relative paths)
        'train': os.path.abspath(train_path),
        'val': os.path.abspath(val_path),
        'names': names_dict,
        'nc': len(classes)
    }

    # Write to yaml
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    return os.path.abspath(output_path)

import shutil
import random
from pathlib import Path

def split_dataset(source_folder, output_folder, split_ratio=0.8, progress_callback=None):
    """
    Splits a raw dataset into YOLO train/val structure.
    
    Args:
        source_folder (str): Folder containing images and txt files.
        output_folder (str): Destination folder.
        split_ratio (float): Ratio of training set (0.0 to 1.0).
        progress_callback (func): Optional callback for logging.
    """
    source = Path(source_folder)
    dest = Path(output_folder)
    
    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    # Create directories
    (dest / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (dest / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (dest / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (dest / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # Get all images
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in source.iterdir() if f.suffix.lower() in valid_exts]
    
    if not images:
        if progress_callback:
            progress_callback("No images found in source folder.")
        return

    # Shuffle
    random.shuffle(images)
    
    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    if progress_callback:
        progress_callback(f"Found {len(images)} images. Split: {len(train_imgs)} Train, {len(val_imgs)} Val.")

    def copy_files(file_list, split_type):
        for img_path in file_list:
            # Copy image
            shutil.copy2(img_path, dest / 'images' / split_type / img_path.name)
            
            # Copy label if exists
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy2(label_path, dest / 'labels' / split_type / label_path.name)
            
            # Check for classes.txt (copy once if found, but usually it's one per dataset)
            # We will handle classes.txt separately
            
    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    
    # Handle classes.txt
    classes_file = source / 'classes.txt'
    if classes_file.exists():
        shutil.copy2(classes_file, dest / 'classes.txt')
        if progress_callback:
            progress_callback("Copied classes.txt")

    if progress_callback:
        progress_callback(f"Dataset preparation complete at {dest}")
