import os
import sys
import shutil
import random
import argparse

def create_dirs(dest_dir):
    """
    Create the ml_data directory structure in the specified destination directory.
    """
    dirs = [
        os.path.join(dest_dir, "images", "train"),
        os.path.join(dest_dir, "images", "val"),
        os.path.join(dest_dir, "labels", "train"),
        os.path.join(dest_dir, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

def split_and_copy_files(train_ratio, image_dir, label_dir, dest_dir):
    """
    Split files from image_dir into train and validation sets based on train_ratio,
    and copy each image along with its corresponding label (matched by base name) 
    to the appropriate destination directory.
    """
    # List all files in the images directory
    image_files = [f for f in os.listdir(image_dir)
                   if os.path.isfile(os.path.join(image_dir, f))]
    
    # Shuffle the files to randomize the split
    random.shuffle(image_files)
    n_train = int(len(image_files) * train_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    
    def copy_files(file_list, subset):
        for f in file_list:
            # Copy image file
            src_image = os.path.join(image_dir, f)
            dst_image = os.path.join(dest_dir, "images", subset, f)
            shutil.copy(src_image, dst_image)
            
            # Match label file by base name (e.g., DWI_001.JPG -> DWI_001.txt)
            base, _ = os.path.splitext(f)
            src_label = os.path.join(label_dir, base + ".txt")
            dst_label = os.path.join(dest_dir, "labels", subset, base + ".txt")
            
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                print(f"Warning: Label file not found for {f}")
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    
    print(f"Total image files: {len(image_files)}")
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split images and labels into train and validation directories."
    )
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of files to use for training (default: 0.8)")
    args = parser.parse_args()
    
    # Determine directory paths relative to the script location.
    # Assuming the following project structure:
    # project/
    # ├── images/
    # ├── labels/
    # ├── ml_data/         <- Will be created here
    # └── src/
    #     └── split_data.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")
    dest_dir = os.path.join(root_dir, "ml_data")
    
    # Clean the destination directory if it exists.
    if os.path.exists(dest_dir):
        print(f"Cleaning destination directory {dest_dir}...")
        shutil.rmtree(dest_dir)
    
    # Create destination directory structure
    create_dirs(dest_dir)
    
    # Split and copy files
    split_and_copy_files(args.train_ratio, image_dir, label_dir, dest_dir)