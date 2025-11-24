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
