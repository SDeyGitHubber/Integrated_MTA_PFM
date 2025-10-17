import os

folders = [
    "datasets",
    "models",
    "utils",
    "train",
    "data",
    "checkpoints"
]
files = {
    "datasets/pfm_trajectory_dataset.py": "",
    "models/pfm_model.py": "",
    "utils/collate.py": "",
    "utils/speed_utils.py": "",
    "train/train_pfm.py": "",
    "main.py": "",
    "requirements.txt": "",
    "data/combined_annotations.csv": "",
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file_path, content in files.items():
    with open(file_path, "w") as f:
        f.write(content)

print("VS Code project structure created!")