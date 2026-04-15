# src/data/loaders.py

import json
import subprocess
import torch
from pathlib import Path
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent

# LEAF FEMNIST dataset: https://github.com/TalwalkarLab/leaf
FEMNIST_ROOT_DIR    = ROOT / "src" / "data" / "leaf" / "data" / "femnist"
FEMNIST_TRAIN_DIR   = ROOT / "src" / "data" / "leaf" / "data" / "femnist" / "data" / "train"
FEMNIST_TEST_DIR    = ROOT / "src" / "data" / "leaf" / "data" / "femnist" / "data" / "test"
FEMNIST_CLIENTS_DIR = ROOT / "src" / "data" / "clients" / "femnist"

def process_leaf_femnist():
    print("Processing FEMNIST...")
    FEMNIST_CLIENTS_DIR.mkdir(parents=True, exist_ok=True)

    preprocess_cmd = [
        "./preprocess.sh",
        "-s", "niid",         # Sample non-i.i.d. data
        "--sf", "1.0",        # Sample 100% of data
        "-k", "0",            # Minimum sample size per client
        "-t", "sample",       # Partition each clients' data into train-test groups
        "--smplseed", "2026", # Seed for random sampling of data
        "--spltseed", "2026"  # Seed for random splitting of data
    ]

    try:
        subprocess.run(preprocess_cmd, cwd=FEMNIST_ROOT_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute '{FEMNIST_ROOT_DIR}.preprocess.sh': {e}")
        return # Exit upon fail

    # Grab all JSON outputs from subprocess
    train_files = sorted(FEMNIST_TRAIN_DIR.glob('*.json'))
    test_files  = sorted(FEMNIST_TEST_DIR.glob('*.json'))

    client_train_data = {}
    global_val_x, global_val_y = [], []

    # Parse each clients' training data
    for file in train_files:
        with file.open('r') as f:
            data = json.load(f)

            for client_id, client_data in data['user_data'].items():
                # Reshape to (N, 1, 28, 28)
                x = torch.tensor(client_data['x'], dtype=torch.float32).view(-1, 1, 28, 28)
                y = torch.tensor(client_data['y'], dtype=torch.long)
                client_train_data[client_id] = TensorDataset(x, y)

    selected_clients = set(client_train_data.keys())

    # Parse each clients' testing data
    # and combine all testing data into global validation set
    for file in test_files:
        with file.open('r') as f:
            data = json.load(f)

            for client_id, client_data in data['user_data'].items():
                if client_id in selected_clients:
                    global_val_x.extend(client_data['x'])
                    global_val_y.extend(client_data['y'])
            
    val_x_tensor = torch.tensor(global_val_x, dtype=torch.float32).view(-1, 1, 28, 28)
    val_y_tensor = torch.tensor(global_val_y, dtype=torch.long)
    global_val_dataset = TensorDataset(val_x_tensor, val_y_tensor)

    # Save to disk
    print(f"Saving {len(client_train_data)} client training files...")
    for i, (client_id, dataset) in enumerate(client_train_data.items()):
        torch.save(dataset, FEMNIST_CLIENTS_DIR / f"client_{i}.pt")

    print(f"Saving server-side validation dataset with {len(global_val_dataset)} samples...")
    torch.save(global_val_dataset, FEMNIST_CLIENTS_DIR / "global_val.pt")
    print("FEMNIST PROCESSING COMPLETE")


if __name__ == "__main__":
    process_leaf_femnist()