# src/data/loaders.py

import json
import subprocess
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent

# LEAF FEMNIST dataset
FEMNIST_ROOT_DIR    = ROOT / "src" / "data" / "leaf" / "data" / "femnist"
FEMNIST_TRAIN_DIR   = ROOT / "src" / "data" / "leaf" / "data" / "femnist" / "data" / "train"
FEMNIST_TEST_DIR    = ROOT / "src" / "data" / "leaf" / "data" / "femnist" / "data" / "test"
FEMNIST_CLIENTS_DIR = ROOT / "src" / "data" / "clients" / "femnist"

# CIFAR-10 dataset
CIFAR10_CLIENTS_DIR = ROOT / "src" / "data" / "clients" / "cifar10"
CIFAR10_DOWNLOAD_DIR = ROOT / "src" / "data" / "cifar10_raw"

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

def process_cifar10(alpha=0.3, num_clients=100, seed=2026):
    CIFAR10_CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    train_ds = torchvision.datasets.CIFAR10(root=CIFAR10_DOWNLOAD_DIR, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(root=CIFAR10_DOWNLOAD_DIR, train=False, download=True, transform=transform)

    targets = np.array(train_ds.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        counts = (proportions * len(class_idx)).astype(int)
        counts[-1] = len(class_idx) - counts[:-1].sum()
        splits = np.split(class_idx, np.cumsum(counts)[:-1])

        for k, idx in enumerate(splits):
            client_indices[k].extend(idx.tolist())

    all_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    all_y = torch.tensor(train_ds.targets, dtype=torch.long)

    sizes = []

    for i, idx in enumerate(client_indices):
        idx_t = torch.tensor(idx, dtype=torch.long)
        ds = TensorDataset(all_x[idx_t], all_y[idx_t])
        torch.save(ds, CIFAR10_CLIENTS_DIR / f"client_{i}.pt")
        sizes.append(len(idx))

    val_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    val_y = torch.tensor(test_ds.targets, dtype=torch.long)
    torch.save(TensorDataset(val_x, val_y), CIFAR10_CLIENTS_DIR / "global_val.pt")

    print(f"CIFAR-10 partitioned: {num_clients} clients, alpha={alpha}, seed={seed}")
    print(f"Client sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")


if __name__ == "__main__":
    process_leaf_femnist()
    process_cifar10()