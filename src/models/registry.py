# src/models/registry.py

from models.cnn import FEMNIST_CNN

_REGISTRY = {
    'femnist': FEMNIST_CNN,
}

def get_model(dataset: str):
    if dataset not in _REGISTRY:
        raise ValueError(f"No model registered for dataset '{dataset}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[dataset]
