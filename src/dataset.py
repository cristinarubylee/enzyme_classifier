from datasets import load_from_disk

def load_datasets(train_path: str, val_path: str):
    train_ds = load_from_disk(train_path).with_format("torch")
    val_ds = load_from_disk(val_path).with_format("torch")
    return train_ds, val_ds
