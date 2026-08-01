import torch
import torch.nn.functional as F

def collate_fn(dataset_items):
    specs = []
    labels = []
    filenames = []

    max_length = max(item["data_object"].shape[-1] for item in dataset_items)

    for item in dataset_items:
        spec = item["data_object"]
        padding = max_length - spec.shape[-1]
        spec = F.pad(spec, (0, padding))
        specs.append(spec)
        labels.append(item["labels"])
        filenames.append(item["filename"])

    return {
        "data_object": torch.stack(specs),
        "labels": torch.tensor(labels),
        "filename": filenames,
    }