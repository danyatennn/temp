import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate dataset items into a batch.

    Stacks the measurement, PSF, and (optional) ground-truth tensors and
    collects the sample ids into a list.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.
    Returns:
        result_batch (dict): batch with stacked tensors and a list of ids.
    """
    result_batch = {
        "measurement": torch.stack([item["measurement"] for item in dataset_items]),
        "psf": torch.stack([item["psf"] for item in dataset_items]),
        "id": [item["id"] for item in dataset_items],
    }

    if "gt" in dataset_items[0]:
        result_batch["gt"] = torch.stack([item["gt"] for item in dataset_items])

    return result_batch
