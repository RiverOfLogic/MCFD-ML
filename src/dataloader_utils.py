import torch

import config


def loader_kwargs(drop_last=False):
    num_workers = int(getattr(config, "num_workers", 0))
    pin_memory = bool(getattr(config, "pin_memory", torch.cuda.is_available()))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(config, "persistent_workers", True))
        kwargs["prefetch_factor"] = int(getattr(config, "prefetch_factor", 2))
    return kwargs
