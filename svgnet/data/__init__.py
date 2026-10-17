from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from svgnet.util import worker_init_fn
from .svg3 import SVGDataset
from .archcad_categories import get_categories
__all__ = ["SVGDataset", "build_dataset", "get_categories"]


def build_dataset(data_cfg, logger):
    assert "type" in data_cfg
    _data_cfg = data_cfg.copy()
    _data_cfg["logger"] = logger
    data_type = _data_cfg.pop("type")
    categories = _data_cfg.pop("categories", "floorplancad")
    _data_cfg["bg_id"] = len(get_categories(categories)) - 1
    
    if data_type == "svg":
        return SVGDataset(**_data_cfg)
    else:
        raise ValueError(f"Unknown {data_type}")


def build_dataloader(args,dataset, batch_size=1, num_workers=1, training=True, dist=False):
    shuffle = training
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist else None
    if sampler is not None:
        shuffle = False
    if training:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=partial(worker_init_fn, seed=args.seed)
        )
    else:
        # assert batch_size == 1
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=partial(worker_init_fn, seed=args.seed)
        )
