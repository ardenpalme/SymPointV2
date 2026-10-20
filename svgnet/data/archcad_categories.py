"""Category lists by dataset name, for svgnet.data.build_dataset and tools/{train,test}.py."""

from .svg3 import SVG_CATEGORIES


def get_categories(name):
    if name == "floorplancad":
        return SVG_CATEGORIES
    raise ValueError(f"unknown categories {name!r}: only 'floorplancad' is supported (ArchCAD list unverified)")
