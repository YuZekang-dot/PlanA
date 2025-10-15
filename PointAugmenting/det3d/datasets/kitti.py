"""
Compatibility shim to avoid import collision between module file `det3d.datasets.kitti` and
the package directory `det3d/datasets/kitti/` containing `kitti_common.py`.

This file re-exports `KittiDataset` from `kitti_dataset.py` and sets `__path__` so that
`from det3d.datasets.kitti import kitti_common` works as expected.
"""
from pathlib import Path as _Path

# Re-export the dataset class from the new module name
from .kitti_dataset import KittiDataset  # noqa: F401

# Treat this module as a package by defining __path__ pointing to the directory
# where `kitti_common.py` lives. This allows submodule imports under
# `det3d.datasets.kitti` to resolve (e.g., `det3d.datasets.kitti.kitti_common`).
__path__ = [str(_Path(__file__).with_name("kitti"))]

__all__ = ["KittiDataset"]
