import os
import pickle
from pathlib import Path
import numpy as np

from tqdm import tqdm

from det3d.utils.utils_kitti import KittiDB

KITTI_CLASSES = ["Car", "Cyclist", "Truck"]


def _read_imageset(root_path: str, split: str):
    image_sets_dir = Path(root_path) / "ImageSets"
    txt = image_sets_dir / f"{split}.txt"
    assert txt.exists(), f"{txt} not found."
    with open(txt, "r") as f:
        ids = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return ids


def _read_calib(calib_file: str):
    """Parse minimal calib fields from KITTI-like calib file.
    Only keep P0 (K), Tr_velo_to_cam. r0_rect can be identity in this dataset.
    """
    lines = [l.strip() for l in open(calib_file).readlines() if len(l.strip()) > 0]
    calib = {}
    for l in lines:
        if l.startswith("P0:") or l.startswith("P2:"):
            vals = np.array(l.split()[1:], dtype=np.float32)
            calib["P0"] = vals.reshape(3, 4)
        elif l.startswith("R0_rect:"):
            vals = np.array(l.split()[1:], dtype=np.float32)
            calib["R0_rect"] = vals.reshape(3, 3)
        elif l.startswith("Tr_velo_to_cam:"):
            vals = np.array(l.split()[1:], dtype=np.float32)
            calib["Tr_velo_to_cam"] = vals.reshape(3, 4)
    # Fallbacks
    if "R0_rect" not in calib:
        calib["R0_rect"] = np.eye(3, dtype=np.float32)
    if "P0" not in calib:
        raise RuntimeError("P0 (camera intrinsics) not found in calib file.")
    return calib


def _read_label(label_file: str):
    annos = []
    if not os.path.exists(label_file):
        return annos
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(" ")
            if len(parts) < 15:
                continue
            name = parts[0]
            # Map custom classes Car/Cyclist/Truck; others ignored
            if name not in ["Car", "Cyclist", "Truck"]:
                continue
            truncated = float(parts[1])
            occluded = int(float(parts[2]))
            alpha = float(parts[3])
            bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
            h = float(parts[8]); w = float(parts[9]); l = float(parts[10])
            x = float(parts[11]); y = float(parts[12]); z = float(parts[13])
            ry = float(parts[14])
            annos.append({
                "name": name,
                "bbox": bbox,
                "dimensions": [l, w, h],
                "location": [x, y, z],
                "rotation_y": ry,
                "truncated": truncated,
                "occluded": occluded,
                "alpha": alpha,
            })
    return annos


def _build_info(root_path: str, split: str):
    root = Path(root_path)
    is_test = split == "test"
    ids = _read_imageset(root_path, split)

    infos = []
    for idx in tqdm(ids, desc=f"Building KITTI-radar {split} infos"):
        if is_test:
            img_path = root / "testing" / "image_2" / f"{idx}.png"
            lidar_path = root / "testing" / "velodyne" / f"{idx}.bin"
            calib_path = root / "testing" / "calib" / f"{idx}.txt"
            annos = None
        else:
            img_path = root / "training" / "image_2" / f"{idx}.png"
            lidar_path = root / "training" / "velodyne" / f"{idx}.bin"
            calib_path = root / "training" / "calib" / f"{idx}.txt"
            label_path = root / "training" / "label_2" / f"{idx}.txt"
            annos = _read_label(str(label_path))
        calib = _read_calib(str(calib_path))

        # only keep 5-dim points [x,y,z,D,P], remove redundant R,A,E if any
        # actual cropping will be handled in pipeline reader; here we store paths only
        info = {
            "image": str(img_path),
            "lidar_path": str(lidar_path),
            "token": idx,
            "calib": calib,
        }
        if annos is not None:
            # convert to arrays for training
            names = []
            gt_boxes = []
            boxes2d = []
            depths = []
            for a in annos:
                names.append(a["name"])  # Car/Cyclist/Truck
                # KITTI label uses camera frame bottom-centered; our model expects lidar frame center-based in meters.
                # Here keep camera-format info; conversion done in dataset pipeline using calib per sample.
                l, w, h = a["dimensions"]
                x, y, z = a["location"]
                ry = a["rotation_y"]
                gt_boxes.append([x, y, z, l, w, h, ry])
                boxes2d.append(a["bbox"])
                depths.append(z)
            if len(gt_boxes) == 0:
                gt_boxes = np.zeros((0, 7), dtype=np.float32)
                boxes2d = np.zeros((0, 4), dtype=np.float32)
                depths = np.zeros((0,), dtype=np.float32)
            else:
                gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                boxes2d = np.asarray(boxes2d, dtype=np.float32)
                depths = np.asarray(depths, dtype=np.float32)
            info.update({
                "gt_boxes_camera": gt_boxes,  # camera frame, to be converted later
                "gt_names": np.array(names),
                "boxes_2d": boxes2d,
                "depths": depths,
            })
        infos.append(info)
    return infos


def create_kitti_infos(root_path: str, split: str):
    infos = _build_info(root_path, split)
    out = Path(root_path) / f"infos_{split}_kitti_radar.pkl"
    with open(out, "wb") as f:
        pickle.dump(infos, f)
    print(f"Saved {len(infos)} infos to {out}")
    return str(out)
