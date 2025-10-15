import os
import argparse
import pickle
import numpy as np

from det3d.core.bbox import box_np_ops


def compute_alpha(ry, x, z):
    """Compute observation angle alpha from rotation_y and position in camera frame.
    alpha = ry - arctan2(x, z)
    """
    return float(ry - np.arctan2(x, z))


def format_line(cls_name,
                truncation,
                occlusion,
                alpha,
                bbox2d,
                dims_lhw,
                center_xyz,
                rotation_y,
                score):
    # bbox2d: [xmin, ymin, xmax, ymax] in pixels
    # dims_lhw: [l, h, w] in meters (KITTI camera frame order)
    # center_xyz: [x, y, z] camera frame bottom-centered
    fields = [
        cls_name,
        f"{truncation:.2f}",
        str(int(occlusion)),
        f"{alpha:.6f}",
        f"{bbox2d[0]:.2f}", f"{bbox2d[1]:.2f}", f"{bbox2d[2]:.2f}", f"{bbox2d[3]:.2f}",
        f"{dims_lhw[0]:.6f}", f"{dims_lhw[1]:.6f}", f"{dims_lhw[2]:.6f}",
        f"{center_xyz[0]:.6f}", f"{center_xyz[1]:.6f}", f"{center_xyz[2]:.6f}",
        f"{rotation_y:.6f}",
        f"{score:.6f}",
    ]
    return " ".join(fields)


def export(prediction_pkl,
           infos_pkl,
           out_dir,
           class_names=("Car", "Cyclist", "Truck")):
    os.makedirs(out_dir, exist_ok=True)

    with open(prediction_pkl, "rb") as f:
        predictions = pickle.load(f)

    with open(infos_pkl, "rb") as f:
        infos = pickle.load(f)

    # token -> calib + file stem
    token_map = {}
    for info in infos:
        token = info["token"]
        calib = info["calib"]
        # filename stem like 000001 from image path or lidar path
        if "image" in info and os.path.exists(info["image"]):
            stem = os.path.splitext(os.path.basename(info["image"]))[0]
        elif "lidar_path" in info:
            stem = os.path.splitext(os.path.basename(info["lidar_path"]))[0]
        else:
            stem = token
        token_map[token] = (calib, stem)

    for token, det in predictions.items():
        if token not in token_map:
            continue
        calib, stem = token_map[token]
        rect = np.array(calib["R0_rect"], dtype=np.float32)
        Trv2c = np.array(calib["Tr_velo_to_cam"], dtype=np.float32)
        P2 = np.array(calib.get("P0", calib.get("P2", np.eye(3))), dtype=np.float32)

        boxes_lidar = det["box3d_lidar"].cpu().numpy().astype(np.float32)
        scores = det["scores"].cpu().numpy().astype(np.float32)
        labels = det["label_preds"].cpu().numpy().astype(np.int32)

        # Convert to camera frame label convention
        # lidar [x,y,z,w,l,h,yaw] -> camera [x,y,z,l,h,w,ry]
        boxes_cam = box_np_ops.box_lidar_to_camera(boxes_lidar, rect, Trv2c)
        centers = boxes_cam[:, :3]
        dims_lhw = boxes_cam[:, 3:6]
        ry = boxes_cam[:, 6]

        # bottom-centered: our conversion already outputs camera bottom-centered (l,h,w order)
        # 2D boxes from 3D corners
        bbox2d = box_np_ops.box3d_to_bbox(boxes_lidar, rect, Trv2c, P2)  # uses lidar->cam->image internally

        lines = []
        for i in range(boxes_lidar.shape[0]):
            cls_idx = int(labels[i])
            if cls_idx < 0 or cls_idx >= len(class_names):
                cls_name = "DontCare"
            else:
                cls_name = class_names[cls_idx]
            # truncate/occlusion unknown in test -> fill 0
            trunc = 0.0
            occ = 0
            a = compute_alpha(ry[i], centers[i, 0], centers[i, 2]) if np.isfinite(ry[i]) else 0.0
            b2d = bbox2d[i]
            d_lhw = dims_lhw[i]
            c_xyz = centers[i]
            r_y = float(ry[i]) if np.isfinite(ry[i]) else 0.0
            sc = float(scores[i]) if np.isfinite(scores[i]) else 0.0
            # clamp bbox to positive values if needed
            if not np.all(np.isfinite(b2d)):
                b2d = np.zeros(4, dtype=np.float32)
            line = format_line(cls_name, trunc, occ, a, b2d, d_lhw, c_xyz, r_y, sc)
            lines.append(line)

        out_path = os.path.join(out_dir, f"{stem}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            if len(lines) > 0:
                f.write("\n".join(lines))
            else:
                f.write("")

    print(f"Exported KITTI label_2 style results to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="prediction.pkl path produced by dist_test.py")
    parser.add_argument("--infos", required=True, help="infos_test_kitti_radar.pkl path")
    parser.add_argument("--out", required=True, help="output folder for txt files (label_2 style)")
    args = parser.parse_args()
    export(args.pred, args.infos, args.out)


if __name__ == "__main__":
    main()
