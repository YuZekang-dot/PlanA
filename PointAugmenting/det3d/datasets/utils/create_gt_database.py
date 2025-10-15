import pickle
from pathlib import Path
import os 
import numpy as np

from det3d.core import box_np_ops
from det3d.datasets.dataset_factory import get_dataset
from tqdm import tqdm

dataset_name_map = {
    "NUSC": "NuScenesDataset",
    "WAYMO": "WaymoDataset",
    "KITTI": "KittiDataset",
}


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_path=None,
    rate=1.,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    **kwargs,
):
    pipeline = [
        {
            "type": "LoadPointCloudFromFile",
            "dataset": dataset_name_map[dataset_class_name],
            "use_img": True
        },
        {"type": "LoadPointCloudAnnotations", "with_bbox": True, "use_img": True},
    ]

    if "nsweeps" in kwargs:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path,
            root_path=data_path,
            pipeline=pipeline,
            test_mode=True,
            use_img=True,
            nsweeps=kwargs["nsweeps"],
        )
        nsweeps = dataset.nsweeps
    else:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline
        )
        nsweeps = 1

    root_path = Path(data_path)

    if dataset_class_name in ["WAYMO", "NUSC", "KITTI"]: 
        if db_path is None:
            suffix = "{}sweeps_withvelo_crossmodal".format(f"{nsweeps:02d}") if dataset_class_name != "KITTI" else "01sweeps_withvelo_crossmodal"
            db_path = root_path / "gt_database_{:03d}rate_{}".format(int(rate*100), suffix)
        if dbinfo_path is None:
            suffix = "{}sweeps_withvelo_crossmodal".format(f"{nsweeps:02d}") if dataset_class_name != "KITTI" else "01sweeps_withvelo_crossmodal"
            dbinfo_path = root_path / "dbinfos_{:03d}rate_{}.pkl".format(int(rate*100), suffix)
    else:
        raise NotImplementedError()

    if dataset_class_name == "NUSC":
        point_features = 5 + 3
    elif dataset_class_name == "WAYMO":
        point_features = 5 if nsweeps == 1 else 6 
    elif dataset_class_name == "KITTI":
        point_features = 5 + 3  # [x,y,z,D,P,R,A,E]
    else:
        raise NotImplementedError()

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    cam_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'] if dataset_class_name != 'KITTI' else ['CAM_FRONT']

    for index in tqdm(range(len(dataset))):
        image_idx = index
        # 获取一条样本数据（各数据集 get_sensor_data 返回结构一致）
        sensor_data = dataset.get_sensor_data(index)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        if dataset_class_name == 'KITTI':
            points = sensor_data["lidar"]["combined"]
        elif nsweeps > 1: 
            points = sensor_data["lidar"]["combined"]
        else:
            points = sensor_data["lidar"]["points"]
            
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]
        gt_frustums = annos.get("frustums", np.zeros((gt_boxes.shape[0], 8, 3), dtype=np.float32))

        # For KITTI, boxes in annos are camera-frame by default. Convert to LiDAR frame here
        # to ensure points_in_rbbox uses correct coordinates, consistent with training pipeline.
        if dataset_class_name == 'KITTI' and gt_boxes.shape[-1] >= 7:
            calib = sensor_data.get("calib", {})
            R0 = np.eye(4, dtype=np.float32)
            if "R0_rect" in calib:
                R0[:3, :3] = np.array(calib["R0_rect"], dtype=np.float32)
            Tr = np.eye(4, dtype=np.float32)
            if "Tr_velo_to_cam" in calib:
                Tr[:3, :4] = np.array(calib["Tr_velo_to_cam"], dtype=np.float32)

            Rt = np.linalg.inv(Tr) @ np.linalg.inv(R0)
            R_only = Rt[:3, :3]

            centers_cam = gt_boxes[:, :3].astype(np.float32)
            dims_lwh = gt_boxes[:, 3:6].astype(np.float32)
            ry = gt_boxes[:, 6].astype(np.float32)
            # convert bottom center to true center in camera frame (y downwards)
            centers_cam_true = centers_cam.copy()
            centers_cam_true[:, 1] -= dims_lwh[:, 2] / 2.0
            centers_cam_h = np.concatenate([centers_cam_true, np.ones((centers_cam_true.shape[0], 1), dtype=np.float32)], axis=1)
            centers_velo = (Rt @ centers_cam_h.T).T[:, :3]

            dir_cam = np.stack([np.sin(ry), np.zeros_like(ry), np.cos(ry)], axis=1)
            dir_velo = (R_only @ dir_cam.T).T
            yaw_lidar = np.arctan2(dir_velo[:, 1], dir_velo[:, 0])

            # dims to [w, l, h] from camera [l, w, h]
            w = dims_lwh[:, 1]
            l = dims_lwh[:, 0]
            h = dims_lwh[:, 2]

            gt_boxes = np.stack([centers_velo[:, 0], centers_velo[:, 1], centers_velo[:, 2], w, l, h, yaw_lidar], axis=1).astype(np.float32)
            # pad vx,vy zeros and place rot at end to be compatible with downstream code
            zeros = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)
            gt_boxes = np.concatenate([gt_boxes[:, :6], zeros, gt_boxes[:, 6:7]], axis=1).astype(np.float32)

        if dataset_class_name == 'WAYMO':
            # waymo dataset contains millions of objects and it is not possible to store
            # all of them into a single folder
            # we randomly sample a few objects for gt augmentation
            # We keep all cyclist as they are rare 
            if index % 4 != 0:
                mask = (names == 'VEHICLE') 
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]

            if index % 2 != 0:
                mask = (names == 'PEDESTRIAN')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]


        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        annos_img = sensor_data.get("camera", {}).get("annotations", None)
        if annos_img is not None:
            avail_2d = annos_img["avail_2d"]
            boxes_2d = annos_img["boxes_2d"]  # N * C * 4
            depths = annos_img["depths"]
        else:
            avail_2d = np.zeros((gt_boxes.shape[0], len(cam_name)), dtype=np.bool_)
            boxes_2d = np.zeros((gt_boxes.shape[0], len(cam_name), 4), dtype=np.int32)
            depths = np.zeros((gt_boxes.shape[0], len(cam_name)), dtype=np.float32)
        import cv2
        imgs = []
        for cam in cam_name:
            path = sensor_data["camera"]["cam_paths"].get(cam, None) if "camera" in sensor_data else None
            imgs.append(cv2.imread(path) if path is not None and os.path.exists(path) else None)

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue 
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)

                # save image patches
                cam_paths = [''] * len(cam_name)
                for cam_id, flag in enumerate(avail_2d[i]):
                    if flag and imgs[cam_id] is not None:
                        cur_box = boxes_2d[i, cam_id]
                        x1, y1, x2, y2 = cur_box.astype(int)
                        if x2 > x1 and y2 > y1:
                            patch = imgs[cam_id][y1:y2, x1:x2, :]
                            filename = '{}_{}_{}_{}.jpg'.format(image_idx, names[i], i, cam_id)
                            filepath = os.path.join(str(db_path), names[i], filename)
                            cam_paths[cam_id] = filepath
                            try:
                                cv2.imwrite(filepath, patch)
                            except Exception:
                                pass

                # save pts
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                filepath = os.path.join(str(db_path), names[i], filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "wb") as f:
                    try:
                        gt_points[:, :point_features].tofile(f)
                    except:
                        print("process {} files".format(index))
                        break

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = os.path.join(db_path.stem, names[i], filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    # save LiDAR-frame box
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    "frustum": gt_frustums[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }

                db_info.update({
                    "avail_2d": avail_2d[i],
                    "bbox": boxes_2d[i],
                    "depth": depths[i],
                    "cam_paths": np.array(cam_paths)
                })

                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)
