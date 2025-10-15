import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os 
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=None, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_'+path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1]+'.npy')
        points =  np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] # remove ring_index from features 
    else:
        raw = np.fromfile(path, dtype=np.float32)
        # 统一返回8维点特征；若为旧版5维，则零填充到8维
        if raw.size % 8 == 0:
            pts8 = raw.reshape(-1, 8)
        elif raw.size % 5 == 0:
            base5 = raw.reshape(-1, 5)
            pts8 = np.zeros((base5.shape[0], 8), dtype=np.float32)
            pts8[:, :5] = base5
        else:
            # 既不是5维也不是8维，抛出明确错误，便于清理数据
            raise ValueError(f"Unexpected point feature width in {path}: total floats {raw.size} not divisible by 5 or 8")
        # 默认返回全部可用维度；如指定 num_point_feature 则裁剪
        points = pts8[:, :num_point_feature] if num_point_feature is not None else pts8

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.use_img = kwargs.get("use_img", False)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":
            nsweeps = res["lidar"]["nsweeps"]
            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(info["sweeps"]), \
                "nsweeps {} should equal to list length {}.".format(nsweeps, len(info["sweeps"]))

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])  # use_seg NotImplemented
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times

            combined = np.hstack([points, times])

            if self.use_img:
                cam_name = res['camera']['name']
                im_shape = (448 * 2, 1600, 3)
                pts_uv_all = np.ones([points.shape[0], 3]).astype(np.float32) * -100

                for cam_id, cam_sensor in enumerate(cam_name):
                    cam_from_global = info["cams_from_global"][cam_sensor]
                    cam_intrinsic = info["cam_intrinsics"][cam_sensor]

                    # lidar to global
                    ref_to_global = info["ref_to_global"]
                    pts_hom = np.concatenate([points[:, :3], np.ones([points.shape[0], 1])], axis=1)
                    pts_global = ref_to_global.dot(pts_hom.T)  # 4 * N

                    # global to cam
                    pts_cam = cam_from_global.dot(pts_global)[:3, :]  # 3 * N

                    # cam to uv
                    pts_uv = view_points(pts_cam, np.array(cam_intrinsic), normalize=True).T  # N * 3

                    # Remove points that are either outside or behind the camera.
                    mask = (pts_cam[2, :] > 0) & (pts_uv[:, 0] > 1) & (pts_uv[:, 0] < im_shape[1] - 1) & \
                           (pts_uv[:, 1] > 1) & (pts_uv[:, 1] < im_shape[0] - 1)

                    pts_uv_all[mask, :2] = pts_uv[mask, :2]
                    pts_uv_all[mask, 2] = float(cam_id)

                # normalization to [-1, 1]
                pts_uv_all[..., 0] = pts_uv_all[..., 0] / (im_shape[1] - 1) * 2 - 1
                pts_uv_all[..., 1] = pts_uv_all[..., 1] / (im_shape[0] - 1) * 2 - 1
                pts_uv_all[..., 2] = pts_uv_all[..., 2] / (6 - 1) * 2 - 1
                res["metadata"]["num_point_features"] += 3
                combined = np.concatenate([combined, pts_uv_all], axis=1).astype(np.float32)

            res["lidar"]["combined"] = combined

        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1:
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(info["sweeps"]), \
                    "nsweeps {} should be equal to the list length {}.".format(nsweeps, len(info["sweeps"]))

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])

            if self.use_img:
                NotImplementedError

        elif self.type == "KittiDataset":
            # KITTI-radar single sweep, keep 8-dim points [x,y,z,D,P,R,A,E]
            lidar_path = Path(info["lidar_path"]).as_posix()
            points = read_file(lidar_path, num_point_feature=None)
            res["lidar"]["points"] = points

            combined = points
            if self.use_img:
                # project to single camera using calib matrices
                calib = info["calib"]
                P = np.array(calib["P0"], dtype=np.float32)
                R0 = np.eye(4, dtype=np.float32)
                R0[:3, :3] = np.array(calib.get("R0_rect", np.eye(3)), dtype=np.float32)
                Tr = np.eye(4, dtype=np.float32)
                Tr[:3, :4] = np.array(calib["Tr_velo_to_cam"], dtype=np.float32)

                pts = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
                pts_cam = (R0 @ (Tr @ pts.T))[:3, :]
                uvw = P @ np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]), dtype=np.float32)])
                uv = (uvw[:2, :] / np.maximum(uvw[2:3, :], 1e-6)).T
                im_w, im_h = 640, 480
                pts_uv_all = np.ones([points.shape[0], 3], dtype=np.float32) * -100
                mask = (pts_cam[2, :] > 0) & (uv[:, 0] > 1) & (uv[:, 0] < im_w - 1) & (uv[:, 1] > 1) & (uv[:, 1] < im_h - 1)
                pts_uv_all[mask, 0] = uv[mask, 0] / (im_w - 1) * 2 - 1
                pts_uv_all[mask, 1] = uv[mask, 1] / (im_h - 1) * 2 - 1
                pts_uv_all[mask, 2] = -1  # single camera id normalized
                res["metadata"]["num_point_features"] += 3
                combined = np.concatenate([points, pts_uv_all], axis=1).astype(np.float32)
            res["lidar"]["combined"] = combined

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, use_img=False, **kwargs):
        self.use_img = use_img

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
                "frustums": info["gt_frustum"]
            }

            if self.use_img:
                res["camera"]["annotations"] = {
                    "avail_2d": info["avail_2d"].astype(np.bool),
                    "boxes_2d": info["boxes_2d"].astype(np.int32),
                    "depths": info["depths"].astype(np.float32),
                }

        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
            if self.use_img:
                NotImplementedError
        else:
            pass 

        return res, info
