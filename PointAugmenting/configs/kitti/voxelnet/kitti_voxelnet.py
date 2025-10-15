import logging

from det3d.utils.config_tool import get_downsample_factor

# Data root
data_root = "./data/kitti_radar"
train_anno = "/infos_train_kitti_radar.pkl"  # will be created by kitti_data_prep
val_anno = "/infos_val_kitti_radar.pkl"
test_anno = "/infos_test_kitti_radar.pkl"

use_img = True
DOUBLE_FLIP = False

# Radar short-range FOV ~ 0-25m; give a bit margin
pc_range = [-30.0, -30.0, -3.0, 30.0, 30.0, 3.0]
# 提高空间分辨率以利小目标（Cyclist 等），体素缩小
voxel_size = [0.16, 0.16, 0.2]

# 使用统一的8维点特征 [x, y, z, D, P, R, A, E]
num_input_features = 8 + (64 if use_img else 0)

# Classes
tasks = [dict(num_class=3, class_names=["Car", "Cyclist", "Truck"])]
class_names = ["Car", "Cyclist", "Truck"]

model = dict(
    type="VoxelNetFusion" if use_img else "VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=num_input_features,
    ),
    img_backbone=dict(type="DLASeg") if use_img else None,
    backbone=dict(type="SpMiddleResNetFHD", num_input_features=num_input_features, ds_factor=8),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
    dataset='kitti',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)},
        share_conv_channel=64,
        dcn_head=False,
        # 类别权重：放大小样本类别在热力图焦点损失中的权重
        class_weights=[1.0, 2.0, 2.0],  # Car, Cyclist, Truck
        focal_alpha=2.0,
        focal_beta=4.0,
    ),
)

assigner = dict(
    target_assigner=dict(tasks=tasks),
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=200,
    # 减小半径有助于小目标热力图峰更集中
    min_radius=1.5,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-40, -40, -10.0, 40, 40, 10.0],
    nms=dict(
        nms_pre_max_size=1000,   # 稍微放宽预选数量
        nms_post_max_size=200,   # 保留更多候选，利于小目标
        nms_iou_threshold=0.25,  # 适度提升 IoU 阈值，避免小目标被误抑制
    ),
    score_threshold=0.05,       # 适度降低置信度阈值
    pc_range=pc_range[:2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=voxel_size[:2],
    double_flip=DOUBLE_FLIP,
)

# dataset
dataset_type = "KittiDataset"

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    class_names=class_names,
    use_img=use_img,
    # GT-AUG: 数据库采样器
    db_sampler=dict(
        type="DataBaseSamplerV2",
        rate=1.0,
        db_info_path=data_root + "/dbinfos_100rate_01sweeps_withvelo_crossmodal.pkl",
        sample_groups=[
            {"Car": 4, "Cyclist": 16, "Truck": 16},
        ],
        # 每个插入目标随机旋转 ±45°，增强方向多样性
        global_random_rotation_range_per_object=[-0.78539816, 0.78539816],
        # 过滤掉点太少的目标，保证插入样本质量
        db_prep_steps=[
            dict(filter_by_min_num_points={"Car": 5, "Cyclist": 3, "Truck": 5}),
        ],
    ),
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    use_img=use_img,
)

voxel_generator = dict(
    range=pc_range,
    voxel_size=voxel_size,
    max_points_in_voxel=10,
    # 减小体素后适当增大体素数量上限
    max_voxel_num=[90000, 120000],
    double_flip=DOUBLE_FLIP,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, use_img=use_img),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, use_img=use_img),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, use_img=use_img),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, use_img=use_img),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat", double_flip=DOUBLE_FLIP),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + train_anno,
        ann_file=data_root + train_anno,
        class_names=class_names,
        pipeline=train_pipeline,
        use_img=use_img,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + val_anno,
        ann_file=data_root + val_anno,
        class_names=class_names,
        pipeline=test_pipeline,
        use_img=use_img,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + test_anno,
        ann_file=data_root + test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
        use_img=use_img,
        test_mode=True,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False)

lr_config = dict(type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

# 训练轮数上限与早停策略
total_epochs = 50
fade_epoch = 1000  # keep db_sampler on for all epochs; set smaller to disable after N epochs
# Early stopping configs (picked up by Trainer via attributes override if present)
early_stop = dict(patience=5, min_delta=0.0)
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
# 每轮训练结束后跑一次验证
workflow = [('train', 1), ('val', 1)]
