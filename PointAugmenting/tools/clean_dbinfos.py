import argparse
import os
import pathlib
import pickle
import numpy as np
from typing import Tuple

"""
清洗 KITTI Radar 数据库：
- 逐条检查 dbinfos 中的 path 指向的 .bin 是否存在、可读取、并且 float32 数量能被 8 或 5 整除；
- 如果是 5 维（老格式），保留该条记录（训练时有 5->8 的兜底填充）；
- 如果文件不存在/损坏/长度为 0 或不能整除，剔除该条记录；
- 输出新的 dbinfos_clean.pkl，并打印各类别保留/剔除统计；
- 可选：将坏样本路径写入 bad_bins.txt。
用法示例：
python tools/clean_dbinfos.py --db-pkl data/kitti_radar/dbinfos_100rate_01sweeps_withvelo_crossmodal.pkl --root data/kitti_radar --out data/kitti_radar/dbinfos_clean.pkl
"""

def is_valid_bin(bin_path: str) -> Tuple[bool, str]:
    try:
        if not os.path.isfile(bin_path):
            return False, "file_not_found"
        size = os.path.getsize(bin_path)
        if size == 0:
            return False, "empty_file"
        # 快速整除检查
        if (size % 4) != 0:
            return False, "not_float32_multiple"
        count = size // 4
        if (count % 8) == 0 or (count % 5) == 0:
            # 进一步尝试映射读取，确认没权限/IO错误
            _ = np.memmap(bin_path, dtype=np.float32, mode='r')
            return True, "ok"
        return False, f"float_count_{count}_not_div_by_5_or_8"
    except Exception as e:
        return False, f"exception:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-pkl', required=True, help='原始 dbinfos.pkl 路径')
    parser.add_argument('--root', required=True, help='数据根目录（与 dbinfos 里的相对 path 拼接）')
    parser.add_argument('--out', required=True, help='清洗后的 dbinfos 输出路径')
    parser.add_argument('--write-bad-list', action='store_true', help='是否输出 bad_bins.txt')
    args = parser.parse_args()

    with open(args.db_pkl, 'rb') as f:
        dbinfos = pickle.load(f)

    kept = {}
    removed = {}
    bad_bins = []

    for cls, items in dbinfos.items():
        kept_list = []
        removed_cnt = 0
        for it in items:
            full_path = str(pathlib.Path(args.root) / it['path'])
            ok, reason = is_valid_bin(full_path)
            if ok:
                kept_list.append(it)
            else:
                removed_cnt += 1
                bad_bins.append(f"{full_path} \t {reason}")
        kept[cls] = kept_list
        removed[cls] = removed_cnt

    # 统计输出
    total_before = sum(len(v) for v in dbinfos.values())
    total_after = sum(len(v) for v in kept.values())
    print("清洗完成：")
    for cls in dbinfos.keys():
        print(f"  {cls}: 保留 {len(kept[cls])} / {len(dbinfos[cls])}，移除 {removed[cls]}")
    print(f"总计：保留 {total_after} / {total_before} 条")

    # 写新 pkl
    with open(args.out, 'wb') as f:
        pickle.dump(kept, f)
    print(f"已写出清洗后的 dbinfos -> {args.out}")

    if args.write_bad_list:
        bad_list_path = os.path.join(os.path.dirname(args.out), 'bad_bins.txt')
        with open(bad_list_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(bad_bins))
        print(f"坏样本列表 -> {bad_list_path}")


if __name__ == '__main__':
    main()
