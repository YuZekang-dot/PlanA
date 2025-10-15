import os
import numpy as np
import glob

def count_points_in_bin(file_path):
    """计算单个.bin文件中的点云数量"""
    # 每个点由8个float32值组成(x,y,z,D,P,R,A,E)
    # 每个float32占用4字节，因此每个点占用32字节
    try:
        file_size = os.path.getsize(file_path)
        # 计算点的数量：总字节数 / 每个点的字节数(8*4)
        point_count = file_size // 32
        return point_count
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def calculate_statistics(point_counts):
    """计算点云数量的统计信息"""
    if not point_counts:
        return None
    
    # 转换为numpy数组以便计算
    counts_array = np.array(point_counts)
    
    stats = {
        'min': np.min(counts_array),
        'max': np.max(counts_array),
        'mean': np.mean(counts_array),
        'median': np.median(counts_array),
        'total_files': len(counts_array),
        'total_points': np.sum(counts_array)
    }
    
    return stats

def main():
    # 点云文件目录
    bin_directory = r"PointAugmenting\data\kitti_radar\training\velodyne"
    
    # 检查目录是否存在
    if not os.path.exists(bin_directory):
        print(f"错误: 目录不存在 - {bin_directory}")
        return
    
    # 获取所有.bin文件
    bin_files = glob.glob(os.path.join(bin_directory, "*.bin"))
    
    if not bin_files:
        print(f"警告: 在 {bin_directory} 中未找到任何.bin文件")
        return
    
    print(f"找到 {len(bin_files)} 个.bin点云文件，正在处理...")
    
    # 统计每个文件的点数量
    point_counts = []
    for i, file_path in enumerate(bin_files, 1):
        # 显示进度
        if i % 10 == 0 or i == len(bin_files):
            print(f"已处理 {i}/{len(bin_files)} 个文件")
        
        count = count_points_in_bin(file_path)
        if count is not None:
            point_counts.append(count)
    
    # 计算统计信息
    stats = calculate_statistics(point_counts)
    
    if not stats:
        print("无法计算统计信息，没有有效的点云数据")
        return
    
    # 输出结果
    print("\n点云文件统计结果:")
    print("----------------------------------------")
    print(f"文件总数: {stats['total_files']}")
    print(f"总点云数量: {stats['total_points']:,}")
    print(f"最小点云数量: {stats['min']}")
    print(f"最大点云数量: {stats['max']}")
    print(f"平均点云数量: {stats['mean']:.2f}")
    print(f"中位数点云数量: {stats['median']}")
    print("----------------------------------------")

if __name__ == "__main__":
    main()
    