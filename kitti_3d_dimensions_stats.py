import os

def calculate_average_dimensions(label_folder):
    # 初始化字典存储每个类别的尺寸总和和计数
    # 键为类别名，值为一个字典包含总和和计数
    categories = {
        'Car': {'height': 0.0, 'width': 0.0, 'length': 0.0, 'count': 0},
        'Truck': {'height': 0.0, 'width': 0.0, 'length': 0.0, 'count': 0},
        'Cyclist': {'height': 0.0, 'width': 0.0, 'length': 0.0, 'count': 0}
    }
    
    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(label_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(label_folder, filename)
            
            # 读取文件内容
            with open(file_path, 'r') as file:
                for line in file:
                    # 分割每行数据
                    data = line.strip().split()
                    
                    # 确保数据格式正确
                    if len(data) >= 11:  # 至少需要到length字段
                        obj_type = data[0]
                        
                        # 只处理指定的三个类别
                        if obj_type in categories:
                            try:
                                # 提取3D尺寸数据（索引8,9,10分别对应height,width,length）
                                height = float(data[8])
                                width = float(data[9])
                                length = float(data[10])
                                
                                # 累加尺寸值并增加计数
                                categories[obj_type]['height'] += height
                                categories[obj_type]['width'] += width
                                categories[obj_type]['length'] += length
                                categories[obj_type]['count'] += 1
                            except ValueError:
                                print(f"警告：文件 {filename} 中存在无效的数值数据: {line}")
    
    # 计算每个类别的平均值
    results = {}
    for obj_type, stats in categories.items():
        if stats['count'] > 0:
            results[obj_type] = {
                'average_height': stats['height'] / stats['count'],
                'average_width': stats['width'] / stats['count'],
                'average_length': stats['length'] / stats['count'],
                'count': stats['count']
            }
        else:
            results[obj_type] = {
                'average_height': 0.0,
                'average_width': 0.0,
                'average_length': 0.0,
                'count': 0
            }
    
    return results

def main():
    # 标签文件夹路径
    label_folder = r"PointAugmenting\data\kitti_radar\training\label_2"
    
    # 检查文件夹是否存在
    if not os.path.exists(label_folder):
        print(f"错误：文件夹不存在 - {label_folder}")
        return
    
    # 计算平均值
    averages = calculate_average_dimensions(label_folder)
    
    # 打印结果
    print("KITTI雷达数据集3D尺寸统计结果：")
    print("-----------------------------------")
    for obj_type, stats in averages.items():
        print(f"类别: {obj_type}")
        print(f"  样本数量: {stats['count']}")
        print(f"  平均高度: {stats['average_height']:.4f}")
        print(f"  平均宽度: {stats['average_width']:.4f}")
        print(f"  平均长度: {stats['average_length']:.4f}")
        print("-----------------------------------")

if __name__ == "__main__":
    main()
