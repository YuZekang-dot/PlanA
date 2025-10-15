import math

def calculate_pc_range():
    # 雷达参数
    radar_height = 1.85  # 雷达安装高度，单位：米
    max_distance = 25    # 最大探测距离，单位：米
    azimuth_angle = 56.5 # 方位角范围，单位：度
    elevation_angle = 22.5 # 俯仰角范围，单位：度
    
    # 将角度转换为弧度
    azimuth_rad = math.radians(azimuth_angle)
    elevation_rad = math.radians(elevation_angle)
    
    # 计算x方向范围 (假设x轴为车辆前进方向)
    x_min = 0.0  # 雷达位置为起点
    x_max = max_distance  # 最大探测距离
    
    # 计算y方向范围 (假设y轴为水平侧向)
    # 根据方位角和最大距离计算左右边界
    y_half_range = max_distance * math.tan(azimuth_rad)
    y_min = -y_half_range  # 左侧边界
    y_max = y_half_range   # 右侧边界
    
    # 计算z方向范围 (高度方向)
    # 根据俯仰角和最大距离计算上下边界
    z_vertical_range = max_distance * math.tan(elevation_rad)
    z_min = radar_height - z_vertical_range  # 下方边界
    z_max = radar_height + z_vertical_range  # 上方边界
    
    # 确保z_min不会低于地面太多（可选）
    # 如果需要限制最低到地面，可以取消下面这行注释
    z_min = max(z_min, -0.5)  # 允许略低于地面，处理地面附近物体
    
    # 整理pc_range结果，保留4位小数
    pc_range = [
        round(x_min, 4),
        round(y_min, 4),
        round(z_min, 4),
        round(x_max, 4),
        round(y_max, 4),
        round(z_max, 4)
    ]
    
    return pc_range, {
        "radar_height": radar_height,
        "max_distance": max_distance,
        "azimuth_angle": azimuth_angle,
        "elevation_angle": elevation_angle,
        "y_half_range": y_half_range,
        "z_vertical_range": z_vertical_range
    }

def main():
    pc_range, params = calculate_pc_range()
    
    print("OCULII-EAGLE 4D毫米波雷达 pc_range 计算结果：")
    print("------------------------------------------")
    print(f"雷达参数:")
    print(f"  安装高度: {params['radar_height']}米")
    print(f"  最大探测距离: {params['max_distance']}米")
    print(f"  方位角范围: ±{params['azimuth_angle']}°")
    print(f"  俯仰角范围: ±{params['elevation_angle']}°")
    print("\n计算详情:")
    print(f"  侧向半宽: {params['y_half_range']:.4f}米")
    print(f"  垂直范围: ±{params['z_vertical_range']:.4f}米")
    print("\n最终pc_range:")
    print(f"  [x_min, y_min, z_min, x_max, y_max, z_max]")
    print(f"  {pc_range}")
    print("\n可以直接使用的配置格式:")
    print(f"pc_range = {pc_range}")

if __name__ == "__main__":
    main()
    