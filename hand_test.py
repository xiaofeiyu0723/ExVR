import numpy as np
from scipy.interpolate import PchipInterpolator


def create_spline_mapping(control_points):
    """
    创建0-1区间非线性映射的样条函数

    参数:
        control_points (list): 包含(x, y)元组的列表，必须包含(0,0)和(1,1)
        示例: [(0,0), (0.3,0.2), (0.7,0.8), (1,1)]

    返回:
        function: 输入x返回映射后的y值
    """
    x = np.array([cp[0] for cp in control_points])
    y = np.array([cp[1] for cp in control_points])

    # 验证输入有效性
    assert x[0] == 0 and y[0] == 0, "起始点必须为(0,0)"
    assert x[-1] == 1 and y[-1] == 1, "结束点必须为(1,1)"
    assert np.all((x >= 0) & (x <= 1)), "x值必须在[0,1]区间"
    assert np.all((y >= 0) & (y <= 1)), "y值必须在[0,1]区间"
    assert np.all(np.diff(x) > 0), "x值必须严格递增"

    return PchipInterpolator(x, y)


# 使用示例
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 定义控制点（可自定义修改）
    control_points = [
        (0, 0),
        (0.05, 0.2),  # 曲线起始平缓
        (0.5, 0.3),  # 曲线中部陡峭
        (1, 1)
    ]

    # 创建映射函数
    mapper = create_spline_mapping(control_points)

    # 生成测试数据
    x = np.linspace(0, 1, 100)
    y = mapper(x)

    # 绘制映射曲线
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='样条映射')
    plt.plot([0, 1], [0, 1], '--', label='线性映射')
    plt.scatter(*zip(*control_points), color='red', zorder=5, label='控制点')
    plt.legend()
    plt.xlabel('输入值')
    plt.ylabel('输出值')
    plt.title('0-1区间非线性样条映射')
    plt.grid(True)
    plt.show()