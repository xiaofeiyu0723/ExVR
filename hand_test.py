import numpy as np
import cv2


class Stabilizer:
    """支持1D/2D/3D点的卡尔曼滤波器稳定器"""

    def __init__(self,
                 state_num=4,
                 measure_num=2,
                 cov_process=0.0001,
                 cov_measure=0.1):
        # 参数校验
        assert measure_num in (1, 2, 3), "仅支持1D/2D/3D测量"
        assert state_num in (2, 4, 6), "状态维度应为2(1D)/4(2D)/6(3D)"

        self.state_num = state_num
        self.measure_num = measure_num

        # 初始化卡尔曼滤波器
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)
        self.state = np.zeros((state_num, 1), dtype=np.float32)
        self.measurement = np.zeros((measure_num, 1), np.float32)
        self.prediction = np.zeros((state_num, 1), np.float32)

        # 配置参数矩阵
        if measure_num == 1:  # 1D情况
            self.filter.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
            self.filter.measurementMatrix = np.array([[1, 0]], np.float32)
            self.filter.processNoiseCov = np.eye(2, dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.eye(1, dtype=np.float32) * cov_measure

        elif measure_num == 2:  # 2D情况
            self.filter.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], np.float32)
            self.filter.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]], np.float32)
            self.filter.processNoiseCov = np.eye(4, dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * cov_measure

        elif measure_num == 3:  # 3D情况
            self.filter.transitionMatrix = np.array([
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]], np.float32)
            self.filter.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]], np.float32)
            self.filter.processNoiseCov = np.eye(6, dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.eye(3, dtype=np.float32) * cov_measure

    def update(self, measurement):
        """更新滤波器状态"""
        self.prediction = self.filter.predict()

        # 处理不同维度的测量输入
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement[0])]])
        elif self.measure_num == 2:
            self.measurement = np.array([
                [np.float32(measurement[0])],
                [np.float32(measurement[1])]])
        else:
            self.measurement = np.array([
                [np.float32(measurement[0])],
                [np.float32(measurement[1])],
                [np.float32(measurement[2])]])

        self.filter.correct(self.measurement)
        self.state = self.filter.statePost

    def set_q_r(self, cov_process=0.1, cov_measure=0.001):
        """动态调整噪声参数"""
        if self.measure_num == 1:
            self.filter.processNoiseCov = np.eye(2) * cov_process
            self.filter.measurementNoiseCov = np.eye(1) * cov_measure
        elif self.measure_num == 2:
            self.filter.processNoiseCov = np.eye(4) * cov_process
            self.filter.measurementNoiseCov = np.eye(2) * cov_measure
        else:
            self.filter.processNoiseCov = np.eye(6) * cov_process
            self.filter.measurementNoiseCov = np.eye(3) * cov_measure


# 测试三种情况的演示代码
def main():
    # 测试1D滤波器
    test_1d()
    # 测试2D滤波器
    test_2d()
    # 测试3D滤波器
    test_3d()


def test_1d():
    print("测试1D滤波器...")
    kalman = Stabilizer(state_num=2, measure_num=1)
    measurements = [x + np.random.normal(0, 3) for x in range(1, 100, 2)]

    filtered = []
    for m in measurements:
        kalman.update([m])
        filtered.append(kalman.state[0][0])

    print("原始测量:", measurements[:5])
    print("滤波结果:", [round(x, 2) for x in filtered[:5]])


def test_2d():
    print("\n测试2D滤波器...")
    kalman = Stabilizer(state_num=4, measure_num=2)

    # 生成带噪声的圆形轨迹
    points = []
    for t in np.linspace(0, 2 * np.pi, 20):
        x = 300 + 100 * np.cos(t) + np.random.normal(0, 5)
        y = 300 + 100 * np.sin(t) + np.random.normal(0, 5)
        points.append((x, y))

    # 可视化
    cv2.namedWindow("2D Stabilization")
    canvas = np.zeros((600, 600, 3), np.uint8)

    for (x, y) in points:
        kalman.update([x, y])
        state = kalman.state

        # 绘制原始测量（红色）
        cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1)
        # 绘制滤波结果（绿色）
        cv2.circle(canvas, (int(state[0]), int(state[1])), 3, (0, 255, 0), -1)

        cv2.imshow("2D Stabilization", canvas)
        cv2.waitKey(100)
        canvas[:] = 0

    cv2.destroyAllWindows()


def test_3d():
    print("\n测试3D滤波器...")
    kalman = Stabilizer(state_num=6, measure_num=3)

    # 生成螺旋轨迹
    points = []
    for t in np.linspace(0, 4 * np.pi, 50):
        x = 300 + 50 * t * np.cos(t) + np.random.normal(0, 5)
        y = 300 + 50 * t * np.sin(t) + np.random.normal(0, 5)
        z = 100 * t + np.random.normal(0, 10)
        points.append((x, y, z))

    # 3D投影可视化
    cv2.namedWindow("3D Stabilization")
    canvas = np.zeros((600, 800, 3), np.uint8)

    for (x, y, z) in points:
        kalman.update([x, y, z])
        state = kalman.state

        # 3D投影（简单正交投影）
        def project(point):
            return (int(point[0] + point[2] / 5), int(point[1] - point[2] / 5))

        # 绘制原始点（红色）
        raw_proj = project((x, y, z))
        cv2.circle(canvas, raw_proj, 4, (0, 0, 255), -1)

        # 绘制滤波结果（绿色）
        filtered_proj = project(state[:3].flatten())
        cv2.circle(canvas, filtered_proj, 3, (0, 255, 0), -1)

        cv2.imshow("3D Stabilization", canvas)
        cv2.waitKey(100)
        canvas[:] = 0

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()