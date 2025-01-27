import cv2
import numpy as np

def enhance_brightness(image, alpha=1.2, beta=30):
    """
    增强图像亮度和对比度
    :param image: 输入图像
    :param alpha: 对比度增益系数 (1.0 ~ 3.0)
    :param beta: 亮度增加值 (0 ~ 100)
    :return: 增强后的图像
    """
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_image

def apply_clahe(image):
    """
    应用 CLAHE（限制对比度自适应直方图均衡化）增强细节
    :param image: 灰度图像
    :return: 增强后的图像
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# 打开摄像头（默认摄像头编号为0）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 实时读取视频流
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    # 调整亮度和对比度
    brightened_frame = enhance_brightness(frame, alpha=1.5, beta=50)

    # 转换为灰度图并应用 CLAHE
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_frame = apply_clahe(gray_frame)

    # 显示原始视频和增强后的视频
    cv2.imshow('Original Video', frame)
    cv2.imshow('Brightened Video', brightened_frame)
    cv2.imshow('CLAHE Enhanced Video', clahe_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
