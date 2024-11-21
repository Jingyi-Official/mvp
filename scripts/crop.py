import cv2
import mediapipe as mp
import numpy as np
import os

def create_hand_centered_video(input_video, output_video, output_size=(1024, 1024)):
    """
    将视频裁剪为以手为中心的 1024x1024 视频。

    Args:
        input_video (str): 输入视频路径。
        output_video (str): 输出视频路径。
        output_size (tuple): 裁剪视频帧的目标尺寸，默认为 (1024, 1024)。
    """
    # 初始化 Mediapipe Hands 模块
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 打开输入视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"无法打开视频文件：{input_video}")
        return

    # 获取视频帧率和尺寸
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, output_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为 RGB 格式（Mediapipe 使用 RGB 格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用 Mediapipe 进行手部检测
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取手部关键点坐标
                x_coords = [lm.x * width for lm in hand_landmarks.landmark]
                y_coords = [lm.y * height for lm in hand_landmarks.landmark]

                # 计算手部的边界框
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

                # 计算裁剪区域
                crop_x1 = max(center_x - output_size[0] // 2, 0)
                crop_y1 = max(center_y - output_size[1] // 2, 0)
                crop_x2 = min(crop_x1 + output_size[0], width)
                crop_y2 = min(crop_y1 + output_size[1], height)

                # 裁剪手部区域
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                # 填充不足区域
                padded_frame = cv2.copyMakeBorder(
                    cropped_frame,
                    top=(output_size[1] - cropped_frame.shape[0]) // 2,
                    bottom=(output_size[1] - cropped_frame.shape[0]) // 2,
                    left=(output_size[0] - cropped_frame.shape[1]) // 2,
                    right=(output_size[0] - cropped_frame.shape[1]) // 2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

                # 调整到目标尺寸
                padded_frame = cv2.resize(padded_frame, output_size)
                out.write(padded_frame)
                break  # 如果只需要关注一只手，处理一只手后跳出循环
        else:
            # 如果未检测到手，直接缩放帧到目标大小
            resized_frame = cv2.resize(frame, output_size)
            out.write(resized_frame)

    # 释放资源
    cap.release()
    out.release()
    hands.close()
    print(f"视频处理完成，保存为：{output_video}")

# 使用示例
input_video = "/mnt/ssd/jingyi/Projects/hamer/data/test/video.mp4"  # 输入视频路径
output_video = "/mnt/ssd/jingyi/Projects/hamer/data/test/video_crop.mp4"  # 输出视频路径
create_hand_centered_video(input_video, output_video)
