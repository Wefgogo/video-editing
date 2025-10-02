import cv2
import numpy as np
from PIL import Image


def video_frames_to_grid(video_path, output_path, frames_per_row=4, max_frames=None, scale=1.0):
    """
    将视频帧拼接成网格大图

    参数:
        video_path (str): 输入视频文件路径
        output_path (str): 输出图片路径
        frames_per_row (int): 每行显示的帧数，默认为4
        max_frames (int): 最大处理帧数，None表示处理所有帧
        scale (float): 缩放因子，1.0为原尺寸
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 如果设置了最大帧数限制且已达到，则停止
        if max_frames is not None and frame_count >= max_frames:
            break

        # 转换颜色空间从BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 缩放帧
        if scale != 1.0:
            width = int(frame_rgb.shape[1] * scale)
            height = int(frame_rgb.shape[0] * scale)
            frame_rgb = cv2.resize(frame_rgb, (width, height))

        frames.append(Image.fromarray(frame_rgb))
        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError("视频中没有读取到任何帧")

    # 计算网格布局
    num_frames = len(frames)
    num_rows = (num_frames + frames_per_row - 1) // frames_per_row

    # 获取单帧尺寸
    frame_width, frame_height = frames[0].size

    # 创建大图
    grid_width = frame_width * frames_per_row
    grid_height = frame_height * num_rows
    grid = Image.new('RGB', (grid_width, grid_height))

    # 将帧粘贴到网格中
    for i, frame in enumerate(frames):
        row = i // frames_per_row
        col = i % frames_per_row
        grid.paste(frame, (col * frame_width, row * frame_height))

    # 保存结果
    grid.save(output_path)
    print(f"成功保存拼接图到 {output_path}，共 {num_frames} 帧")


# 使用示例
if __name__ == "__main__":
    video_path = r"D:\Code\Project\Diffusion_eg\Tune-A-Video\data\horsejump-high.mp4"  # 替换为你的视频路径
    output_path = r"C:\Users\17193\Downloads\tmp\big.jpg"  # 输出图片路径
    video_frames_to_grid(video_path, output_path, frames_per_row=4)