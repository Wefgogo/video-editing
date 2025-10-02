import cv2
import os


def extract_frames(video_path, output_dir):
    """
    将视频的每一帧保存为单独的图片

    参数:
        video_path (str): 输入视频文件路径
        output_dir (str): 输出目录路径
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_count = 0

    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果读取失败（可能是视频结束了）
        if not ret:
            break

        # 构造输出文件名
        output_path = os.path.join(output_dir, f"{frame_count}.png")

        # 保存帧为PNG图片
        cv2.imwrite(output_path, frame)

        print(f"已保存帧: {output_path}")
        frame_count += 1

    # 释放资源
    cap.release()
    print(f"处理完成，共保存 {frame_count} 帧")


if __name__ == "__main__":
    # 输入视频文件路径
    video_path = r"E:\RemoteCode\VideoEditingRemote\data\libby.mp4" # 替换为你的视频路径

    # 输出目录
    output_dir = r"C:\Users\17193\Downloads\tmp\imgs"

    # 调用函数提取帧
    extract_frames(video_path, output_dir)