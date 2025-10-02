import os
from PIL import Image

def gif():
    # GIF 文件路径
    gif_path = r"C:\Users\17193\Downloads\tmp\sample-500\a man on roller skates is playing hockey.gif"
    output_dir = r"C:\Users\17193\Downloads\tmp"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开 GIF 文件
    gif = Image.open(gif_path)

    # 每行显示的帧数
    frames_per_row = 4

    # 提取所有帧并保存到列表中
    frames = []
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass  # 所有帧已提取完毕

    # 获取单帧的尺寸
    frame_width, frame_height = frames[0].size

    # 计算拼接后大图的尺寸
    total_rows = (len(frames) + frames_per_row - 1) // frames_per_row
    merged_width = frames_per_row * frame_width
    merged_height = total_rows * frame_height

    # 创建空白大图
    merged_image = Image.new('RGB', (merged_width, merged_height), (255, 255, 255))

    # 将每一帧粘贴到大图的相应位置
    for index, frame in enumerate(frames):
        x = (index % frames_per_row) * frame_width
        y = (index // frames_per_row) * frame_height
        merged_image.paste(frame.convert('RGB'), (x, y))

    # 保存拼接后的大图
    merged_image_path = os.path.join(output_dir, "merged_frames.png")
    merged_image.save(merged_image_path)
    print(f"所有帧已提取并拼接为一张大图，保存路径为: {merged_image_path}")

gif()