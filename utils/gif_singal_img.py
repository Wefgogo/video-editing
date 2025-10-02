from PIL import Image
import os

# ====== 配置路径 ======
gif_path = r"C:\Users\17193\Downloads\tmp\a fox is running through the grass.gif"# 替换为你的 GIF 路径
output_dir = r"C:\Users\17193\Downloads\tmp\imgs"      # 输出图片文件夹

# ====== 创建输出文件夹 ======
os.makedirs(output_dir, exist_ok=True)

# ====== 打开 GIF 并提取帧 ======
with Image.open(gif_path) as im:
    frame_index = 0
    try:
        while True:
            frame = im.copy().convert("RGB")  # 可加 convert("RGBA") 视需求
            frame.save(os.path.join(output_dir, f"frame_{frame_index:03d}.png"))
            frame_index += 1
            im.seek(im.tell() + 1)
    except EOFError:
        print(f"共保存 {frame_index} 帧到 {output_dir}")
