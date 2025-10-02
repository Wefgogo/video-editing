import cv2
import os


def images_to_video(image_folder, output_path="output.mp4", fps=30):
    # 获取文件夹下所有图片，并按数字排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))  # 按数字顺序排序

    if not images:
        print("文件夹中没有找到图片！")
        return

    # 读取第一张图，获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可改为 'XVID'
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 按顺序写入每张图片
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"视频已保存到: {output_path}")

# 示例调用
image_path = r"E:\dataset\g_t\a_teal_moped"
out_path = r"C:\Users\17193\Downloads\tmp\output.mp4"
images_to_video(image_path, out_path, fps=24)
