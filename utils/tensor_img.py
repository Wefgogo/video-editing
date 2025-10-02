import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def visualize_tensor_video(tensor, mode="single", save_dir="output", prefix="frame"):
    """
    可视化形状为 (1, f, 3, h, w) 的 tensor
    Args:
        tensor: torch.Tensor, shape = (1, f, 3, h, w)
        mode: "single" 或 "grid"
        save_dir: 输出目录
        prefix: 文件名前缀
    """
    assert tensor.ndim == 5 and tensor.shape[0] == 1, "输入必须是 (1, f, 3, h, w) 形状"
    _, f, c, h, w = tensor.shape
    assert c == 3, "图像必须有3通道 (RGB)"

    os.makedirs(save_dir, exist_ok=True)

    # 归一化到 [0,1]
    imgs = tensor.squeeze(0).clone()  # (f, 3, h, w)
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)

    if mode == "single":
        for i in range(f):
            img = imgs[i]  # (3, h, w)
            save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
            vutils.save_image(img, save_path)
            print(f"保存 {save_path}")

    elif mode == "grid":
        grid = vutils.make_grid(imgs, nrow=4, padding=2, normalize=False)
        save_path = os.path.join(save_dir, f"{prefix}_grid.png")
        vutils.save_image(grid, save_path)
        print(f"保存 {save_path}")

        # 可选：显示
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.show()
    else:
        raise ValueError("mode 必须是 'single' 或 'grid'")


if __name__ == "__main__":
    # 测试代码
    # 随机生成 (1, 10, 3, 64, 64) 的视频
    t = torch.randn(1, 10, 3, 64, 64)

    visualize_tensor_video(t, mode="single", save_dir="output_single", prefix="test")
    visualize_tensor_video(t, mode="grid", save_dir="output_grid", prefix="test")
