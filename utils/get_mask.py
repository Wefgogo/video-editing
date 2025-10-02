import torch
import numpy as np
from rembg import remove
from PIL import Image
import os

def extract_video_masks(video_tensor, save_dir="masks", filename="video_masks.pt"):
    """
    输入: video_tensor (1, f, 3, 512, 512)
    输出: 保存 mask tensor (1, f, 1, 512, 512)，并返回路径
    """
    assert video_tensor.dim() == 5, "Input must be (1, f, 3, H, W)"
    _, f, _, h, w = video_tensor.shape

    os.makedirs(save_dir, exist_ok=True)
    masks = []

    for i in range(f):
        # 取第 i 帧 (3,H,W) -> (H,W,3)
        frame = video_tensor[0, i].permute(1, 2, 0).detach().cpu().numpy()
        frame = (frame * 255).astype(np.uint8)  # 转到 [0,255] 范围
        img = Image.fromarray(frame)

        # 前景分割
        out = remove(img)  # RGBA 图
        alpha = np.array(out.getchannel("A"))  # 取 alpha 通道

        # 转成 mask: 1=前景, 0=背景
        mask = (alpha > 0).astype(np.uint8)
        mask = torch.tensor(mask).unsqueeze(0)  # (1,H,W)
        masks.append(mask)

        # 可选: 保存单帧 mask PNG
        Image.fromarray((mask.squeeze().numpy() * 255).astype(np.uint8)).save(
            os.path.join(save_dir, f"mask_{i:03d}.png")
        )

    # 堆叠成 tensor (1,f,1,H,W)
    masks = torch.stack(masks, dim=0).unsqueeze(0)
    torch.save(masks, os.path.join(save_dir, filename))

    print(f"Saved masks tensor to {os.path.join(save_dir, filename)}")
    return masks


# ===== 示例用法 =====
if __name__ == "__main__":
    # 假设你的视频张量在 ./video.pt
    video_tensor = torch.load("video.pt")  # shape (1,f,3,512,512)
    masks = extract_video_masks(video_tensor, save_dir="masks", filename="video_masks.pt")
