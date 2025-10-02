import torch
import numpy as np
import os
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

def run_sam_on_video(video_tensor, sam_checkpoint="C:/Users/17193/Downloads/sam_vit_b_01ec64.pth", save_dir="C:/Users/17193/Downloads/tmp"):
    """
    Args:
        video_tensor: torch.Tensor, shape (1,f,3,H,W)
        sam_checkpoint: str, SAM权重路径
        save_dir: str, 保存mask的目录
    Returns:
        masks: torch.Tensor, shape (1,f,1,H,W)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 加载SAM模型
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    _, f, _, H, W = video_tensor.shape
    all_masks = []

    for i in range(f):
        # 取第 i 帧 (3,H,W) -> (H,W,3)
        frame = video_tensor[0, i].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)  # [0,255]
        predictor.set_image(frame)

        # ⚠️ 提示点（这里简单用图像中心）
        # 实际用时建议你在第一帧点狐狸身体坐标
        input_point = np.array([[W // 2, H // 2]])
        input_label = np.array([1])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        mask = masks[0].astype(np.uint8)  # (H,W)

        # 保存单帧mask PNG
        Image.fromarray(mask * 255).save(os.path.join(save_dir, f"mask_{i:03d}.png"))

        # 转tensor (1,H,W)
        all_masks.append(torch.tensor(mask).unsqueeze(0))

    # 拼接成 (1,f,1,H,W)
    all_masks = torch.stack(all_masks).unsqueeze(0).unsqueeze(2)
    torch.save(all_masks, os.path.join(save_dir, "video_masks.pt"))
    print(f"Saved mask tensor: {all_masks.shape} -> {save_dir}/video_masks.pt")
    return all_masks


# ===== 示例用法 =====
if __name__ == "__main__":
    video_tensor = torch.load("video.pt")  # shape (1,f,3,512,512)
    masks = run_sam_on_video(video_tensor, sam_checkpoint="sam_vit_b_01ec64.pth")
