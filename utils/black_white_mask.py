import os
import torch
import torchvision.utils as vutils
from PIL import Image

def double_mask(source, mask, save_dir=r"C:\Users\17193\Downloads\tmp\imgs", scale_mode="auto", save_overlay=True):
    """
    source: torch.Tensor, shape (12,3,H,W), values can be in arbitrary range (e.g. [-1,0])
    mask:   torch.Tensor, shape (1,12,1,H,W) or (12,1,H,W), values in [0,1] (probability or binary)
    save_dir: 输出保存目录
    scale_mode: "auto" | "fixed" | "shift"
        - "fixed": 假设 source ∈ [-1,1], 使用 (x+1)/2
        - "shift": 假设 source ∈ [-1,0], 使用 (x+1)
        - "auto": 按实际 min/max 线性缩放到 [0,1]
    save_overlay: 是否保存带红色半透明掩码的叠加图用于调试
    """
    os.makedirs(os.path.join(save_dir, "white"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "black"), exist_ok=True)
    if save_overlay:
        os.makedirs(os.path.join(save_dir, "overlay"), exist_ok=True)

    # 确保 tensor 类型
    source = source.float()
    mask = mask.float()

    # 调整 mask 形状 (1,12,1,H,W) -> (12,1,H,W) 或保持 (12,1,H,W)
    if mask.dim() == 5 and mask.shape[0] == 1:
        mask = mask.squeeze(0)  # -> (12,1,H,W)
    elif mask.dim() == 4 and mask.shape[0] == source.shape[0]:
        # mask is already (12,1,H,W)
        pass
    else:
        raise ValueError(f"mask shape not supported: {mask.shape}")

    # 将 mask 归一并确保范围 [0,1]
    mask = mask.clamp(0,1)

    # 选择缩放策略
    if scale_mode == "fixed":
        source_norm = (source + 1.0) / 2.0
    elif scale_mode == "shift":
        # 适合 source 在 [-1,0] 的情况 -> 直接 +1 映射到 [0,1]
        source_norm = (source + 1.0)
    elif scale_mode == "auto":
        s_min = float(source.min().item())
        s_max = float(source.max().item())
        if s_max <= s_min:
            # 常数图像，直接归一化为 0.5 避免除0
            source_norm = torch.full_like(source, 0.5)
        else:
            source_norm = (source - s_min) / (s_max - s_min)
    else:
        raise ValueError("scale_mode must be one of 'fixed','shift','auto'")

    # 确保在 [0,1]
    source_norm = source_norm.clamp(0, 1)

    # 计算 background（未被 mask 的区域保留源像素）
    # mask shape (12,1,H,W) -> 广播到 (12,3,H,W)
    back = source_norm * (1 - mask)

    # 白底：mask 区域设为白色 (1.0)
    white_bg = back + mask * 1.0

    # 黑底：mask 区域为 0（back 已经是这样）
    black_bg = back

    # 保存逐帧图像；并可选择保存overlay供调试
    import numpy as np
    for i in range(source.shape[0]):
        w_path = os.path.join(save_dir, "white", f"frame_{i:02d}.png")
        b_path = os.path.join(save_dir, "black", f"frame_{i:02d}.png")

        vutils.save_image(white_bg[i], w_path)
        vutils.save_image(black_bg[i], b_path)

        if save_overlay:
            # 生成带红色半透明 mask 的调试图
            img_uint8 = to_uint8_img(white_bg[i])  # 或者 black_bg[i] 都可以
            # mask 单通道 HxW
            mask_np = mask[i, 0].cpu().numpy()
            overlay_path = os.path.join(save_dir, "overlay", f"overlay_{i:02d}.png")
            save_overlay_img(img_uint8, mask_np, overlay_path, alpha=0.45)

    print(f"✅ 已保存 {source.shape[0]} 帧到：{os.path.abspath(save_dir)} (mode={scale_mode})")
       # back.shape:(1, 12, 3, 512, 512)   mask.shape(1, 12, 1, 512, 512)

def to_uint8_img(tensor):
    """
    tensor: [3,H,W] in [0,1], torch.Tensor (cpu).
    返回: HxW x3 的 uint8 numpy array
    """
    arr = (tensor.clamp(0,1).mul(255)).byte().permute(1, 2, 0).cpu().numpy()
    return arr

def save_overlay_img(img_uint8, mask_01, out_path, alpha=0.45):
    """
    img_uint8: HxWx3 uint8 np array
    mask_01: HxW float np array in [0,1]
    将 mask>0 位置用半透明红色覆盖并保存（便于调试）
    """
    from PIL import Image
    base = Image.fromarray(img_uint8)  # RGB
    red = Image.new("RGB", base.size, (255, 0, 0))
    # mask -> L (0..255)
    mask_img = Image.fromarray((mask_01 * 255).clip(0,255).astype("uint8"))
    # 先把 mask_img 变成二值以便更清晰（你可按需注释）
    # mask_img = mask_img.point(lambda p: 255 if p>127 else 0)
    # 使用 mask_img 作为 alpha 将 red 与 base 合成，然后与 base 调整透明度
    red_part = Image.composite(red, base, mask_img)  # red over base where mask>0
    blended = Image.blend(base, red_part, alpha)
    blended.save(out_path)