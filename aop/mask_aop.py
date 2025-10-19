import torch
from einops import rearrange

def ComparingMask(pixel_values, mask, vae, video_length):
    mask = mask.squeeze(2)

    weight_dtype = pixel_values.dtype

    # ------------------- 新增：生成对称掩码的两个图像变体 -------------------
    # 变体1：mask区域（1）填黑（RGB=0），非mask区域保留原始像素
    pixel_black = pixel_values.clone()
    pixel_black = torch.where(mask.unsqueeze(2) == 1, torch.tensor(0.0, dtype=weight_dtype, device=pixel_black.device),
                              pixel_black)

    # 变体2：mask区域（1）填白（RGB=1，假设pixel_values已归一化到[0,1]），非mask区域保留原始像素
    pixel_white = pixel_values.clone()
    pixel_white = torch.where(mask.unsqueeze(2) == 1, torch.tensor(1.0, dtype=weight_dtype, device=pixel_white.device),
                              pixel_white)

    # ------------------- 新增：双变体编码为latent并拼接 -------------------
    # 1. 处理pixel_black的latent
    pixel_black_flat = rearrange(pixel_black, "b f c h w -> (b f) c h w")
    latents_black = vae.encode(pixel_black_flat).latent_dist.sample()
    latents_black = rearrange(latents_black, "(b f) c h w -> b c f h w", f=video_length)
    latents_black = latents_black * 0.18215

    # 2. 处理pixel_white的latent
    pixel_white_flat = rearrange(pixel_white, "b f c h w -> (b f) c h w")
    latents_white = vae.encode(pixel_white_flat).latent_dist.sample()
    latents_white = rearrange(latents_white, "(b f) c h w -> b c f h w", f=video_length)
    latents_white = latents_white * 0.18215

    # 3. 拼接双变体latent：shape从(b, c, f, h, w)变为(b, 2*c, f, h, w)
    # 这一步替换你原有“mask_latent与latents结合”的逻辑，无需再下采样mask
    latents_input = torch.cat([latents_black, latents_white], dim=1)  # 作为UNet的输入latent

    return latents_input