from tuneavideo.pipelines.pipeline_3dvideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch
from tuneavideo.util import save_videos_grid, ddim_inversion
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from tuneavideo.data.dataset import TuneAVideoDataset
from einops import rearrange

pretrained_model_path = "/local_codes/pretrain/stable-diffusion-2-1-base/"
my_model_path = "./outputs/libby"
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to("cuda")


# 准备视频
train_data = {
  "video_path": "data/libby.mp4",
  "prompt": "a german shepherd dog running through the grass",
  # "prompt": "a rasadf fox",
  "n_sample_frames": 12,
  "width": 512,
  "height": 512,
  "sample_start_idx": 0,
  "sample_frame_rate": 1}
train_dataset = TuneAVideoDataset(**train_data)
example = train_dataset.__getitem__(0)
pixel_values = example["pixel_values"]
pixel_values = pixel_values.unsqueeze(0).to(torch.float16).to("cuda")
prompt_ids = example["prompt_ids"]
video_length = pixel_values.shape[1]
pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
latents = vae.encode(pixel_values).latent_dist.sample()
# latents = torch.randn((24, 4, 64, 64), dtype=torch.float16, device=accelerator.device)
latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
latents = latents * 0.18215


validation_data ={
    "prompts": "a rasadf fox",
    # - "spider man is skiing on the beach, cartoon style"
    # - "wonder woman, wearing a cowboy hat, is skiing"
    # - "a man, wearing pink clothes, is skiing at sunset"
  "video_length": 12,
  "width": 512,
  "height": 512,
  "num_inference_steps": 50,
  "guidance_scale": 12.5,
  "use_inv_latent": True,
  "num_inv_steps": 50}

ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
ddim_inv_scheduler.set_timesteps(validation_data["num_inv_steps"])
ddim_inv_latent = ddim_inversion(
                                pipe, ddim_inv_scheduler, video_latent=latents,
                                num_inv_steps=validation_data["num_inv_steps"], prompt="")[-1].to(torch.float16)

grad = torch.load("./data_3d/fox/target.pt")   #(12, 4, 64, 64)
print("shape", grad.shape)

prompt = "a rasadf fox running through the grass"
# ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)
video = pipe(prompt, latents=ddim_inv_latent, video_length=12, height=512, width=512, num_inference_steps=50, guidance_scale=12.5, grad=grad).videos

save_videos_grid(video, f"./outputs/zero_out/{prompt}.gif") 