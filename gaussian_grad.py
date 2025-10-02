from threestudio.systems.GaussianDreamer import GaussianDreamer
import argparse
from threestudio.utils.config import *
from threestudio.data.uncond import *
from gaussiansplatting.scene.gaussian_model import *
from gaussiansplatting.arguments import *
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.gaussian_renderer import render
from threestudio.models.guidance.stable_diffusion_guidance import StableDiffusionGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor


class gaussian_model():
    def __init__(self, cfg_model, cfg_data, cfg_guidance, cfg_prompt, max_steps=1000):
        self.cfg_model = cfg_model
        self.max_steps = max_steps
        self.datasetModel = RandomCameraDataModule(cfg_data)
        self.datasetModel.setup("fit")
        self.dataloader = self.datasetModel.train_dataloader()
        self.guidance = StableDiffusionGuidance(cfg_guidance)
        self.prompt_processor = threestudio.find("stable-diffusion-prompt-processor")(
            cfg_prompt
        )
        self.prompt_utils = self.prompt_processor()

    def get_gaussian(self):
        gaussians = GaussianModel(sh_degree=0)
        bg_color = [0, 0, 0]
        background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        parser = ArgumentParser(description="Training script parameters")
        opt = OptimizationParams(parser)
        pipe = PipelineParams(parser)
        gaussians.load_ply("./data_3d/fox/last_3dgs.ply")
        return gaussians, pipe, background_tensor

    def forward(self):
        gaussian, pipe, renderbackground = self.get_gaussian()
        dataloader_iter = iter(self.dataloader)
        batch = next(dataloader_iter)
        images = []
        for id in range(batch['c2w_3dgs'].shape[0]):
            viewpoint_cam = Camera(c2w=batch['c2w_3dgs'][id], FoVy=batch['fovy'][id], height=batch['height'],
                                   width=batch['width'])
            render_pkg = render(viewpoint_cam, gaussian, pipe, renderbackground)

            image = render_pkg["render"]
            image = image.permute(1, 2, 0)
            images.append(image)
        guidance_eval = False
        images = torch.stack(images, 0)
        guidance_out = self.guidance(
            images, self.prompt_utils, **batch, rgb_as_latents=False, guidance_eval=guidance_eval
        )
        target = guidance_out['target']
        return images, target



def train_gaussian_dreamer(cfg_model, cfg_data, cfg_guidance, cfg_prompt, max_steps=1000):
    # 初始化数据
    datasetModel = RandomCameraDataModule(cfg_data)
    datasetModel.setup("fit")
    dataloader = datasetModel.train_dataloader()
    guidance = StableDiffusionGuidance(cfg_guidance)
    # prompt_utils = StableDiffusionPromptProcessor(cfg_prompt)
    prompt_processor = threestudio.find("stable-diffusion-prompt-processor")(
            cfg_prompt
        )
    prompt_utils = prompt_processor()
    

    # 初始化模型
    # gaussDream = GaussianDreamer(cfg_model)
    # optimizer = gaussDream.configure_optimizers()
    gaussian, pipe, renderbackground = get_gaussian()

    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)
    images = []
    for id in range(batch['c2w_3dgs'].shape[0]):
        viewpoint_cam = Camera(c2w=batch['c2w_3dgs'][id], FoVy=batch['fovy'][id], height=batch['height'],
                               width=batch['width'])
        render_pkg = render(viewpoint_cam, gaussian, pipe, renderbackground)

        image = render_pkg["render"]
        image = image.permute(1, 2, 0)
        images.append(image)
    guidance_eval = False
    images = torch.stack(images, 0)
    guidance_out = guidance(
        images, prompt_utils, **batch, rgb_as_latents=False, guidance_eval=guidance_eval
    )

    grad = guidance_out['grad']
    target = guidance_out['target']
    torch.save(grad, "./data_3d/fox/grad.pt")
    torch.save(target , "./data_3d/fox/target.pt")
    print("grad---shape", grad.shape)


    # gaussDream.on_fit_start()

    # for step in range(max_steps):
    #     gaussDream.true_global_step = step
    #
    #     # 取出一个 batch，循环 dataloader（避免 StopIteration）
    #     try:
    #         batch = next(dataloader_iter)
    #     except StopIteration:
    #         dataloader_iter = iter(dataloader)
    #         batch = next(dataloader_iter)
    #
    #     # 正常训练逻辑
    #     optimizer.zero_grad()
    #     outputs = gaussDream.training_step(batch, 0)
    #     loss = outputs["loss"]
    #     gaussDream.on_before_optimizer_step(optimizer)
    #     loss.backward()
    #     optimizer.step()
    #
    #     # 打印日志
    #     print(f"[Train] Step {step:04d} | Loss: {loss.item():.6f}")

def get_cfg():
    cfg = {
        "radius": 4,
        "sh_degree": 0,
        "load_type": 1,
        "load_path": "./load/shapes/stand.obj",
        "prompt": "stand",
        "loss": {
            "lambda_sds": 1.0,
            "lambda_sparsity": 0.5,
            "lambda_opaque": 0.1
        }
    }
    cfg_data = {
        "load_type": 1,
        "batch_size": 12,
        "eval_camera_distance": 4.0,
        "camera_distance_range": [1.5, 4.0],
        "light_sample_strategy": "dreamfusion3dgs",
        "height": 1024,
        "width": 1024,
        "eval_height": 1024,
        "eval_width": 1024,
    }

    cfg_guidance = {
        "pretrained_model_name_or_path": "/local_codes/pretrain/stable-diffusion-2-1-base/",
        "guidance_scale": 100.,
        "weighting_strategy": "sds",
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
        "grad_clip": [0, 1.5, 2.0, 1000],
    }

    cfg_prompt = {
        "pretrained_model_name_or_path": "/local_codes/pretrain/stable-diffusion-2-1-base/",
        "prompt": "stand",
        "negative_prompt": "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions, unfocused",

    }
    return cfg, cfg_data, cfg_guidance, cfg_prompt


def get_gaussian():
    gaussians = GaussianModel(sh_degree=0)
    bg_color = [0, 0, 0]
    background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    parser = ArgumentParser(description="Training script parameters")
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)
    gaussians.load_ply("./data_3d/fox/last_3dgs.ply")
    return gaussians, pipe, background_tensor

def main(args, extras):
    cfg = {
        "radius": 4,
        "sh_degree": 0,
        "load_type": 1,
        "load_path": "./load/shapes/stand.obj",
        "prompt": "stand",
        "loss": {
            "lambda_sds": 1.0,
            "lambda_sparsity": 0.5,
            "lambda_opaque": 0.1
        }
    }
    cfg_data = {
        "load_type": 1,
        "batch_size": 12,
        "eval_camera_distance": 4.0,
        "camera_distance_range": [1.5, 4.0],
        "light_sample_strategy": "dreamfusion3dgs",
        "height": 1024,
        "width": 1024,
        "eval_height": 1024,
        "eval_width": 1024,
    }

    cfg_guidance = {
        "pretrained_model_name_or_path": "/local_codes/pretrain/stable-diffusion-2-1-base/",
        "guidance_scale": 100.,
        "weighting_strategy": "sds",
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
        "grad_clip": [0, 1.5, 2.0, 1000],
    }

    cfg_prompt = {
        "pretrained_model_name_or_path": "/local_codes/pretrain/stable-diffusion-2-1-base/",
        "prompt": "stand",
        "negative_prompt": "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions, unfocused",

    }
    # 初始化数据模块
    train_gaussian_dreamer(cfg, cfg_data, cfg_guidance, cfg_prompt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs\gaussiandreamer-sd.yaml", help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
             "1,2 means use the 2nd and 3rd available GPU. "
             "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
             "this argument is ignored and all available GPUs are always used.",
    )
    args, extras = parser.parse_known_args()

    main(args, extras)
