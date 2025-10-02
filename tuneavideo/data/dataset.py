import decord
import torch

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
import torch.nn.functional as F


class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1



    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids,
        }

        return example

class TuneAVideoDepthDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            depth_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None
        self.depth_path = depth_path

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def _resize_depth_tensor(self, depth_tensor, target_width, target_height):
        """
        使用双线性插值将深度图缩放到目标尺寸（与 decord.VideoReader 默认行为一致）

        参数:
            depth_tensor (torch.Tensor): 输入深度图 [B, C, H, W]
            target_width (int): 目标宽度
            target_height (int): 目标高度

        返回:
            torch.Tensor: 缩放后的深度图 [B, C, target_height, target_width]
        """
        # 确保输入是 4D 张量 [B, C, H, W]
        if len(depth_tensor.shape) != 4:
            raise ValueError("输入深度图必须是 4D 张量 [B, C, H, W]")

        # 使用双线性插值缩放
        resized_depth = F.interpolate(
            depth_tensor,
            size=(target_height, target_width),
            mode="bilinear",  # 与 decord 默认一致
            align_corners=False  # 与 OpenCV/decord 的默认行为一致
        )

        return resized_depth

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        depth = torch.load(self.depth_path)
        depth = self._resize_depth_tensor(depth, 64, 64)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids,
            "depth": depth
        }

        return example
