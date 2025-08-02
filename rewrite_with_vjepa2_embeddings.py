import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseVideoDataset
import sys

import diffusion_policy.vjepa2.datasets.utils.video.transforms as video_transforms
import diffusion_policy.vjepa2.datasets.utils.video.volume_transforms as volume_transforms

# from src.models.attentive_pooler import AttentiveClassifier
from diffusion_policy.vjepa2.models.vision_transformer import (
    vit_giant_xformers_rope,
    vit_large,
    vit_large_rope,
    vit_huge,
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
from omegaconf import OmegaConf, open_dict
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform


def _convert_h5_to_embeddings(
    dataset_path, policy, shape_dict, pt_video_transform, embedding_key="embedding", batch_size=32, device="cpu"
):
    # Open the HDF5 dataset
    with h5py.File(dataset_path, "r+") as h5_file:
        demos = list(h5_file["data"].keys())
        demo_keys = [int(key.split("_")[1]) for key in demos]

        # ** Delete existing embeddings if present (using safe deletion) **
        for demo_idx in demo_keys:
            demo_group = h5_file[f"data/demo_{demo_idx}"]
            if embedding_key in demo_group["obs"]:
                try:
                    print(f"Deleting existing embeddings in demo_{demo_idx}")
                    demo_group["obs"].pop(embedding_key)  # Safe deletion
                except KeyError as e:
                    print(f"Failed to delete existing embedding in demo_{demo_idx}: {e}")
                except Exception as e:
                    print(f"Unexpected error while deleting embedding in demo_{demo_idx}: {e}")

        # Prepare PyTorch dataset and dataloader
        dataset = HDF5Dataset(h5_file, demo_keys, shape_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        obs_keys = list(shape_dict.keys())

        # Process each batch and save embeddings
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating embeddings")):
                # Move batch data to the device
                # breakpoint()
                # [print(f"{key}: {batch[key].shape}") for key in obs_keys]

                # Generate embeddings
                agent_view_vide = batch["agentview_image"]  # .permute(0, 1, 3, 4, 2)
                # x_pt = pt_video_transform(agent_view_vide).cuda().unsqueeze(0)
                x_pt = torch.stack([pt_video_transform(video).to(device) for video in agent_view_vide], dim=0)
                agent_view_embeddings = policy(x_pt).cpu().numpy().squeeze()[:, 0]  # [batch 200 1024]

                robot0_eye_in_hand_image = batch["robot0_eye_in_hand_image"]  # .permute(0, 1, 3, 4, 2)
                # x_pt = pt_video_transform(robot0_eye_in_hand_image).cuda().unsqueeze(0)
                x_pt = torch.stack([pt_video_transform(video).to(device) for video in robot0_eye_in_hand_image], dim=0)
                robot0_eye_in_hand_embeddings = policy(x_pt).cpu().numpy().squeeze()[:, 0]

                embeddings = np.concatenate([agent_view_embeddings, robot0_eye_in_hand_embeddings], axis=1)
                # embeddings = agen

                # Save embeddings back into the HDF5 file
                batch_start = batch_idx * batch_size
                batch_end = batch_start + len(batch[obs_keys[0]])

                for i, (demo_idx, timestep) in enumerate(dataset.indices[batch_start:batch_end]):
                    # breakpoint()
                    demo_group = h5_file[f"data/demo_{demo_idx}"]
                    if embedding_key not in demo_group["obs"]:
                        shape = (demo_group["obs"][obs_keys[0]].shape[0],) + embeddings.shape[1:]
                        print(f"demo shape: {shape}")
                        demo_group["obs"].create_dataset(embedding_key, shape=shape, dtype=embeddings.dtype)
                    demo_group["obs"][embedding_key][timestep] = embeddings[i]

    print("Embeddings generated and saved successfully!")
    return


class HDF5Dataset(Dataset):
    def __init__(self, h5_file, demo_keys, obs_dict):
        self.h5_file = h5_file
        self.demo_keys = demo_keys
        self.obs_dict = obs_dict
        self.obs_keys = list(obs_dict.keys())
        self.indices = self._create_indices()
        self.n_frames = 16

    def _create_indices(self):
        indices = []
        for demo_idx in self.demo_keys:
            demo = self.h5_file[f"data/demo_{demo_idx}"]
            n_timesteps = demo["obs"][self.obs_keys[0]].shape[0]
            indices.extend([(demo_idx, t) for t in range(n_timesteps)])
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_idx, timestep = self.indices[idx]
        demo = self.h5_file[f"data/demo_{demo_idx}"]
        obs = {}
        for key in self.obs_keys:
            if "type" not in self.obs_dict[key] or self.obs_dict[key]["type"] != "rgb":
                obs[key] = np.expand_dims(demo["obs"][key][timestep], axis=0)
            else:
                # For RGB images, we need to handle the shape and normalization
                img = demo["obs"][key][timestep]
                video_sequence = np.zeros(shape=(self.n_frames, *img.shape), dtype=img.dtype)
                if img.ndim == 3:
                    start_time = max(0, timestep + 1 - self.n_frames)
                    pad_len = self.n_frames - (timestep + 1 - start_time)
                    video_sequence[pad_len - self.n_frames :] = demo["obs"][key][start_time : timestep + 1]
                    # obs[key] = np.expand_dims(video_sequence, axis=0)  # Add batch dimension
                    obs[key] = video_sequence
                else:
                    # obs[key] = np.expand_dims(img, axis=0)  # Add batch dimension
                    obs[key] = img

        # Normalize RGB keys
        for key in self.obs_keys:
            if "type" in self.obs_dict[key] and self.obs_dict[key]["type"] == "rgb":
                obs[key] = np.moveaxis(obs[key], -1, 1).astype(np.float32)  # / 255.0

        return obs


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-v", "--vjepa2", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-f", "--convert_file", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, vjepa2, output_dir, convert_file, device):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    img_size = 84  # Assuming img_size is defined somewhere, adjust as needed
    policy = vit_large_rope(img_size=(img_size, img_size), num_frames=16)
    policy.cuda().eval()
    load_pretrained_vjepa_pt_weights(policy, vjepa2)

    pt_video_transform = build_pt_video_transform(img_size=img_size)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    # workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    # policy = workspace.model

    # # configure dataset
    # dataset: BaseVideoDataset
    # cfg.task.dataset._target_ = "diffusion_policy.dataset.robomimic_replay_video_dataset.RobomimicReplayVideoDataset"
    # cfg.task.dataset.use_cache = False
    # cfg.shape_meta.obs["agentview_image"]["shape"] = [16, 3, img_size, img_size]
    # cfg.shape_meta.obs["robot0_eye_in_hand_image"]["shape"] = [16, 3, img_size, img_size]
    # dataset = hydra.utils.instantiate(cfg.task.dataset)
    # assert isinstance(dataset, BaseVideoDataset)
    # train_dataloader = DataLoader(dataset, **cfg.dataloader)
    # normalizer = dataset.get_normalizer()

    # policy.set_normalizer(normalizer)
    device = torch.device(device)
    # policy.to(device)

    policy.eval()

    print(f"Keys: {cfg['shape_meta']['obs']}")
    print(f"Using dataset: {cfg.task.dataset.dataset_path} to convert dataset {convert_file}")
    _convert_h5_to_embeddings(
        convert_file, policy, cfg.shape_meta.obs, pt_video_transform=pt_video_transform, batch_size=64, device=device
    )
    print("Done converting!")


if __name__ == "__main__":
    main()
