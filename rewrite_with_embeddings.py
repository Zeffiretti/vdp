import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import sys

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


def _convert_h5_to_embeddings(dataset_path, policy, shape_dict, embedding_key="embedding", batch_size=32, device="cpu"):
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
        demo_keys.sort()
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
                embeddings = policy.embed_observation(batch).cpu().numpy().squeeze()

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
        obs = {key: np.expand_dims(demo["obs"][key][timestep], axis=0) for key in self.obs_keys}

        # Normalize RGB keys
        for key in self.obs_keys:
            if "type" in self.obs_dict[key] and self.obs_dict[key]["type"] == "rgb":
                obs[key] = np.moveaxis(obs[key], -1, 1).astype(np.float32) / 255.0

        return obs


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-f", "--convert_file", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, convert_file, device):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model

    # configure dataset
    dataset: BaseImageDataset
    cfg.task.dataset.use_cache = False
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()

    policy.set_normalizer(normalizer)
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    print(f"Keys: {cfg['shape_meta']['obs']}")
    print(f"Using dataset: {cfg.task.dataset.dataset_path} to convert dataset {convert_file}")
    _convert_h5_to_embeddings(convert_file, policy, cfg.shape_meta.obs, batch_size=64, device=device)
    print("Done converting!")


if __name__ == "__main__":
    main()
