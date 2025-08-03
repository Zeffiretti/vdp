from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
import torchvision.transforms as T
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)
from math import ceil, floor
from mlp_correlation import batch_mlp_corr

register_codecs()


class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        use_embed_if_present=False,
        n_obs_steps=None,
        abs_action=False,
        rot_rep_orig="axis_angle",
        rotation_rep="rotation_6d",  # ignored when abs_action=False
        use_legacy_normalizer=False,
        subsample_frames=1,
        subsampling_method="uniform",
        use_cache=False,
        seed=42,
        val_ratio=0.0,
    ):
        from_convention = None
        if rot_rep_orig == "euler_angles":
            from_convention = "XYZ"

        rotation_transformer = RotationTransformer(
            from_rep=rot_rep_orig, to_rep=rotation_rep, from_convention=from_convention
        )

        self.use_embed_if_present = use_embed_if_present
        self.subsample_frames = subsample_frames
        self.subsampling_method = subsampling_method
        self.image_transforms = transform = T.Compose(
            [
                T.ColorJitter(
                    brightness=0.2, contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=0.05
                ),  # Adjust brightness, contrast, saturation, hue
            ]
        )

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + ".zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim" and key != "embedding":
                lowdim_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps * subsample_frames
            key_first_k["embedding"] = n_obs_steps * subsample_frames

        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask

        keys = list(replay_buffer.keys())
        if self.use_embed_if_present and "embedding" in keys:
            keys = ["embedding", "action"]

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=(horizon - n_obs_steps) + n_obs_steps * subsample_frames,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        print("subsample frames dataset", subsample_frames)
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

        actions = np.array(self.replay_buffer["action"])[None]
        corr = batch_mlp_corr(actions)
        print(actions.shape, {"expert_actions_corr": corr})

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        keys = list(self.replay_buffer.keys())
        if self.use_embed_if_present and "embedding" in keys:
            keys = ["embedding", "action"]
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=(self.horizon - self.n_obs_steps) + self.n_obs_steps * self.subsample_frames,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=keys,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if (
                key.endswith("pos")
                or key.endswith("position")
                or key == "past_act"
                or key == ("EE_EULER")
                or key == "EE_POS"
                or key == "GRIPPER"
            ):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.array(self.replay_buffer["action"]))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)
        obs_dict = dict()
        rgb_keys = self.rgb_keys
        lowdim_keys = self.lowdim_keys

        if self.use_embed_if_present:
            rgb_keys = []
            lowdim_keys = ["embedding"]

        for key in rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32))
            subsampled = data[key]
            past_data = subsampled[: self.n_obs_steps * self.subsample_frames]
            if self.subsampling_method == "uniform":
                past_data = past_data[self.subsample_frames - 1 :: self.subsample_frames]
            elif self.subsampling_method == "mixed":
                sparse = past_data[self.subsample_frames - 1 :: self.subsample_frames][-floor(self.n_obs_steps / 2) :]
                dense = past_data[-self.subsample_frames :][-ceil(self.n_obs_steps / 2) :]
                past_data = np.concatenate([sparse, dense])
            else:
                raise NotImplementedError

            # past_data = past_data[self.subsample_frames-1::self.subsample_frames]
            comb_data = past_data
            obs_dict[key] = (
                self.image_transforms(torch.from_numpy(np.moveaxis(comb_data[T_slice], -1, 1)).type(torch.uint8)).type(
                    torch.float32
                )
                / 255.0
            ).numpy()
            # T,C,H,W
            del data[key]
        for key in lowdim_keys:
            subsampled = data[key]
            past_data = subsampled[: self.n_obs_steps * self.subsample_frames]
            if self.subsampling_method == "uniform":
                past_data = past_data[self.subsample_frames - 1 :: self.subsample_frames]
            elif self.subsampling_method == "mixed":
                sparse = past_data[self.subsample_frames - 1 :: self.subsample_frames][-floor(self.n_obs_steps / 2) :]
                dense = past_data[-self.subsample_frames :][-ceil(self.n_obs_steps / 2) :]
                past_data = np.concatenate([sparse, dense])
            else:
                raise NotImplementedError

            comb_data = past_data
            obs_dict[key] = comb_data[T_slice].astype(np.float32)
            del data[key]

        subsampled = data["action"]
        future_data = subsampled[-(self.horizon - self.n_obs_steps) :]
        past_data = subsampled[: -(self.horizon - self.n_obs_steps)]
        past_data = past_data[self.subsample_frames - 1 :: self.subsample_frames]
        comb_data = np.concatenate([past_data, future_data])
        data["action"] = comb_data
        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store, shape_meta, dataset_path, abs_action, rotation_transformer, n_workers=None, max_inflight_tasks=None
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file["data"]
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            if "action" in demo.keys():
                act_key = "action"
            else:
                act_key = "actions"
            episode_length = demo[act_key].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array("episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True)
        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = act_key
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]
                # print("observations present", demo["obs"].keys())
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data, abs_action=abs_action, rotation_transformer=rotation_transformer
                )
                print(n_steps, tuple(shape_meta["action"]["shape"]), this_data.shape)
                assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])
            else:
                print(n_steps, tuple(shape_meta["action"]["shape"]), this_data.shape)

                assert this_data.shape == (n_steps,) + tuple(shape_meta["obs"][key]["shape"])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype,
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False

        with tqdm(total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=this_compressor,
                        dtype=np.uint8,
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f"demo_{episode_idx}"]
                        hdf5_arr = demo["obs"][key]  # [:,0]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)

    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)
