"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output -p perturbation
"""

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


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-p", "--force_perturbs", default=None)
@click.option("-d", "--device", default="cuda:0")
@click.option("-n", "--num_samples", default=1)
def main(checkpoint, output_dir, force_perturbs, device, num_samples):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if force_perturbs:
        perturb_cfg = OmegaConf.load(force_perturbs)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    if force_perturbs and "chunk" in force_perturbs:
        policy.n_action_steps = perturb_cfg["chunk"]
        with open_dict(cfg):
            cfg.task.env_runner.n_action_steps = policy.n_action_steps
        print(policy.n_action_steps, "cfg: ", cfg.task.env_runner.n_action_steps)

    # rewrite config for env_runner
    if force_perturbs:
        with open_dict(cfg):
            print(perturb_cfg)
            cfg.task.env_runner.n_samples = num_samples
            cfg.task.env_runner.perturbations = perturb_cfg
            cfg.task.env_runner.n_test = 150  # many evals for lower variance

    # run eval
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    print(runner_log)
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    print(json_log)
    import IPython

    IPython.embed()
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
