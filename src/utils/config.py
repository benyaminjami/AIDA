import os

import hydra
from omegaconf import OmegaConf

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def load_yaml_config(fpath: str) -> OmegaConf:
    return OmegaConf.load(fpath)


def find_last_epoch_checkpoint(ckpt_dir):
    prefix = "N-Step-Checkpoint_epoch"
    # list all files in the directory
    files = os.listdir(ckpt_dir)

    # filter the files based on their name
    matching_files = [file for file in files if file.startswith(prefix)]

    # extract the epoch number from each file name
    epoch_numbers = [int(file.split("=")[-1].split(".")[0]) for file in matching_files]

    # return the file with the highest epoch number
    if epoch_numbers:
        max_epoch_number = max(epoch_numbers)
        return os.path.join(ckpt_dir, f"{prefix}={max_epoch_number:04d}.ckpt")
    else:
        if os.path.exists(os.path.join(ckpt_dir, "last.ckpt")):
            return os.path.join(ckpt_dir, "last.ckpt")
        else:
            return None


def resolve_slurm_interupt(cfg):
    if cfg.train.get("force_restart", False):
        return
    ckpt_path = cfg.train.get("ckpt_path")
    if ckpt_path:
        ckpt_path = resolve_ckpt_path(ckpt_dir=cfg.paths.ckpt_dir, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path):
            log.info(f"Resuming checkpoint from <{ckpt_path}>")
        else:
            log.info(f"Failed to resume checkpoint from <{ckpt_path}>: file not exists. Skip.")
            ckpt_path = None

    if os.path.exists(cfg.paths.output_dir) and os.path.exists(cfg.paths.ckpt_dir):
        cfg.ckpt_path = find_last_epoch_checkpoint(cfg.paths.ckpt_dir)
        if cfg.ckpt_path:
            log.info(f"Resume from Interruption <{cfg.paths.output_dir}> <{cfg.ckpt_path}>")

    elif cfg.train.get("resume_slurm_id") and cfg.ckpt_path is None:
        resume_slurm_id = str(cfg.train.resume_slurm_id)
        cfg.ckpt_path = find_last_epoch_checkpoint(
            os.path.join(cfg.paths.slurm_dir, resume_slurm_id, "sundae-train/checkpoints")
        )
        log.info(f"Resume from Interrupted Slurm Job <{resume_slurm_id}> <{cfg.ckpt_path}>")

    if cfg.get("resume_slurm_id"):
        cfg.logger.wand.group = cfg.get("resume_slurm_id")


def resolve_ckpt_path(ckpt_dir, ckpt_path):
    # if not absolute path, it should be inferred from current working directory or ckeckpoint directory
    if not os.path.isabs(ckpt_path):
        # if ckpt_path is in cwd
        if os.path.exists(os.path.join(hydra.utils.get_original_cwd(), ckpt_path)):
            ckpt_path = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), ckpt_path))

        # or if ckpt_path is in the predefined checkpoint directory
        elif os.path.exists(os.path.join(ckpt_dir, ckpt_path)):
            ckpt_path = os.path.abspath(os.path.join(ckpt_dir, ckpt_path))

    return ckpt_path
