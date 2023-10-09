from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch


def get_optimizer(cfg, params):
    if cfg.type == "adam":
        return torch.optim.Adam(
            params=params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(
                cfg.beta1,
                cfg.beta2,
            ),
        )
    elif cfg.type == "adamw":
        return torch.optim.AdamW(
            params=params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
    else:
        raise NotImplementedError("Optimizer not supported: %s" % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.mode,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr,
            ),
            {"monitor": "val/loss", "interval": "epoch"},
        )
    elif cfg.type == "noam":
        return (
            NoamScheduler(
                optimizer,
                lr=cfg.lr,
                warmup_steps=cfg.warmup_steps,
                model_size=cfg.model_size,
                warmup_init_lr=cfg.get("warmup_init_lr"),
            ),
            {"frequency": 1, "interval": "step"},
        )
    elif cfg.type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type == "inversesqrt":
        return (
            InverseSqrtLRScheduler(
                optimizer=optimizer,
                rate=cfg.decay_rate,
                warmup_steps=cfg.warmup_steps,
            ),
            {"frequency": 1, "interval": "step"},
        )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError("Scheduler not supported: %s" % cfg.type)


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def inverse_sqrt_lr_schedule(step, warmup_steps, r):
    if step == 0:
        step = 1
    return warmup_steps**-0.5 * min(
        step**(-r) * warmup_steps ** (-0.5 + r), step * warmup_steps**-1.5
    )


class InverseSqrtLRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        rate: float = 0.5,
        warmup_steps: int = 4000,
    ) -> None:
        self.warmup_steps = warmup_steps
        rate = self.rate

        def lr_lambda(step):
            return inverse_sqrt_lr_schedule(step, self.warmup_steps, rate) 

        super().__init__(optimizer, lr_lambda=lr_lambda)


def noam_lr_schedule(step, warmup_steps, factor, model_size):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )


class NoamScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        lr,
        warmup_init_lr,
        model_size: int = 128,
        warmup_steps: int = 0,
        factor: int = 2,
    ) -> None:
        # dummy_lr = self.base_lrs[0]
        def lr_lambda(step):
            return noam_lr_schedule(step, warmup_steps, factor, model_size) / lr

        super().__init__(optimizer, lr_lambda=lr_lambda)
