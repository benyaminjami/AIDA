from typing import Any, Callable, Dict, List, Optional, Union

import torch
from src.utils.optim import get_optimizer, get_scheduler
from pytorch_lightning import LightningModule

# from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from torch import distributed as dist
from torch import nn
from torchmetrics import MaxMetric, MeanMetric, MinMetric, SumMetric

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TaskLitModule(LightningModule):
    """Example of LightningModule for sequence-to-sequence learning.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: List[nn.Module],
        criterion: nn.Module = None,
        optimizer: Union[Callable, torch.optim.Optimizer] = None,
        lr_scheduler: Union[Callable, torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=True)
        self.valid_logged = {}

    def setup(self, stage=None) -> None:
        self._stage = stage
        super().setup(stage)

    @property
    def lrate(self):
        for param_group in self.trainer.optimizers[0].param_groups:
            return param_group["lr"]

    @property
    def stage(self):
        return self._stage

    def log(
        self,
        name: str,
        value,  #: _METRIC_COLLECTION,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if on_epoch and not self.training:
            self.valid_logged[name] = value
        return super().log(name, value, prog_bar, logger, on_step, on_epoch, **kwargs)

    # -------# Training #-------- #
    def step(self, batch):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def training_step_end(
        self, step_output: Union[torch.Tensor, Dict[str, Any]]
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        return super().training_step_end(step_output)

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def validation_step_end(
        self, *args, **kwargs
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        return super().validation_step_end(*args, **kwargs)

    def on_validation_epoch_end(self, outputs: List[Any] = None):
        logging_info = ", ".join(
            f"{key}={val:.3f}" for key, val in self.valid_logged.items()
        )
        logging_info = f"Validation Info @ (Epoch {self.current_epoch}, global step {self.global_step}): {logging_info}"
        log.info(logging_info)

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def test_step_end(
        self, *args, **kwargs
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        return self.validation_step_end(*args, **kwargs)

    def on_test_epoch_end(self, outputs: List[Any]):
        return self.on_validation_epoch_end(outputs)

    # -------# Inference/Prediction #-------- #
    def forward(self, batch):
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError

    def predict_epoch_end(self, results: List[Any], log_pref=None) -> None:
        raise NotImplementedError

    # -------# Optimizers & Lr Schedulers #-------- #
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = get_optimizer(self.hparams.optimizer, self.parameters())
        if "lr_scheduler" in self.hparams and self.hparams.lr_scheduler is not None:
            lr_scheduler, extra_kwargs = get_scheduler(
                self.hparams.lr_scheduler, optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, **extra_kwargs},
            }
        return optimizer

    # -------# Others #-------- #
    def on_train_epoch_end(self) -> None:
        if dist.is_initialized() and hasattr(
            self.trainer.datamodule, "train_batch_sampler"
        ):
            self.trainer.datamodule.train_batch_sampler.set_epoch(
                self.current_epoch + 1
            )
            self.trainer.datamodule.train_batch_sampler._build_batches()

    def on_epoch_end(self):
        pass


class AutoMetric(nn.Module):
    _type_shortnames = dict(
        mean=MeanMetric,
        sum=SumMetric,
        max=MaxMetric,
        min=MinMetric,
    )

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("_device", torch.zeros(1))

    @property
    def device(self):
        return self._device.device

    def update(self, name, value, type="mean", **kwds):
        if not hasattr(self, name):
            if isinstance(type, str):
                type = self._type_shortnames[type]
            setattr(self, name, type(**kwds))

            getattr(self, name).to(self.device)

        getattr(self, name).update(value)

    def compute(self, name):
        return getattr(self, name).compute()

    def reset(self, name):
        getattr(self, name).reset()
