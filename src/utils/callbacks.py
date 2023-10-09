import os
from typing import Dict, Any, Union, Iterable, List
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from functools import partial
import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning import callbacks

# from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from torch import Tensor
from omegaconf import OmegaConf

from src.utils import RankedLogger

from pytorch_lightning.callbacks.progress.rich_progress import (
    CustomProgress,
    MetricsTextColumn,
    RichProgressBar,
)
from rich import get_console, reconfigure
from rich.text import Text


log = RankedLogger(__name__, rank_zero_only=True)


# NOTE[zzx]: modify here to display float in e-format when lower than 1e-3
def float_fmt(float_value):
    if float_value.is_integer():
        return round(float_value)
    elif float_value < 1e-3:
        return f"{float_value:.4e}"
    else:
        return round(float_value, 3)


class BetterMetricsTextColumn(MetricsTextColumn):
    """A column containing text."""

    def render(self, task) -> Text:
        if (
            self._trainer.state.fn != "fit"
            or self._trainer.sanity_checking
            or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        ):
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._tasks[self._current_task_id] = self._renderable_cache[
                    self._current_task_id
                ][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        text = ""

        for k, v in self._metrics.items():
            text += f"{k}: {float_fmt(v) if isinstance(v, float) else v} "
        return Text(text, justify="left", style=self._style)


class BetterRichProgressBar(RichProgressBar):
    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = BetterMetricsTextColumn(
                trainer, self.theme.metrics
            )
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False


class ValEveryNSteps(pl.Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.validate()


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                epoch = epoch // 10
                filename = f"{self.prefix}_epoch={epoch:04d}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class ModelCheckpoint(callbacks.ModelCheckpoint):
    CHECKPOINT_NAME_BEST = "best"

    @classmethod
    def _format_checkpoint_name(
        cls,
        filename,
        metrics: Dict[str, Tensor],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        filename = super()._format_checkpoint_name(
            filename, metrics, prefix, auto_insert_metric_name
        )
        filename = filename.replace(
            "/", "_"
        )  # avoid '/' in filename unexpectedly creates folder
        return filename

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_start(trainer, pl_module)
        trainer.callback_metrics[self.monitor] = self.best_model_score

    def _update_best_and_save(
        self,
        current: Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, Tensor],
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"), device=current.device
            )

        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates, trainer, del_filepath
        )

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        # update best checkpoint
        if self.best_model_path == filepath:
            self._save_checkpoint(
                trainer,
                self.format_checkpoint_name(
                    monitor_candidates, self.CHECKPOINT_NAME_BEST
                ),
            )

        if del_filepath is not None and filepath != del_filepath:
            trainer.strategy.remove_checkpoint(del_filepath)

    def _save_last_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_LAST
        )

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        self._save_checkpoint(trainer, filepath)
        if previous and previous != filepath:
            trainer.strategy.remove_checkpoint(previous)


class TrackNorms(pl.Callback):
    # TODO do callbacks happen before or after the method in the main LightningModule?
    # @rank_zero_only # needed?
    def on_after_training_step(
        self, batch, batch_idx, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        # Log extra metrics
        metrics = {}

        if hasattr(pl_module, "_grad_norms"):
            metrics.update(pl_module._grad_norms)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # example to inspect gradient information in tensorboard
        if OmegaConf.select(
            trainer.hparams, "trainer.track_grad_norms"
        ):  # TODO dot notation should work with omegaconf?
            norms = {}
            for name, p in pl_module.named_parameters():
                if p.grad is None:
                    continue

                # param_norm = float(p.grad.data.norm(norm_type))
                param_norm = torch.mean(p.grad.data**2)
                norms[f"grad_norm.{name}"] = param_norm
            pl_module._grad_norms = norms


class CustomBackboneFinetuning(callbacks.BackboneFinetuning):
    r"""
    Replacing the backbone attribute with customizble one
    """

    def __init__(
        self,
        backbone_module_paths: str = "backbone",
        excluded_names: List[str] = [],
        lambda_func: int = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        lambda_func = partial(linear_lambda_func, const=lambda_func)
        self.excluded_names = excluded_names
        self.backbone_module_paths = backbone_module_paths
        super().__init__(lambda_func=lambda_func, *args, **kwargs)
        self.previous_backbone_lr = [0 for _ in backbone_module_paths]

    # def state_dict(self) -> Dict[str, Any]:
    #     return {
    #         f"{self.backbone_module_path}_internal_optimizer_metadata": self._internal_optimizer_metadata,
    #         f"{self.backbone_module_path}_previous_backbone_lr": self.previous_backbone_lr,
    #     }

    # def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    #     self.previous_backbone_lr = state_dict[
    #         f"{self.backbone_module_path}_previous_backbone_lr"
    #     ]
    #     self.internal_optimizer_metadata = state_dict[
    #         f"{self.backbone_module_path}_internal_optimizer_metadata"
    #     ]

    def _rec_hasattr(self, obj, backbone_module_path):
        properties = backbone_module_path.split(".")
        for prop in properties:
            if not hasattr(obj, prop):
                return False
            obj = getattr(obj, prop)
        return True

    def _rec_getattr(self, obj, backbone_module_path):
        properties = backbone_module_path.split(".")
        for prop in properties:
            if not hasattr(obj, prop):
                return False
            obj = getattr(obj, prop)
        return obj

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if all(
            self._rec_hasattr(pl_module, backbone_module_path)
            and isinstance(self._rec_getattr(pl_module, backbone_module_path), nn.Module)
            for backbone_module_path in self.backbone_module_paths
        ):
            return super(callbacks.BackboneFinetuning, self).on_fit_start(trainer, pl_module)
        
        raise MisconfigurationException(
            f"The LightningModule should have {self.backbone_module_path} attribute"
        )

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        for backbone_module_path in self.backbone_module_paths:
            self.freeze(
                self._rec_getattr(pl_module, backbone_module_path),
                self.train_bn,
                excluded_names=self.excluded_names,
            )

    def finetune_function(
        self,
        pl_module: "pl.LightningModule",
        epoch: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Called when the epoch begins."""
        for i, unfreeze_epoch in enumerate(self.unfreeze_backbone_at_epoch):
            if epoch == unfreeze_epoch:
                current_lr = optimizer.param_groups[0]["lr"]
                initial_backbone_lr = (
                    self.backbone_initial_lr
                    if self.backbone_initial_lr is not None
                    else current_lr * self.backbone_initial_ratio_lr
                )
                self.previous_backbone_lr[i] = initial_backbone_lr
                self.unfreeze_and_add_param_group(
                    self._rec_getattr(pl_module, self.backbone_module_paths[i]),
                    optimizer,
                    initial_backbone_lr,
                    train_bn=self.train_bn,
                    initial_denom_lr=self.initial_denom_lr,
                )
                if self.verbose:
                    log.info(
                        f"Current lr: {round(current_lr, self.rounding)}, "
                        f"{self.backbone_module_paths[i]} lr: {round(initial_backbone_lr, self.rounding)}"
                    )

            elif epoch > unfreeze_epoch:
                current_lr = optimizer.param_groups[0]["lr"]
                next_current_backbone_lr = (
                    self.lambda_func(epoch + 1) * self.previous_backbone_lr[i]
                )
                next_current_backbone_lr = (
                    current_lr
                    if (self.should_align and next_current_backbone_lr > current_lr)
                    else next_current_backbone_lr
                )
                optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
                self.previous_backbone_lr[i] = next_current_backbone_lr
                if self.verbose:
                    log.info(
                        f"Current lr: {round(current_lr, self.rounding)}, "
                        f"{self.backbone_module_paths[i]} lr: {round(next_current_backbone_lr, self.rounding)}"
                    )

    @staticmethod
    def freeze(
        modules: Union[nn.Module, Iterable[Union[nn.Module, Iterable]]],
        train_bn: bool = True,
        excluded_names: List[str] = [],
    ) -> None:
        """Freezes the parameters of the provided modules.

        Args:
            modules: A given module or an iterable of modules
            train_bn: If True, leave the BatchNorm layers in training mode

        Returns:
            None

        """
        names = list(modules.named_modules())
        names = [m[0] for m in names if not list(m[1].children()) or m[1]._parameters]
        modules = callbacks.BackboneFinetuning.flatten_modules(modules)
        for i, (mod, name) in enumerate(zip(modules, names)):
            if any(excluded_name in name for excluded_name in excluded_names):
                continue
            if isinstance(mod, nn.modules.batchnorm._BatchNorm) and train_bn:
                callbacks.BackboneFinetuning.make_trainable(mod)
            else:
                callbacks.BackboneFinetuning.freeze_module(mod)


def linear_lambda_func(x, const):
    return x + const
