import os
from typing import Any, Union

import json
import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric

from src.models.generator import IterativeRefinementGenerator, sample_from_categorical
from src.models.optimizer import AntibodyOptimizer
# from src.models.optimizer_sundae import AntibodyOptimizer
from src.tasks import TaskLitModule
from src.utils import RankedLogger, metrics
from src.utils.utils import dictoflist_to_listofdict
from src.utils.data_utils import Alphabet
from src.tasks.utils import inject_noise, resolve_noise_mask
log = RankedLogger(__name__, rank_zero_only=True)


class SUNDAE(TaskLitModule):
    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        alphabet: DictConfig,
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning: DictConfig = None,
        generator: DictConfig = None,
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)

        self.alphabet_encoder = Alphabet(**alphabet.encoder)
        self.alphabet_decoder = Alphabet(**alphabet.decoder)
        self.generator = generator
        self.build_model()
        self.build_generator_optimizer()

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.metrics = nn.ModuleList(
            [self.build_torchmetric(), self.build_torchmetric()]
        )

        if self.stage == "fit":
            log.info(f"\n{self.model}")

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = hydra.utils.instantiate(self.hparams.model)

    def build_generator_optimizer(self):
        if not self.hparams.generator.get('optimize', False):
            self.generator = IterativeRefinementGenerator(
                alphabet=self.alphabet_decoder, **self.hparams.generator
            )
        else:
            self.generator = AntibodyOptimizer(
                alphabet=self.alphabet_decoder, **self.hparams.generator
            )

    def build_criterion(self):
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)

    def build_torchmetric(self):
        metrics = {}
        for task in self.hparams.generator.tasks:
            metrics[f"{task}_loss"] = MeanMetric()
            metrics[f"{task}_nll_loss"] = MeanMetric()
            metrics[f"{task}_ppl_best"] = MinMetric()
            metrics[f"{task}_acc"] = MeanMetric()
            metrics[f"{task}_acc_best"] = MaxMetric()
            metrics[f"{task}_acc_median"] = CatMetric()
            metrics[f"{task}_contact_acc"] = MeanMetric()
        metrics["count"] = SumMetric()
        return nn.ModuleDict(metrics)

    def load_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(
            f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def step(self, batch, task):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids
        """
        noise_mask = task.get("mask", None)
        if noise_mask is None:
            noise_mask = batch["antibody"]["cdr_weights"]
        else:
            noise_mask = resolve_noise_mask(batch['antibody'], noise_mask)

        prev_tokens, prev_token_mask = inject_noise(
            batch["antibody"]["tokens"],
            batch["antibody"]["attention_mask"],
            self.alphabet_decoder,
            task.noise,
            sel_mask=noise_mask,
            keep_special_tokens=task.get('keep_special_tokens', False),
        )
        batch["antibody"]["prev_tokens"] = prev_tokens
        batch["antibody"]["prev_token_mask"] = label_mask = prev_token_mask

        logits = self.model(batch)
        loss, logging_output = self.criterion(
            logits,
            batch["antibody"]["tokens"],
            label_mask=label_mask if task.get("mask_loss", True) else None,
        )

        # Unroll steps
        for step in range(task.get("unroll_steps", 0)):
            batch["antibody"]["prev_tokens"] = (
                sample_from_categorical(logits.detach())[0]
            )
            logits = self.model(batch)
            loss_step, logging_output_step = self.criterion(
                logits,
                batch["antibody"]["tokens"],
                label_mask=label_mask if task.get("mask_loss", True) else None,
            )
            loss += loss_step

        if self.hparams.learning.get("avg_unroll_loss", True):
            loss /= task.get("unroll_steps", 0) + 1

        # TODO ADD SUNDAE
        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(
            batch=batch,
            task=self.hparams.learning,
        )

        # log train metrics
        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("lr", self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(
                f"train/{log_key}",
                log_value,
                on_step=True,
                on_epoch=False,
            )

        return {"loss": loss}

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return_dict = {}
        for task in self.hparams.generator.tasks:

            loss, logging_output = self.step(
                batch=batch,
                task=self.hparams.generator.tasks[task]
            )

            return_dict[f"{task}_loss"] = loss

            # log other metrics
            sample_size = logging_output["sample_size"]
            self.metrics[dataloader_idx][f"{task}_loss"].update(
                logging_output["loss"], weight=sample_size
            )
            self.metrics[dataloader_idx][f"{task}_nll_loss"].update(
                logging_output["nll_loss"], weight=sample_size
            )

        if self.stage == "fit":
            _ = self.predict_step(
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )
        return return_dict

    def on_validation_epoch_end(self):
        # compute metrics averaged over the whole dataset
        for i, log_key in enumerate(["val", "test"]):
            for task in self.hparams.generator.tasks:
                if self.metrics[i][f"{task}_loss"]._update_count == 0:
                    continue
                eval_loss = self.metrics[i][f"{task}_loss"].compute()
                self.metrics[i][f"{task}_loss"].reset()
                eval_nll_loss = self.metrics[i][f"{task}_nll_loss"].compute()
                self.metrics[i][f"{task}_nll_loss"].reset()
                eval_ppl = torch.exp(eval_nll_loss)

                self.log(f"{log_key}/{task}_loss", eval_loss)
                self.log(f"{log_key}/{task}_nll_loss", eval_nll_loss)
                self.log(f"{log_key}/{task}_ppl", eval_ppl)

                if self.stage == "fit":
                    self.metrics[i][f"{task}_ppl_best"].update(eval_ppl)
                    self.log(f"val/{task}_ppl_best", self.metrics[i][f"{task}_ppl_best"].compute())
        if self.stage == "fit":
            self.on_predict_epoch_end()

        super().on_validation_epoch_end()

    # -------# Inference/Prediction #-------- #
    def forward(self, batch, noise, noise_mask, return_ids=False):
        # In testing, remove target tokens to ensure no data leakage!
        # or you can just use the following one if you really know what you are doing:
        # tokens = batch['tokens']
        tokens = batch["antibody"]["tokens"]
        batch_antibody = batch["antibody"]
        attention_mask = batch_antibody["attention_mask"]
        batch_antibody['noise_mask'] = noise_mask
        prev_tokens, prev_token_mask = inject_noise(
            tokens,
            attention_mask,
            self.alphabet_decoder,
            noise=noise,
            sel_mask=noise_mask,
            # mask_special_tokens=False
        )
        if self.hparams.generator.get("use_T5_initialization", False):
            t5_prev_tokens = batch["antibody"].get("prev_tokens", None)
            if t5_prev_tokens is None:
                log.error(
                    "Could not find T5 initialization, continue with random initialization"
                )
            else:
                prev_tokens = t5_prev_tokens
        batch["antibody"]["prev_tokens"] = prev_tokens
        batch["antibody"]["prev_token_mask"] = prev_token_mask

        output_tokens, output_scores, history = self.generator.generate(
            model=self.model,
            batch=batch,
            max_iter=self.hparams.generator.max_iter,
            strategy=self.hparams.generator.strategy,
            temperature=self.hparams.generator.temperature,
            n_samples=self.hparams.generator.get('n_samples', 1),
        )
        if not return_ids:
            return self.alphabet_decoder.decode(output_tokens)
        return output_tokens, history, prev_token_mask

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True
    ) -> Any:
        tokens = batch["antibody"]["tokens"]

        results = {
            "pred_tokens": {},
            "names": batch["antibody"]["names"],
            "native": batch["antibody"]["tokens"],
            "recovery": {},
            "dataloader_idx": dataloader_idx,
            "history": {},
            "masks": {},
        }
        if log_metrics:
            self.metrics[dataloader_idx]["count"].update(len(tokens))

        recovery_acc_per_sample = torch.tensor([100])
        for task in self.hparams.generator.tasks:
            noise_mask = self.hparams.generator.tasks[task].get("mask", None)
            if noise_mask is None:
                noise_mask = batch["antibody"]["cdr_weights"]
            else:
                noise_mask = resolve_noise_mask(batch["antibody"], noise_mask)
            pred_tokens, history, generated_mask = self.forward(
                batch,
                noise=self.hparams.generator.tasks[task].noise,
                noise_mask=noise_mask,
                return_ids=True,
            )
            if log_metrics:
                tokens = batch["antibody"]["tokens"]
                mask = batch["antibody"]["prev_token_mask"]
                n_samples = self.hparams.generator.get('n_samples', 1)

                if n_samples != 1:
                    tokens = tokens.repeat(n_samples, 1)
                    mask = mask.repeat(n_samples, 1)

                recovery_acc_per_sample = metrics.accuracy_per_sample(
                    pred_tokens, tokens, mask=mask
                )
                if n_samples != 1:
                    recovery_acc_per_sample = [recovery_acc_per_sample.mean()]

                self.metrics[dataloader_idx][f"{task}_acc_median"].update(
                    recovery_acc_per_sample
                )

                self.metrics[dataloader_idx][f"{task}_acc"].update(
                    recovery_acc_per_sample
                )
                if self.hparams.generator.tasks[task].get("contact", False):
                    contact_mask = batch["antibody"]["prev_token_mask"] & (batch["antibody"]["dists"] < 6.6)
                    if n_samples != 1:
                        contact_mask = contact_mask.repeat(n_samples, 1)
                    recovery_contact_acc_per_sample = metrics.accuracy_per_sample(
                        pred_tokens, tokens, mask=contact_mask
                    )
                    if n_samples != 1:
                        recovery_contact_acc_per_sample = [recovery_contact_acc_per_sample.mean()]

                    self.metrics[dataloader_idx][f"{task}_contact_acc"].update(
                        recovery_contact_acc_per_sample
                    )

            results["recovery"][task] = recovery_acc_per_sample
            results["pred_tokens"][task] = pred_tokens
            results["history"][task] = history
            results["masks"][task] = generated_mask
        return results

    def on_predict_epoch_end(self) -> None:
        for i, log_key in enumerate(["val", "test"]):
            if self.metrics[i]["count"]._update_count == 0:
                continue

            count = self.metrics[i]["count"].compute()
            self.metrics[i]["count"].reset()
            self.log(
                f"{log_key}/count", count, on_step=False, on_epoch=True, prog_bar=True
            )

            for task in self.hparams.generator.tasks:
                if self.hparams.generator.tasks[task].get("contact", False):
                    acc = self.metrics[i][f"{task}_contact_acc"].compute() * 100
                    self.metrics[i][f"{task}_contact_acc"].reset()
                    self.log(
                        f"{log_key}/{task}_contact_acc",
                        acc
                    )

                acc = self.metrics[i][f"{task}_acc"].compute() * 100
                self.metrics[i][f"{task}_acc"].reset()
                self.log(
                    f"{log_key}/{task}_acc",
                    acc
                )

                acc_median = (
                    torch.median(self.metrics[i][f"{task}_acc_median"].compute()) * 100
                )
                self.metrics[i][f"{task}_acc_median"].reset()
                self.log(
                    f"{log_key}/{task}_acc_median",
                    acc_median
                )

                if self.stage == "fit":
                    self.metrics[i][f"{task}_acc_best"].update(acc)
                    self.log(
                        f"{log_key}/{task}_acc_best",
                        self.metrics[i][f"{task}_acc_best"].compute(),
                    )

        if self.stage != "fit":
            self.save_prediction(
                self.predict_outputs[0],
                saveto=f"./test_iter_{self.hparams.generator.max_iter}_temp_{self.hparams.generator.temperature}.json",
            )
        return self.predict_outputs

    def save_prediction(self, results, saveto=None):
        if saveto:
            saveto = os.path.abspath(saveto)
            log.info(f"Saving predictions to {saveto}...")
            fp = open(saveto, "w")

        for entry in results:
            for name, prediction, native, recovery in zip(
                entry["names"],
                dictoflist_to_listofdict(
                    {
                        task: self.alphabet_decoder.decode(entry["pred_tokens"][task])
                        for task in entry["pred_tokens"]
                    }, self.hparams.generator.get('optimize', False) or self.hparams.generator.get('n_samples', 1) > 1
                ),
                self.alphabet_decoder.decode(entry["native"]),
                dictoflist_to_listofdict(entry["recovery"]),
            ):
                save_dict = {
                    "pdb": name,
                    "native": native.replace("<pad>", "")
                    .replace("<cls>", "")
                    .replace("<eos>", ":")[:-1],
                    "recovery": {k: recovery[k].item() for k in recovery},
                }
                clean_seq = lambda x: x.replace("<pad>", "").replace("<cls>", "").replace("<eos>", ":")[:-1]
                save_dict.update(
                    {
                        f"prediction_{task}": clean_seq(prediction[task]) if not isinstance(prediction[task], list)
                        else list(set([clean_seq(s) for s in prediction[task]]))
                        for task in prediction
                    }
                )
                if saveto:
                    save_json = json.dumps(save_dict)
                    fp.write(save_json + "\n")

        if saveto:
            fp.close()
        return save_dict
