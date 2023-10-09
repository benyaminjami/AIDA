import os
from typing import Any, List, Union
import torch
import hydra
from omegaconf import OmegaConf
from src.utils import metrics, RankedLogger
from src.tasks import TaskLitModule
from src.utils.data_utils import Alphabet
from src.models.generator import IterativeRefinementGenerator
from src.models.mpnn_cbalm import MPNNcBALM

# from byprot.utils.config import compose_config as Cfg, merge_config

from omegaconf import DictConfig
from torch import nn
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric

# from byprot.datamodules.datasets.data_utils import Alphabet


log = RankedLogger(__name__, rank_zero_only=True)


class callable_staticmethod(staticmethod):
    """Callable version of staticmethod."""

    def __call__(self, *args, **kwargs):
        return self.__func__(*args, **kwargs)


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
        self.build_generator()

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
        self.model: MPNNcBALM = hydra.utils.instantiate(self.hparams.model)
        # self.backbone = self.model.encoder  # For backbonefinetune callback

    def build_generator(self):
        self.generator = IterativeRefinementGenerator(
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
        metrics["count"] = SumMetric()
        metrics["h3_contact_acc"] = MeanMetric()
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

    # -------# Training #-------- #
    @torch.no_grad()
    @callable_staticmethod
    def inject_noise(
        tokens,
        attention_mask,
        alphabet,
        noise,
        sel_mask=None,
        sel_mask_add_prob=0.1,
        keep_idx=None,
    ):
        padding_idx = alphabet.padding_idx
        if keep_idx is None:
            keep_idx = [alphabet.cls_idx, alphabet.eos_idx]

        def get_random_text(shape):
            return torch.randint(
                alphabet.get_idx(alphabet.standard_toks[0]),
                alphabet.get_idx(alphabet.standard_toks[-1]) + 1,
                shape,
            )

        def _full_random(target_tokens):
            target_masks = target_tokens.ne(padding_idx) & attention_mask
            random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
            return (
                target_masks * random_text + ~target_masks * target_tokens,
                target_masks,
            )

        def _selected_random(target_tokens):
            target_masks = (
                target_tokens.ne(padding_idx) & attention_mask & sel_mask.ne(0)
            )
            random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
            return (
                target_masks * random_text + ~target_masks * target_tokens,
                target_masks,
            )

        def _sundae(target_tokens):
            target_masks = target_tokens.ne(padding_idx) & attention_mask
            corruption_prob_per_sequence = torch.rand((target_tokens.shape[0], 1))
            rand = torch.rand(target_tokens.shape)
            mask = (rand < corruption_prob_per_sequence).to(
                target_tokens.device
            ) & target_masks

            random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
            return mask * random_text + ~mask * target_tokens, mask

        def _selected_sundae(target_tokens):
            target_masks = (
                target_tokens.ne(padding_idx) & attention_mask & sel_mask.ne(0)
            )
            corruption_prob_per_sequence = torch.rand((target_tokens.shape[0], 1))
            rand = torch.rand(target_tokens.shape)
            mask = (rand < corruption_prob_per_sequence).to(
                target_tokens.device
            ) & target_masks

            random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
            return mask * random_text + ~mask * target_tokens, mask

        def _selected_guided_sundae(target_tokens):
            target_masks = target_tokens.ne(padding_idx) & attention_mask
            corruption_prob_per_sequence = torch.rand((target_tokens.shape[0], 1)).to(
                target_tokens.device
            )
            rand = torch.rand(target_tokens.shape).to(target_tokens.device)
            rand = rand - sel_mask * sel_mask_add_prob
            mask = (rand < corruption_prob_per_sequence).to(
                target_tokens.device
            ) & target_masks

            random_text = get_random_text(target_tokens.shape).to(target_tokens.device)
            return mask * random_text + ~mask * target_tokens, mask

        if noise == "no_noise":
            randomed_tokens, randomed_tokens_mask = tokens
        elif noise == "sundae":
            randomed_tokens, randomed_tokens_mask = _sundae(tokens)
        elif noise == "selected_sundae":
            randomed_tokens, randomed_tokens_mask = _selected_sundae(tokens)
        elif noise == "selected_guided_sundae":
            randomed_tokens, randomed_tokens_mask = _selected_guided_sundae(tokens)
        elif noise == "full_random":
            randomed_tokens, randomed_tokens_mask = _full_random(tokens)
        elif noise == "selected_random":
            randomed_tokens, randomed_tokens_mask = _selected_random(tokens)
        else:
            raise ValueError(f"Noise type ({noise}) not defined.")

        keep_mask = torch.isin(tokens, torch.tensor(keep_idx).to(tokens.device))
        prev_tokens = tokens * keep_mask + randomed_tokens * ~keep_mask
        prev_token_mask = randomed_tokens_mask & attention_mask & ~keep_mask

        return prev_tokens, prev_token_mask

    def step(self, batch, noise, noise_mask="cdr_weights"):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids
        """
        if noise_mask is None:
            noise_mask = "cdr_weights"
        noise_mask = batch["antibody"][noise_mask]
        batch_antibody = batch["antibody"]
        attention_mask = batch_antibody["attention_mask"]
        tokens = batch_antibody["tokens"]

        prev_tokens, prev_token_mask = SUNDAE.inject_noise(
            tokens, attention_mask, self.alphabet_decoder, noise, sel_mask=noise_mask
        )
        batch["antibody"]["prev_tokens"] = prev_tokens
        batch["antibody"]["prev_token_mask"] = label_mask = prev_token_mask

        logits = self.model(batch)
        loss, logging_output = self.criterion(
            logits,
            tokens,
            label_mask=label_mask if self.hparams.learning.mask_loss else None,
        )

        # Unroll steps
        for step in range(self.hparams.learning.unroll_steps):
            batch["antibody"]["prev_tokens"] = logits.argmax(dim=-1)
            logits = self.model(batch)
            loss_step, logging_output_step = self.criterion(
                logits,
                tokens,
                label_mask=label_mask if self.hparams.learning.mask_loss else None,
            )

            loss += loss_step

        if self.hparams.learning.get("avg_unroll_loss", True):
            loss /= self.hparams.learning.unroll_steps + 1
        # TODO ADD SUNDAE
        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(
            batch=batch,
            noise=self.hparams.learning.noise,
            noise_mask=self.hparams.learning.get("noise_mask", "cdr_weights"),
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
                prog_bar=True,
            )

        return {"loss": loss}

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return_dict = {}
        for task in self.hparams.generator.tasks:
            loss, logging_output = self.step(
                batch=batch,
                noise=self.hparams.generator.tasks[task].noise,
                noise_mask=self.hparams.generator.tasks[task].get("noise_mask", None),
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

    def on_validation_epoch_end(self, outputs: List[Any] = None):
        # compute metrics averaged over the whole dataset
        for i, log_key in enumerate(["val", "test"]):
            for task in self.hparams.generator.tasks:
                eval_loss = self.metrics[i][f"{task}_loss"].compute()
                self.metrics[i][f"{task}_loss"].reset()
                eval_nll_loss = self.metrics[i][f"{task}_nll_loss"].compute()
                self.metrics[i][f"{task}_nll_loss"].reset()
                eval_ppl = torch.exp(eval_nll_loss)

                self.log(
                    f"{log_key}/{task}_loss",
                    eval_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.log(
                    f"{log_key}/{task}_nll_loss",
                    eval_nll_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.log(
                    f"{log_key}/{task}_ppl",
                    eval_ppl,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                if self.stage == "fit":
                    self.metrics[i][f"{task}_ppl_best"].update(eval_ppl)
                    self.log(
                        f"val/{task}_ppl_best",
                        self.metrics[i][f"{task}_ppl_best"].compute(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

        self.predict_epoch_end(results=None)

        super().on_validation_epoch_end(outputs)

    # -------# Inference/Prediction #-------- #
    def forward(self, batch, noise, noise_mask="cdr_weights", return_ids=False):
        # In testing, remove target tokens to ensure no data leakage!
        # or you can just use the following one if you really know what you are doing:
        #   tokens = batch['tokens']
        # tokens = batch["antibody"].pop("tokens")
        if noise_mask is None:
            noise_mask = "cdr_weights"
        noise_mask = batch["antibody"][noise_mask]
        batch_antibody = batch["antibody"]
        attention_mask = batch_antibody["attention_mask"]
        tokens = batch_antibody["tokens"]
        prev_tokens, prev_token_mask = SUNDAE.inject_noise(
            tokens,
            attention_mask,
            self.alphabet_decoder,
            noise=noise,
            sel_mask=noise_mask,
        )
        batch["antibody"]["prev_tokens"] = prev_tokens
        batch["antibody"]["prev_token_mask"] = prev_token_mask

        output_tokens, output_scores = self.generator.generate(
            model=self.model,
            batch=batch,
            max_iter=self.hparams.generator.max_iter,
            strategy=self.hparams.generator.strategy,
            temperature=self.hparams.generator.temperature,
        )
        if not return_ids:
            return self.alphabet_decoder.decode(output_tokens)
        return output_tokens

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True
    ) -> Any:
        tokens = batch["antibody"]["tokens"]

        contact_mask = batch["antibody"]["h3_mask"].ne(0) & (
            batch["antibody"]["dists"] < 6.6
        )

        results = {
            "pred_tokens": {},
            "names": batch["antibody"]["names"],
            "native": batch["antibody"]["seqs"],
            "recovery": {},
            "dataloader_idx": dataloader_idx,
        }
        if log_metrics:
            self.metrics[dataloader_idx]["count"].update(len(tokens))

        for task in self.hparams.generator.tasks:
            pred_tokens = self.forward(
                batch,
                noise=self.hparams.generator.tasks[task].noise,
                noise_mask=self.hparams.generator.tasks[task].get("mask", None),
                return_ids=True,
            )

            if log_metrics:
                recovery_acc_per_sample = metrics.accuracy_per_sample(
                    pred_tokens,
                    tokens,
                    mask=batch["antibody"].get(
                        self.hparams.generator.tasks[task].get("mask")
                    ),
                )
                self.metrics[dataloader_idx][f"{task}_acc_median"].update(
                    recovery_acc_per_sample
                )

                self.metrics[dataloader_idx][f"{task}_acc"].update(
                    recovery_acc_per_sample
                )
                if task == "h3":
                    recovery_contact_acc_per_sample = metrics.accuracy_per_sample(
                        pred_tokens, tokens, mask=contact_mask
                    )
                    self.metrics[dataloader_idx][f"{task}_contact_acc"].update(
                        recovery_contact_acc_per_sample
                    )

            results["pred_tokens"][task] = pred_tokens
            results["recovery"][task] = recovery_acc_per_sample

        return results

    def predict_epoch_end(self, results: List[Any]) -> None:
        for i, log_key in enumerate(["val", "test"]):
            if self.metrics[i]["count"].compute() == 0:
                continue

            count = self.metrics[i]["count"].compute()
            self.metrics[i]["count"].reset()
            self.log(
                f"{log_key}/count", count, on_step=False, on_epoch=True, prog_bar=True
            )

            for task in self.hparams.generator.tasks:
                if task == "h3":
                    acc = self.metrics[i][f"{task}_contact_acc"].compute() * 100
                    self.metrics[i][f"{task}_contact_acc"].reset()
                    self.log(
                        f"{log_key}/{task}_contact_acc",
                        acc,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

                acc = self.metrics[i][f"{task}_acc"].compute() * 100
                self.metrics[i][f"{task}_acc"].reset()
                self.log(
                    f"{log_key}/{task}_acc",
                    acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

                acc_median = (
                    torch.median(self.metrics[i][f"{task}_acc_median"].compute()) * 100
                )
                self.metrics[i][f"{task}_acc_median"].reset()
                self.log(
                    f"{log_key}/{task}_acc_median",
                    acc_median,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

                if self.stage == "fit":
                    self.metrics[i][f"{task}_acc_best"].update(acc)
                    self.log(
                        f"{log_key}/{task}_acc_best",
                        self.metrics[i][f"{task}_acc_best"].compute(),
                        on_epoch=True,
                        prog_bar=True,
                    )

        if self.stage != "fit":
            self.save_prediction(
                results, saveto=f"./test_tau{self.hparams.generator.temperature}.fasta"
            )

    def save_prediction(self, results, saveto=None):
        save_dict = {}
        if saveto:
            saveto = os.path.abspath(saveto)
            log.info(f"Saving predictions to {saveto}...")
            fp = open(saveto, "w")
            fp_native = open("./native.fasta", "w")

        for entry in results:
            for name, prediction, native, recovery in zip(
                entry["names"],
                self.alphabet_decoder.decode(entry["pred_tokens"], remove_special=True),
                entry["native"],
                entry["recovery"],
            ):
                save_dict[name] = {
                    "prediction_h3": prediction["h3"],
                    "prediction_cdr": prediction["cdr"],
                    "prediction_full": prediction["full"],
                    "native": native,
                    "recovery": recovery,
                }
                if saveto:
                    fp.write(
                        f">name={name} | L={len(prediction['h3'])} | AAR={recovery:.2f}\n"
                    )
                    fp.write(f"h3: \t {prediction['h3']}\n")
                    fp.write(f"cdr: \t {prediction['cdr']}\n")
                    fp.write(f"full: \t {prediction['full']}\n\n")
                    fp_native.write(f">name={name}\n{native}\n\n")

        if saveto:
            fp.close()
            fp_native.close()
        return save_dict
