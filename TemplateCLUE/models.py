from pytorch_lightning import LightningModule
from transformers import (
    AdamW,
    BartConfig,
    BartForConditionalGeneration,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from ner_metrics import *


# TODO: evaluate 怎么搞、存模型、inference
class BartModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.args = config.model_args

        self.tokenizer = BertTokenizer.from_pretrained(config.plm_name)
        self.model = BartForConditionalGeneration.from_pretrained(config.plm_name)

        self.total_steps = config.total_steps

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def _get_inputs(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone()
        labels[y[:, 1:] == pad_token_id] = -100
        inputs = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y_ids,
            "labels": labels,
        }
        return inputs

    def training_step(self, batch, batch_idx):
        inputs = self._get_inputs(batch)
        outputs = self.model(**inputs)
        loss = outputs['loss']

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = self._get_inputs(batch)
        outputs = self.model(**inputs)
        val_loss = outputs['loss']

        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]