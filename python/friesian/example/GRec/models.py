import os
import time

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelRankingAveragePrecision

from utils import ViewDataset2
from grec import GRec


class GridwallData(LightningDataModule):
    def __init__(self, params, output_dir, local_data_dir, local_lookup_dirs):
        super().__init__()
        self.params = params
        self.output_dir = output_dir
        self.local_data_dir = local_data_dir
        self.local_lookup_dirs = local_lookup_dirs

    def setup(self, stage=None):
        self.train_dataset = ViewDataset2(
            deep_cols=self.params["train_params"]["deep_cols"],
            wide_cols=self.params["train_params"]["wide_cols"],
            pn_seq_col=self.params["train_params"]["pn_seq_col"],
            ic_seq_col=self.params["train_params"]["ic_seq_col"],
            item_meta_cols=self.params["train_params"]["item_meta_cols"],
            page_meta_wide_cols=self.params["train_params"]["page_meta_wide_cols"],
            vl_col=self.params["train_params"]["vl_col"],
            task_type_cols=self.params["train_params"]["task_type_cols"],
            current_col=self.params["train_params"]["current_col"],
            current_meta_col=self.params["train_params"]["current_meta_col"],
            search_col=self.params["train_params"]["search_col"],
            label_col=self.params["train_params"]["label_col"], 
            from_hdfs=False,
            local_data_dir=os.path.join(self.local_data_dir, "train/"),
            local_lookup_dirs=self.local_lookup_dirs,
            parts_num=self.params["train_params"]["parts_num"],
            data_file_format=self.params["train_params"]["data_file_format"],
            lookup_file_format=self.params["train_params"]["lookup_file_format"],
        )
        print(f"train_dataset len: {len(self.train_dataset)}")
        self.test_dataset = ViewDataset2(
                deep_cols=self.params["train_params"]["deep_cols"],
                wide_cols=self.params["train_params"]["wide_cols"],
                pn_seq_col=self.params["train_params"]["pn_seq_col"],
                ic_seq_col=self.params["train_params"]["ic_seq_col"],
                item_meta_cols=self.params["train_params"]["item_meta_cols"],
                page_meta_wide_cols=self.params["train_params"]["page_meta_wide_cols"],
                vl_col=self.params["train_params"]["vl_col"],
                task_type_cols=self.params["train_params"]["task_type_cols"],
                current_col=self.params["train_params"]["current_col"],
                current_meta_col=self.params["train_params"]["current_meta_col"],
                search_col=self.params["train_params"]["search_col"],
                label_col=self.params["train_params"]["label_col"], 
                from_hdfs=False,
                local_data_dir=os.path.join(self.local_data_dir, "test/"),
                local_lookup_dirs=self.local_lookup_dirs,
                parts_num=self.params["train_params"]["parts_num"],
                data_file_format=self.params["train_params"]["data_file_format"],
                lookup_file_format=self.params["train_params"]["lookup_file_format"],
            )
        self.test_dataset.export_lookups(self.output_dir, "csv")
        self.deep_dims = [int(max(self.train_dataset.get_feature_num(deep_col), self.test_dataset.get_feature_num(deep_col))) for deep_col in self.params["train_params"]["deep_cols"]]
        print(self.deep_dims)
        self.seq_num = []
        pn_seq_num = max(self.train_dataset.get_feature_num(self.params["train_params"]["pn_seq_col"]), self.test_dataset.get_feature_num(self.params["train_params"]["pn_seq_col"])) + 1
        ic_seq_num = max(self.train_dataset.get_feature_num(self.params["train_params"]["ic_seq_col"]), self.test_dataset.get_feature_num(self.params["train_params"]["ic_seq_col"]))
        icm_seq_num = max(self.train_dataset.get_feature_num(self.params["train_params"]["item_meta_cols"]), self.test_dataset.get_feature_num(self.params["train_params"]["item_meta_cols"]))
        self.seq_num.append(pn_seq_num)
        self.seq_num.append(ic_seq_num)
        self.seq_num.append(icm_seq_num)
        self.task_out_dims = []
        self.task_out_dims += 4*[ic_seq_num]
        self.loss_weights = []
        self.loss_weights += 4*[1]
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params["hparam"]["batch_size"], shuffle=True,
                          num_workers=os.cpu_count()//12, pin_memory=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.params["hparam"]["batch_size"], shuffle=True,
                          num_workers=os.cpu_count()//12, pin_memory=False, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.params["hparam"]["batch_size"], shuffle=True,
                          num_workers=os.cpu_count()//12, pin_memory=False, drop_last=True)
    

class GridwallPT(LightningModule):
    def __init__(self, config, example_input_array=None, tokenizer=None):
        super().__init__()
        self.loss_criteria = torch.nn.CrossEntropyLoss()
        self.config = config
        self.learning_rate = config["hparam"]["learning_rate"]
        self.torch_model = GRec(**config["model_kwargs"])
        self.model_dir = config["model_dir"]
        self.epoch_time = time.time()
        self._example_input_array = example_input_array
        self.k = config["k"]
        self.acc = Accuracy(task="multiclass", num_classes=config["model_kwargs"]['seq_dim'], top_k=self.k)
        self.ap = MultilabelRankingAveragePrecision(num_labels=config["model_kwargs"]['seq_dim'])
        self.tokenizer = tokenizer
        self.nlp_device = "cpu"
        self.optimizer = torch.optim.Adam
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.lr_scheduler_kwargs = config["hparam"]["lr_scheduler_kwargs"]

    def format_search(self, search_in):
        return self.tokenizer(list(search_in), return_tensors='pt', truncation=True, padding=True, max_length=32).to(self.nlp_device)
    
    def format_wide(self, wide_in):
        wide_fmt = []
        for wide in wide_in:
            pow_list = torch.pow(wide, 2)
            sqrt_list = torch.sqrt(torch.abs(wide))
            wide_fmt.extend([wide, pow_list, sqrt_list])
        return wide_fmt
        
    def forward(self, deep_in, page_in, item_in, vl_in, tasks_in, current_in, current_meta_in, wide_in, search_in, item_meta_in, page_meta_wide_in):
        search_in = self.format_search(search_in)
        input_ids, attention_mask = search_in.input_ids, search_in.attention_mask
        return self.torch_model(
            deep_in=deep_in,
            page_in=page_in,
            item_in=item_in,
            item_meta_in=item_meta_in,
            page_meta_wide_in=page_meta_wide_in,
            vl_in=vl_in,
            tasks_in=tasks_in,
            current_in=current_in,
            current_meta_in=current_meta_in,
            wide_in=wide_in,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    def forward_step(self, batch, phase, batch_idx):
        (deep_in, page_in, item_in, vl_in, tasks_in, current_in, current_meta_in, wide_in, search_in, item_meta_in, page_meta_wide_in), label_pos = batch
        (task_indices, task_outs, aux_loss) = self(
            deep_in=deep_in,
            page_in=page_in,
            item_in=item_in,
            item_meta_in=item_meta_in,
            page_meta_wide_in=page_meta_wide_in,
            vl_in=vl_in,
            tasks_in=tasks_in,
            current_in=current_in,
            current_meta_in=current_meta_in,
            wide_in=wide_in,
            search_in=search_in
        )
        y1_true = label_pos.long()
        batch_size = y1_true.shape[0]
        

        encoder_router_logits, encoder_expert_indexes = aux_loss
        z_loss = self.router_z_loss_func(encoder_router_logits) * 0.0001
        a_loss = self.load_balancing_loss_func(encoder_router_logits, encoder_expert_indexes) * 0.0001
        
        loss_list = []
        weight_list = []
        for i in range(len(task_outs)):
            task_indice = task_indices[i]
            if len(task_indice) > 0:
                y1_task_pred = task_outs[i]
                y1_task_true = y1_true[task_indice]
                balance_weight = (batch_size/float(len(task_indice))) * self.config["hparam"]["loss_weights"][i]
                loss_task = balance_weight*self.loss_criteria(y1_task_pred, y1_task_true)
                loss_list.append(loss_task)
                weight_list.append(balance_weight)
        loss = sum(loss_list)/sum(weight_list) + z_loss + a_loss

        if phase == "Validation" or (phase == "Train" and batch_idx % self.trainer.log_every_n_steps == 0):
            for i in range(len(task_outs)):
                task_indice = task_indices[i]
                if len(task_indice) > 0:
                    y1_task_pred = task_outs[i]
                    y1_task_true = y1_true[task_indice]
                    y1_task_oh = torch.nn.functional.one_hot(y1_task_true, num_classes=self.config["model_kwargs"]["task_out_dims"][i])
                    task_acc = self.acc(y1_task_pred, y1_task_true)
                    task_ap = self.ap(y1_task_pred, y1_task_oh)
                    self.log(f"task{i+1}_hr@{self.k}/{phase}", task_acc, on_step=True, on_epoch=False)
                    self.log(f"task{i+1}_ap/{phase}", task_ap, on_step=True, on_epoch=False)
        
        if phase == "Train":
            self.log(f"loss/{phase}", loss, on_step=True, on_epoch=False)
        else:
            self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True)
        
        return {"loss": loss}

    def on_train_epoch_start(self):
        print(self.config)

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, "Train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, "Validation", batch_idx)

    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler:
            lr_scheduler = self.lr_scheduler(optimizer, mode="min", verbose=True, **self.lr_scheduler_kwargs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "loss/Validation",
                    "interval": "epoch"
                },
            }
        else:
            return optimizer
    
    def router_z_loss_func(self, router_logits: torch.Tensor) -> float:
        r"""
        Compute the router z-loss implemented in PyTorch.
        The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
        It encourages router logits to remain small in an effort to improve stability.
        Args:
            router_logits (`float`):
                Input logits of shape [batch_size, sequence_length, num_experts]
        Returns:
            Scalar router z-loss.
        """
        num_groups, tokens_per_group, _ = router_logits.shape
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = log_z**2
        return torch.sum(z_loss) / (num_groups * tokens_per_group)

    @staticmethod
    def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
        r"""
        Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
        See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
        function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
        experts is too unbalanced.
        Args:
            router_probs (`torch.Tensor`):
                Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
            expert_indices (`torch.Tensor`):
                Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.
        Returns:
            The auxiliary loss.
        """
        num_experts = router_probs.shape[-1]

        # cast the expert indices to int64, otherwise one-hot encoding will fail
        if expert_indices.dtype != torch.int64:
            expert_indices = expert_indices.to(torch.int64)

        if len(expert_indices.shape) == 2:
            expert_indices = expert_indices.unsqueeze(2)

        expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

        # For a given token, determine if it was routed to a given expert.
        expert_mask = torch.max(expert_mask, axis=-2).values

        # cast to float32 otherwise mean will fail
        expert_mask = expert_mask.to(torch.float32)
        tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

        router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
        return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)
