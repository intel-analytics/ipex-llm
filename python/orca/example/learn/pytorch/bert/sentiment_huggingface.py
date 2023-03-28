# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import re
import warnings

import datasets
import torch
import numpy as np
from datasets import load_dataset
from datasets.download.download_config import DownloadConfig
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
# from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository, create_repo
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from bigdl.orca.learn.pytorch.callbacks import MainCallback, Callback
from bigdl.dllib.utils.log4Error import invalidInputError


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    parser.add_argument(
        "--per_executor_batch_size",
        type=int,
        default=8,
        help="Batch size (per executor) for the training dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    # for bigdl
    parser.add_argument(
        "--cluster_mode",
        type=str,
        default="local",
        help="The mode for the Spark cluster. local, yarn or spark-submit.",
    )

    parser.add_argument(
        "--num_executors", type=int, default=2, help="Total number of executors."
    )


    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None and args.dataset_name is None:
        raise ValueError("Need either a task name, dataset name or a training/validation file.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def train_loader_creator(config, batch_size):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    download_config = DownloadConfig(resume_download=True, max_retries=3)
    if config['task_name'] is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", config['task_name'], split="train")
    elif config['dataset_name'] is not None:
        raw_datasets = load_dataset(config['dataset_name'], config['dataset_config_name'],
                                    split="train", download_config=download_config)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if config['train_file'] is not None:
            data_files["train"] = config['train_file']
        # if config['validation_file'] is not None:
        #     data_files["validation"] = config['validation_file']
        extension = config['train_file'].split(".")[-1]
        if extension == "txt":
            extension = "text"
            raw_datasets = load_dataset(extension, data_files=data_files, split="train")
        elif extension == "tsv":
            extension = "csv"
            raw_datasets = load_dataset(extension, delimiter='\t', data_files=data_files, split="train")
        else:
            raw_datasets = load_dataset(extension, data_files=data_files, split="train")
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # Load pretrained tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'],
                                              use_fast=not config['use_slow_tokenizer'],
                                              resume_download=True, max_retries=3, etag_timeout=100
                                              )


    # Preprocessing the datasets
    if config['task_name'] is not None:
        sentence1_key, sentence2_key = task_to_keys[config['task_name']]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets.column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None


    padding = "max_length" if config['pad_to_max_length'] else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=config['max_length'], truncation=True)

        if "label" in examples:
            if config['label_to_id'] is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [config['label_to_id'][l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    # with accelerator.main_process_first():
    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets.column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if config['pad_to_max_length']:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    def custom_collate(data):  # (2)
        data2 = data_collator(data).data
        target = data2['labels']
        return data2, target

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=custom_collate, batch_size=batch_size
    )
    return train_dataloader

def eval_loader_creator(config, batch_size):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    download_config = DownloadConfig(resume_download=True, max_retries=3)
    if config['task_name'] is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", config['task_name'], split="validation")
    elif config['dataset_name'] is not None:
        raw_datasets = load_dataset(config['dataset_name'], config['dataset_config_name'],
                                    split="validation", download_config=download_config)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if config['validation_file'] is not None:
            data_files["validation"] = config['validation_file']
        extension = config['validation_file'].split(".")[-1]
        if extension == "txt":
            extension = "text"
            raw_datasets = load_dataset(extension, data_files=data_files, split="validation")
        elif extension == "tsv":
            extension = "csv"
            raw_datasets = load_dataset(extension, delimiter='\t', data_files=data_files, split="validation")
        else:
            raw_datasets = load_dataset(extension, data_files=data_files, split="validation")
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'],
                                              use_fast=not config['use_slow_tokenizer'],
                                              resume_download=True, max_retries=3, etag_timeout=100)

    # Preprocessing the datasets
    if config['task_name'] is not None:
        sentence1_key, sentence2_key = task_to_keys[config['task_name']]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets.column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length" if config['pad_to_max_length'] else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=config['max_length'], truncation=True)

        if "label" in examples:
            if config['label_to_id'] is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [config['label_to_id'][l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    # with accelerator.main_process_first():
    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets.column_names,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = processed_datasets

    # DataLoaders creation:
    if config['pad_to_max_length']:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    def custom_collate(data):  # (2)
        data2 = data_collator(data).data
        target = data2['labels']
        return data2, target

    eval_dataloader = DataLoader(eval_dataset, collate_fn=custom_collate, batch_size=batch_size)
    return eval_dataloader


def optimizer_creator(model, config):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])
    return optimizer

def scheduler_creator(optimizer, config):
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(config['train_dataloader_size'] / config['gradient_accumulation_steps'])
    if config['max_train_steps'] is None:
        config['max_train_steps'] = config['num_train_epochs '] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=config['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=config['num_warmup_steps'],
        num_training_steps=config['max_train_steps'],
    )
    return lr_scheduler

def model_creator(config):
    if config['seed'] is not None:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
    model_config = AutoConfig.from_pretrained(config['model_name_or_path'],
                                              num_labels=config['num_labels'],
                                              finetuning_task=config['task_name'],
                                              resume_download=True, max_retries=3, etag_timeout=100
                                              )
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name_or_path'],
        from_tf=bool(".ckpt" in config['model_name_or_path']),
        config=model_config,
        ignore_mismatched_sizes=config['ignore_mismatched_sizes'],
        resume_download=True, max_retries=3, etag_timeout=100
    )
    if config['label_to_id'] is not None:
        model.config.label2id = config['label_to_id']
        model.config.id2label = {id: label for label, id in model_config.label2id.items()}
    elif config['task_name'] is not None and not config['is_regression']:
        model.config.label2id = {l: i for i, l in enumerate(config['label_list'])}
        model.config.id2label = {id: label for label, id in model_config.label2id.items()}

    return model

class MyMainCallback(MainCallback):
    def __init__(self,
                 train_size,
                 gradient_accumulation_steps=1,
                 ):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_size = train_size

    def before_run(self, runner):
        return

    def on_iter_forward(self, runner):
        # Forward features
        features, target = runner.batch
        output = runner.model(**features)
        runner.output = output.logits
        # Compute loss
        loss = output.loss
        runner.loss = loss / self.gradient_accumulation_steps
        # print("loss is:", runner.loss)

    def on_iter_backward(self, runner):
        runner.loss.backward()
        if runner.global_step % self.gradient_accumulation_steps == 0 or runner.global_step == self.train_size - 1:
            runner.optimizer.step()
            runner.scheduler.step()
            runner.optimizer.zero_grad()

    def on_lr_adjust(self, runner):
        pass


def main():
    args = parse_args()
    config = vars(args)

    cluster_mode = args.cluster_mode
    if cluster_mode == "local":
        init_orca_context(cluster_mode="local", cores="*", memory="4g")
        num_workers = 1
    elif cluster_mode.startswith("yarn"):
        num_executor = args.num_executors
        executor_cores = 44
        executor_memory = "20g"
        driver_cores = 20
        driver_memory = "10g"
        spark_conf = {
            "spark.executorEnv.LD_LIBRARY_PATH": "/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64/:/usr/java/jdk1.8.0_181-amd64/jre/lib/amd64/server/",
            "spark.executor.memoryOverhead": "120g"}
        sc = init_orca_context("yarn-client", cores=executor_cores,
                               num_nodes=num_executor, memory=executor_memory,
                               driver_cores=driver_cores, driver_memory=driver_memory,
                               conf=spark_conf, object_store_memory="30g", init_ray_on_spark=True,
                               include_webui=True,
                               additional_archive="/path/to/tf1_ckpt.zip#ckpt",
                               extra_python_lib="optimization.py,modeling.py,deferred_grad_optimizer.py,lamb_optimizer_v1.py")
        num_workers = num_executor
    elif cluster_mode == "spark-submit":
        sc = init_orca_context(cluster_mode="spark-submit", init_ray_on_spark=False, object_store_memory="30g",
                               include_webui=False)
        instances = sc.getConf().get("spark.executor.instances")
        num_workers = int(instances) if instances else 1
        print(f"num_workers is {num_workers}")
    else:
        print("init_orca_context failed. cluster_mode should be one of 'local', 'yarn' and \
                'spark-submit' but got " + cluster_mode)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    download_config = DownloadConfig(resume_download=True, max_retries=3)
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name, split="train")
    elif args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, split="train", download_config=download_config)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            raw_datasets = load_dataset(extension, data_files=data_files, split="train")
        elif extension == "tsv":
            extension = "csv"
            raw_datasets = load_dataset(extension, delimiter='\t', data_files=data_files, split="train")
        else:
            raw_datasets = load_dataset(extension, data_files=data_files, split="train")

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets.features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets.features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets.unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    # add to config
    config['is_regression'] = is_regression
    config['num_labels'] = num_labels
    config['label_list'] = label_list

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_config = AutoConfig.from_pretrained(args.model_name_or_path,
                                              num_labels=num_labels,
                                              finetuning_task=args.task_name,
                                              resume_download=True,
                                              max_retries=3,
                                              etag_timeout=100)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=not args.use_slow_tokenizer,
                                              resume_download=True, max_retries=3, etag_timeout=100)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=model_config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        resume_download=True, max_retries=3, etag_timeout=100
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets.column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    # add to config
    config['label_to_id'] = label_to_id

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    # with accelerator.main_process_first():
    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets.column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    def custom_collate(data):  # (2)
        data2 = data_collator(data).data
        target = data2['labels']
        # data2.pop('labels')
        return data2, target

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=custom_collate, batch_size=args.per_executor_batch_size
    )

    # add to config
    config['train_dataloader_size'] = len(train_dataloader)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Get the metric function
    # if args.task_name is not None:
    #     metric = evaluate.load("glue", args.task_name)
    # else:
    #     metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_executor_batch_size * num_workers * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per executor = {args.per_executor_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0
    starting_epoch = 0

    est = Estimator.from_torch(model=model_creator,
                               optimizer=optimizer_creator,
                               loss=nn.CrossEntropyLoss(),
                               metrics=[Accuracy()],
                               backend="spark",
                               config=config,
                               scheduler_creator=scheduler_creator,
                               use_tqdm=True,
                               model_dir=args.output_dir)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        from bigdl.orca.data.file import is_file
        if is_file(args.resume_from_checkpoint):
            print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            est.load_checkpoint(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        # else:
        #     # Get the most recent checkpoint in folder
        #     print(f"Resumed from checkpoint: {args.resume_from_checkpoint}/last.ckpt")
        #     est.load_checkpoint(os.path.join(args.resume_from_checkpoint, "last.ckpt"))
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("model_epoch=", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("model_step=", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    callbacks = [MyMainCallback(len(train_dataloader), gradient_accumulation_steps=args.gradient_accumulation_steps)]
    if checkpointing_steps is not None:
        if checkpointing_steps == "epoch":
            callbacks.append(
                ModelCheckpoint(filepath=os.path.join(args.output_dir, "model_{epoch}"),
                                save_weights_only=True
                                )
            )
        else:
            callbacks.append(
                ModelCheckpoint(filepath=os.path.join(args.output_dir, "model_{step}"),
                                save_weights_only=True,
                                by_epoch=False,
                                interval=checkpointing_steps
                                )
            )
    stats = est.fit(data=train_loader_creator,
            epochs=args.num_train_epochs,
            batch_size=total_batch_size,
            validation_data=eval_loader_creator,
            callbacks=callbacks
            )

    print("training stats: ", stats)
    res = est.evaluate(data=eval_loader_creator,
                 batch_size=total_batch_size,
                 callbacks=callbacks)
    print("evaluate result is: ", res)

    stop_orca_context()

    # for epoch in range(starting_epoch, args.num_train_epochs):
    #     model.train()
    #     if args.with_tracking:
    #         total_loss = 0
    #     for step, batch in enumerate(train_dataloader):
    #         # We need to skip steps until we reach the resumed step
    #         if args.resume_from_checkpoint and epoch == starting_epoch:
    #             if resume_step is not None and step < resume_step:
    #                 completed_steps += 1
    #                 continue
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         # We keep track of the loss at each epoch
    #         if args.with_tracking:
    #             total_loss += loss.detach().float()
    #         loss = loss / args.gradient_accumulation_steps
    #         accelerator.backward(loss)
    #         if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #             progress_bar.update(1)
    #             completed_steps += 1
    #
    #         if isinstance(checkpointing_steps, int):
    #             if completed_steps % checkpointing_steps == 0:
    #                 output_dir = f"step_{completed_steps }"
    #                 if args.output_dir is not None:
    #                     output_dir = os.path.join(args.output_dir, output_dir)
    #                 accelerator.save_state(output_dir)
    #
    #         if completed_steps >= args.max_train_steps:
    #             break
    #
    #     model.eval()
    #     samples_seen = 0
    #     for step, batch in enumerate(eval_dataloader):
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #         predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
    #         predictions, references = accelerator.gather((predictions, batch["labels"]))
    #         # If we are in a multiprocess environment, the last batch has duplicates
    #         if accelerator.num_processes > 1:
    #             if step == len(eval_dataloader) - 1:
    #                 predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
    #                 references = references[: len(eval_dataloader.dataset) - samples_seen]
    #             else:
    #                 samples_seen += references.shape[0]
    #         metric.add_batch(
    #             predictions=predictions,
    #             references=references,
    #         )
    #
    #     eval_metric = metric.compute()
    #     logger.info(f"epoch {epoch}: {eval_metric}")
    #
    #     if args.with_tracking:
    #         accelerator.log(
    #             {
    #                 "accuracy" if args.task_name is not None else "glue": eval_metric,
    #                 "train_loss": total_loss.item() / len(train_dataloader),
    #                 "epoch": epoch,
    #                 "step": completed_steps,
    #             },
    #             step=completed_steps,
    #         )
    #
    #     if args.push_to_hub and epoch < args.num_train_epochs - 1:
    #         accelerator.wait_for_everyone()
    #         unwrapped_model = accelerator.unwrap_model(model)
    #         unwrapped_model.save_pretrained(
    #             args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    #         )
    #         if accelerator.is_main_process:
    #             tokenizer.save_pretrained(args.output_dir)
    #             repo.push_to_hub(
    #                 commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
    #             )

    #     if args.checkpointing_steps == "epoch":
    #         output_dir = f"epoch_{epoch}"
    #         if args.output_dir is not None:
    #             output_dir = os.path.join(args.output_dir, output_dir)
    #         accelerator.save_state(output_dir)
    #
    # if args.with_tracking:
    #     accelerator.end_training()
    #
    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(
    #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    #     )
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #         if args.push_to_hub:
    #             repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
    #
    # if args.task_name == "mnli":
    #     # Final evaluation on mismatched validation set
    #     eval_dataset = processed_datasets["validation_mismatched"]
    #     eval_dataloader = DataLoader(
    #         eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    #     )
    #     eval_dataloader = accelerator.prepare(eval_dataloader)
    #
    #     model.eval()
    #     for step, batch in enumerate(eval_dataloader):
    #         outputs = model(**batch)
    #         predictions = outputs.logits.argmax(dim=-1)
    #         metric.add_batch(
    #             predictions=accelerator.gather(predictions),
    #             references=accelerator.gather(batch["labels"]),
    #         )
    #
    #     eval_metric = metric.compute()
    #     logger.info(f"mnli-mm: {eval_metric}")
    #
    # if args.output_dir is not None:
    #     all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
    #     with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
    #         json.dump(all_results, f)


if __name__ == "__main__":
    main()
