#
# Copyright 2016 The BigDL Authors.
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
#

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from contextlib import nullcontext
import torch.nn.functional as F
import os
import torch.nn as nn
import time
from bigdl.nano.pytorch.patching import patch_encryption
from bigdl.nano.pytorch.encryption import EncryptedDataset
import logging
from bigdl.ppml.kms.ehsm.client import generate_primary_key, generate_data_key, get_data_key_plaintext
from datasets.load import load_from_disk
import argparse
from transformers import BertTokenizer, BertModel, AdamW

parser = argparse.ArgumentParser(description="PyTorch PERT Example")
parser.add_argument("--local-only", action="store_true", default=False,
                    help="If set to true, then load model from disk")
parser.add_argument("--model-path", type=str, default="/ppml/model",
                    help="Where to load model")
parser.add_argument("--dataset-path", type=str, default="/ppml/dataset",
                    help="Where to load original dataset")


# python3 load_save_encryption_ex.py --local-only --model-path /ppml/model --dataset-path /ppml/save-datasets/train/
args = parser.parse_args()


# Define APPID and APIKEY in os.environment
APPID = os.environ.get('APPID')
APIKEY = os.environ.get('APIKEY')


encrypted_primary_key_path = ""
encrypted_data_key_path = ""

EHSM_IP = os.environ.get('ehsm_ip')
EHSM_PORT = os.environ.get('ehsm_port', "9000")

if args.local_only:
    checkpoint = args.model_path
    tokenizer = BertTokenizer.from_pretrained(
        checkpoint, model_max_length=512, local_files_only=True)
else:
    checkpoint = 'hfl/chinese-pert-base'
    tokenizer = BertTokenizer.from_pretrained(checkpoint, model_max_length=512)


# prepare environment
def prepare_env():
    """
    |1. Check whether arguments required by KMS is set.
    |2. Apply patch to torch.
    """
    if APPID is None or APIKEY is None or EHSM_IP is None:
        print("Please set environment variable APPID, APIKEY, ehsm_ip!")
        exit(1)
    generate_primary_key(EHSM_IP, EHSM_PORT)
    global encrypted_primary_key_path
    encrypted_primary_key_path = "./encrypted_primary_key"
    generate_data_key(EHSM_IP, EHSM_PORT, encrypted_primary_key_path, 32)
    global encrypted_data_key_path
    encrypted_data_key_path = "./encrypted_data_key"
    patch_encryption()


# Get a key from kms that can be used for encryption/decryption
def get_key():
    return get_data_key_plaintext(EHSM_IP, EHSM_PORT, encrypted_primary_key_path, encrypted_data_key_path)


def save_encrypted_dataset(dataset_path, save_path, secret_key):
    dataset = load_from_disk(dataset_path, keep_in_memory=True)
    # This will save the encrypted dataset into disk
    torch.save(dataset, save_path, encryption_key=secret_key)


def collate_fn(batch_samples):
    batch_text = []
    batch_label = []
    for sample in batch_samples:
        batch_text.append(sample['text'])
        batch_label.append(int(sample['label']))
    # The tokenizer will make the data to be in good format for our model to understand
    X = tokenizer(
        batch_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        if args.local_only:
            self.bert_encoder = BertModel.from_pretrained(
                checkpoint, local_files_only=True)
        else:
            self.bert_encoder = BertModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]
        logits = self.classifier(cls_vectors)
        return logits


device = 'cpu'


def train_loop(dataloader, model, loss_fn, optimizer, epoch, total_loss):
    # Set to train mode
    model.train()
    total_dataset = 0
    optimizer.zero_grad(set_to_none=True)
    enumerator = enumerate(dataloader, start=1)
    for batch, (X, y) in enumerator:

        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_dataset += 16
        if batch % 20 == 0:
            msg = "Train Epoch: {} loss={:.4f}".format(epoch, loss.item())
            logging.info(msg)

    return total_loss, total_dataset


def main():
    # Logging setting.
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.DEBUG)

    prepare_env()
    # WARNING: This is only safe in sgx environment, where the memory can not be read
    secret_key = get_key()

    encrypted_dataset_path = "/ppml/encryption_dataset.pt"

    # Assume we are in customer environment when executing this(which is safe and trusted)
    save_encrypted_dataset(args.dataset_path, encrypted_dataset_path, secret_key)

    # Now we have the encrypted dataset, we can safely distribute it into
    # untrusted environments.

    # load the encrypted dataset back and ready for training
    train_dataset = EncryptedDataset(encrypted_dataset_path, secret_key)
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    print("[INFO]Data get loaded successfully", flush=True)

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), 0.01)
    total_loss = 0.

    print(f"training begin\n-------------------------------")
    start = time.perf_counter()
    total_loss, total_dataset = train_loop(
        train_dataloader, model, loss_fn, optimizer, 1, total_loss)
    end = time.perf_counter()
    print(f"Elapsed time:", end - start, flush=True)
    print(f"Processed dataset length:", total_dataset, flush=True)
    msg = "Throughput: {: .4f}".format(1.0 * total_dataset / (end-start))
    print(msg, flush=True)


if __name__ == "__main__":
    main()
