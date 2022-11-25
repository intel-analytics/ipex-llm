import torch
import argparse
import os
import logging
import time
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import BertTokenizer, BertModel, AdamW
import intel_extension_for_pytorch as ipex
#from tqdm.auto import tqdm

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

parser = argparse.ArgumentParser(description="PyTorch PERT Example")

parser.add_argument("--batch-size", type=int, default=16, metavar="N",
                    help="input batch size for training (default: 16)")
parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                    help="input batch size for testing (default: 1000)")
parser.add_argument("--epochs", type=int, default=1, metavar="N",
                    help="number of epochs to train (default: 10)")
parser.add_argument("--lr", type=float, default=1e-5, metavar="LR",
                    help="learning rate (default: 0.01)")
parser.add_argument("--seed", type=int, default=1, metavar="S",
                    help="random seed (default: 1)")
parser.add_argument("--dataset", type=int, default=1, metavar="D",
                    help="dataset size (default 1 * 9600)")
parser.add_argument("--save-model", action="store_true", default=False,
                    help="For Saving the current Model")
parser.add_argument("--local-only", action="store_true", default=False,
                    help="If set to true, then load model from disk")
parser.add_argument("--model-path", type=str, default="/ppml/model",
                    help="Where to load model")
    # Only for test purpose
parser.add_argument("--load-model", action="store_true", default=False,
                    help="For loading the current model")

parser.add_argument("--log-interval", type=int, default=2, metavar="N",
                    help="how many batches to wait before logging training status")

parser.add_argument("--log-path", type=str, default="",
                    help="Path to save logs. Print to StdOut if log-path is not set")

args = parser.parse_args()


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1

def is_distributed():
    return dist.is_available() and dist.is_initialized()

# Define dataset, so it is easier to load different split in the dataset
class Dataset(torch.utils.data.Dataset):
    # data_type is actually split, so that we can define dataset for train set/validate set
    def __init__(self, data_type, dataset_load):
        self.dataset_load = dataset_load
        self.data = self.load_data(data_type)

    def load_data(self, data_type):
        tmp_dataset = load_dataset(path='seamew/ChnSentiCorp', split=data_type)
        Data = {}
        # So enumerate will return a index, and  the line?
        # line is a dict, including 'text', 'label'
        if data_type == 'train':
            for i in range(self.dataset_load):
                for idx, line in enumerate(tmp_dataset):
                    sample = line
                    Data[idx + i * len(tmp_dataset)] = sample
        else:
            for idx, line in enumerate(tmp_dataset):
                sample = line
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if args.local_only:
    checkpoint = args.model_path
    tokenizer = BertTokenizer.from_pretrained(checkpoint, model_max_length=512, local_files_only=True)
else:
    checkpoint = 'hfl/chinese-pert-base'
    tokenizer = BertTokenizer.from_pretrained(checkpoint, model_max_length=512)



# Return a batch of data, which is used for training
def collate_fn(batch_samples):
    batch_text = []
    batch_label = []
    for sample in batch_samples:
        batch_text.append(sample['text'])
        batch_label.append(int(sample['label']))
    # The tokenizer will make the data to be a good format for our model to understand
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
            self.bert_encoder = BertModel.from_pretrained(checkpoint, local_files_only=True)
        else:
            self.bert_encoder = BertModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]
        logits = self.classifier(cls_vectors)
        return logits

device = 'cpu'

def train_loop(args, dataloader, model, loss_fn, optimizer, epoch, total_loss):
    finish_batch_num = (epoch-1)*len(dataloader)

    # Set to train mode
    model.train()
    total_dataset = 0
    for batch, (X, y) in enumerate(dataloader, start=1):


        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", enabled=False):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # Calculate loss
            loss = loss_fn(pred, y)
            loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_dataset += args.batch_size
        if batch % args.log_interval == 0:
            msg = "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                epoch, batch, len(dataloader),
                100. * batch / len(dataloader), loss.item())
            logging.info(msg)

    return total_loss, total_dataset

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    #correct /= size
    #correct *= WORLD_SIZE
    correct = correct / (size / WORLD_SIZE)
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct



def main():
    if args.log_path == "":
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG)
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
            filename=args.log_path)

    torch.manual_seed(args.seed)
    if should_distribute():
        print("Using distributed PyTorch with {} backend".format(
            "GLOO"), flush=True)
        dist.init_process_group(backend=dist.Backend.GLOO)


    # Load the data and dataset
    print("[INFO]Before data get loaded", flush=True)
    train_data = Dataset('train', args.dataset)
    print("######train data length:", len(train_data.data), flush=True)
    valid_data = Dataset('validation', 1)

    if is_distributed():
        train_sampler = DistributedSampler(train_data, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False, seed=args.seed)
        valid_sampler = DistributedSampler(valid_data, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False, seed=args.seed)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler)
        valid_dataloader = DataLoader(valid_data, batch_size=args.test_batch_size,collate_fn=collate_fn, sampler=valid_sampler)
    else:
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn)

    print("[INFO]Data get loaded successfully", flush=True)

    model = NeuralNetwork().to(device)
    if (args.load_model):
        model.load_state_dict('./pert.bin')

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel
        model = Distributor(model,find_unused_parameters=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.train()

    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    total_loss = 0.
    best_acc = 0.

    for t in range(args.epochs):
        print(f"Epoch {t+1}/{args.epochs + 1}\n-------------------------------")
        if is_distributed():
            train_dataloader.sampler.set_epoch(t)
            valid_dataloader.sampler.set_epoch(t)
        start = time.perf_counter()
        total_loss, total_dataset = train_loop(args, train_dataloader, model,loss_fn, optimizer, t+1, total_loss)
        end = time.perf_counter()
        print(f"Epoch {t+1}/{args.epochs + 1} Elapsed time:", end - start, flush=True)
        print(f"Epoch {t+1}/{args.epochs + 1} Processed dataset length:", total_dataset, flush=True)
        msg = "Epoch {}/{} Throughput: {: .4f}".format(t+1, args.epochs+1, 1.0 * total_dataset / (end-start))
        print(msg, flush=True)
        #valid_acc = test_loop(valid_dataloader, model, mode='Valid')

    print("[INFO]Finish all test", flush=True)

    if (args.save_model):
        torch.save(model.state_dict(), "pert.bin")
    if is_distributed():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
