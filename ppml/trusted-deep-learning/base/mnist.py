from __future__ import print_function

import argparse
import logging
import os
import time

from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

RANK = int(os.environ.get("RANK", 0))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            msg = "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item())
            logging.info(msg)
            niter = epoch * len(train_loader) + batch_idx


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logging.info("{{metricName: accuracy, metricValue: {:.4f}}};{{metricName: loss, metricValue: {:.4f}}}\n".format(
        float(correct) / (len(test_loader.dataset) / WORLD_SIZE), test_loss))


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M",
                        help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log-path", type=str, default="",
                        help="Path to save logs. Print to StdOut if log-path is not set")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")

    if dist.is_available():
        parser.add_argument("--backend", type=str, help="Distributed backend",
                            choices=[dist.Backend.GLOO,
                                     dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
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

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print("Using distributed PyTorch with {} backend".format(
            args.backend), flush=True)
        dist.init_process_group(backend=args.backend)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    print("Before downloading data", flush=True)
    train_data = datasets.FashionMNIST("./data",
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor()
                            ]))


    test_data = datasets.FashionMNIST("./data",
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor()
                            ]))
    if is_distributed():
        train_sampler = DistributedSampler(train_data, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False, seed=args.seed)
        test_sampler = DistributedSampler(test_data, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False, seed=args.seed)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,sampler=train_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    print("After downloading data", flush=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data",
                              train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor()
                              ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel
        model = Distributor(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)


    start = time.perf_counter()
    cpu_start = time.process_time()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    cpu_end = time.process_time()
    end = time.perf_counter()
    print("CPU Elapsed time:", cpu_end - cpu_start)
    print("Elapsed time:", end - start)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
