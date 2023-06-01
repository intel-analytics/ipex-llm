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


def parse_args(args):
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=2, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M",
                        help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Where to train model, default is cpu")
    parser.add_argument("--model-save-path", type=str, default="",
                        help="For Saving the current Model")
    return parser.parse_args(args)


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


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            msg = "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item())
            print(msg, flush=True)
            niter = epoch * len(train_loader) + batch_idx


def test(model, device, test_loader, epoch):
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
    msg = "{{metricName: accuracy, metricValue: {:.4f}}};{{metricName: loss, metricValue: {:.4f}}}\n".format(
        float(correct) / (len(test_loader.dataset) / WORLD_SIZE), test_loss)
    print(msg, flush=True)


def main(args=None):
    # Training settings
    args = parse_args(args)
    print(args)


    torch.manual_seed(args.seed)

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

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    print("After downloading data", flush=True)


    model = Net().to(args.device)


    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)


    start = time.perf_counter()
    cpu_start = time.process_time()

    # do_train
    print("mnist.test ----------------")
    import ppml_context
    mycontext = ppml_context.PPMLContext("k8s")
    model, train_dataloader, test_dataloader = mycontext.set_training(model, train_dataloader, test_dataloader)
    for epoch in range(1, args.epochs + 1):
        train(model, args.device, train_dataloader, optimizer, epoch)
        test(model, args.device, test_dataloader, epoch)

    cpu_end = time.process_time()
    end = time.perf_counter()
    print("CPU Elapsed time:", cpu_end - cpu_start)
    print("Elapsed time:", end - start)

    if args.model_save_path != "":
        torch.save(model.state_dict(), args.model_save_path)


if __name__ == "__main__":
    import os
    import sys
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import ppml_context
    local_conf = ppml_context.PPMLConf(k8s_enabled = False) \
        .set("epoch", "2") \
        .set("test-batch-size", "16") \
        .set("batch-size", "16") \
        .set("model-save-path", "/ppml/model/mnist_cnn.pt") \
        .set("momentum", "0.5")

    args1=local_conf.conf_to_args()
    main(args1)
    sys.exit()

# +
import ppml_context
k8s_conf = ppml_context.PPMLConf(k8s_enabled = True, sgx_enabled=False) \
    .set("bigdl.ppml.k8s.env.GLOO_TCP_IFACE", "ens803f0") \
    .set("bigdl.ppml.k8s.env.HF_DATASETS_OFFLINE", "1") \
    .set("bigdl.ppml.k8s.conf.nnodes", "2") \
    .set("bigdl.ppml.k8s.conf.pod_cpu", "13") \
    .set("bigdl.ppml.k8s.conf.pod_memory", "64G") \
    .set("bigdl.ppml.k8s.conf.pod_epc_memory", "68719476736")

# set("bigdl.ppml.k8s.volume_nfs.volume_name,host_path")
# set("bigdl.ppml.k8s.volume_nfs.volume_name, nfs_server, nfs_path")
# set("bigdl.ppml.k8s.volume_mount.mount_path,volume_name")
k8s_conf \
    .set("bigdl.ppml.k8s.volume_nfs.source-code", "172.168.0.205","/mnt/sdb/disk1/nfsdata/wangjian/idc") \
    .set("bigdl.ppml.k8s.volume_mount.source-code", "/ppml/notebook/nfs") \
    .set("bigdl.ppml.k8s.volume_nfs.nfs-data", "172.168.0.205", "/mnt/sdb/disk1/nfsdata/guancheng/hf") \
    .set("bigdl.ppml.k8s.volume_mount.nfs-data", "/root/.cache") \
    .set("bigdl.ppml.k8s.volume_nfs.nfs-model", "172.168.0.205", "/mnt/sdb/disk1/nfsdata/guancheng/model/chinese-pert-base") \
    .set("bigdl.ppml.k8s.volume_mount.nfs-model", "/ppml/model")

k8s_conf.run_k8s()
# -
