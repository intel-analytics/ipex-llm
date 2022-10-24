import torch
import argparse
import os
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
from transformers import BertTokenizer, BertModel, AdamW
from tqdm.auto import tqdm

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1

def is_distributed():
    return dist.is_available() and dist.is_initialized()

#定义数据集,方便后续模型读取批量数据。
class Dataset(torch.utils.data.Dataset):
    # data_type is actually split, so that we can define dataset for train set/validate set
    def __init__(self, data_type):
        self.data = self.load_data(data_type)

    def load_data(self, data_type):
        tmp_dataset = load_dataset(path='seamew/ChnSentiCorp', split=data_type)
        Data = {}
        # So enumerate will return a index, and  the line?
        # line is a dict, including 'text', 'label'
        for idx, line in enumerate(tmp_dataset):
            sample = line
            Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
        self.bert_encoder = BertModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]
        logits = self.classifier(cls_vectors)
        return logits

device = 'cpu'

def train_loop(dataloader, model, loss_fn, optimizer, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1)*len(dataloader)
    
    # Set to train mode
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # Calculate loss
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

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

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct



def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)") 
    parser.add_argument("--epochs", type=int, default=1, metavar="N",
                        help="number of epochs to train (default: 10)") 
    parser.add_argument("--lr", type=float, default=1e-5, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")
    # Only for test purpose
    parser.add_argument("--load-model", action="store_true", default=False,
                        help="For loading the current model")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if should_distribute():
        print("Using distributed PyTorch with {} backend".format(
            "GLOO"), flush=True)
        dist.init_process_group(backend=dist.Backend.GLOO)


    # Load the data and dataset
    print("[INFO]Before data get loaded", flush=True)
    train_data = Dataset('train')
    valid_data = Dataset('validation')
#    test_data = Dataset('test')

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn)
#    test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn)

    print("[INFO]Data get loaded successfully", flush=True)

    model = NeuralNetwork().to(device)
    if (args.load_model):
        model.load_state_dict('./pert.bin')

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel
        model = Distributor(model,find_unused_parameters=True)
    loss_fn = nn.CrossEntropyLoss()
    epoch_num = 3
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_loss = 0.
    best_acc = 0.

    for t in range(args.epochs):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, model,loss_fn, optimizer, t+1, total_loss)
        valid_acc = test_loop(valid_dataloader, model, mode='Valid')

    print("[INFO]Finish all test", flush=True)

    if (args.save_model):
        torch.save(model.state_dict(), "pert.bin")

if __name__ == "__main__":
    main()
