import os
import torch
from bigdl.nano.pytorch.vision.datasets import ImageFolder
from bigdl.nano.pytorch.vision.transforms import transforms
import argparse
from torch.utils.data import DataLoader
from bigdl.nano.pytorch.vision.models import resnet50 
from bigdl.nano.pytorch.vision.models import ImageClassifier 
from bigdl.nano.common import init_nano 
from bigdl.nano.pytorch.trainer import Trainer



parser = argparse.ArgumentParser(description='PyTorch Cat Dog')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--root_dir', default='../root', help='path to cat vs dog dataset'
                    'which should have two folders `cat` and `dog`, each containing'
                    'cat and dog pictures.')


def create_data_loader(root_dir, batch_size):
    dir_path = os.path.realpath(root_dir)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    catdogs = ImageFolder(dir_path, data_transform)
    dataset_size = len(catdogs)
    train_split = 0.8
    train_size = int(dataset_size * train_split)
    val_size = dataset_size - train_size
    train_set, val_set = torch.utils.data.random_split(catdogs, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


class Classifier(ImageClassifier):

    def __init__(self):
        backbone = resnet50(pretrained=True, include_top=False, freeze=True)
        super().__init__(backbone=backbone, num_classes=2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002, amsgrad=True)


def main():
    args = parser.parse_args()
    init_nano()
    train_loader, val_loader = create_data_loader(args.root_dir, args.batch_size)
    classifier = Classifier()
    trainer = Trainer(max_epochs=1)
    trainer.fit(classifier, train_loader)
    trainer.test(classifier, val_loader)


if __name__ == "__main__":
    main()
