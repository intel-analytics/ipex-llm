import torch
from pytorch_lightning import Callback

def _check_data_type(data):
    if isinstance(data, torch.Tensor):
        return
    else:
        for x in data:
            assert isinstance(x, torch.Tensor), ValueError


def _check_loader(loader):
    if loader is None:
        return
    sample = next(iter(loader))
    try:
        x, y = sample
        _check_data_type(x)
        _check_data_type(y)
    except ValueError:
        raise ValueError(
            "DataLoader in Nano can only support following format:\n"
            "- torch.Tensor, torch.Tensor\n"
            "- Tuple(torch.Tensor), Tuple(torch.Tensor)\n"
        )


def _check_loaders(loaders):
    if isinstance(loaders, list):
        for loader in loaders:
            _check_loader(loader)
    else:
        _check_loader(loaders)


class DataLoaderCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        _check_loaders(trainer.train_dataloader)
        _check_loaders(trainer.val_dataloaders)
        _check_loaders(trainer.test_dataloaders)
        if hasattr(trainer, 'predict_dataloaders'):
            _check_loaders(trainer.predict_dataloaders)
