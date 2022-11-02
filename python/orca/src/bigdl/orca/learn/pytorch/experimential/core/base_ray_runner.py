from .lifecycle_manager import LifeCycleManager
from .trainer import Trainer
from .model_io import ModelIO

from abc import ABCMeta


class BaseRayRunner(Trainer, LifeCycleManager, ModelIO, metaclass=ABCMeta):
    pass
