from .LifeCycleManager import LifeCycleManager
from .Trainer import Trainer

class BaseRunner(Trainer, LifeCycleManager):
    pass
