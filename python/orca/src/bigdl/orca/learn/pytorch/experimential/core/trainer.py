from abc import abstractmethod, ABCMeta

class Trainer(metaclass=ABCMeta):    
    @abstractmethod
    def train_epochs(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def validate(self, **kwargs):
        pass
