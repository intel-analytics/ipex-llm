from abc import abstractmethod

class Trainer:    
    @abstractmethod
    def train_epochs(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def validate(self, **kwargs):
        raise NotImplementedError
