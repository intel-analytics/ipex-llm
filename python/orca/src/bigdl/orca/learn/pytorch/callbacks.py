from abc import abstractmethod


class Callback(object):
    def __init__(self):
        self.model = None

    @abstractmethod
    def on_batch_begin(self):
        pass

    @abstractmethod
    def on_batch_end(self):
        pass

    @abstractmethod
    def on_epoch_begin(self):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass

    @abstractmethod
    def on_train_batch_begin(self):
        pass

    @abstractmethod
    def on_train_batch_end(self):
        pass

    @abstractmethod
    def on_train_begin(self):
        pass

    @abstractmethod
    def on_train_end(self):
        pass

    @abstractmethod
    def set_model(self, model):
        self.model = model

    @abstractmethod
    def set_param(self):
        pass
