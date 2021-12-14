from abc import abstractmethod


class Callback(object):
    def __init__(self):
        self.model = None
        self.params = None

    @abstractmethod
    def on_batch_begin(self, batch, logs=None):
        pass

    @abstractmethod
    def on_batch_end(self, batch, logs=None):
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch, logs=None):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch, logs=None):
        pass

    @abstractmethod
    def on_train_batch_begin(self, batch, logs=None):
        pass

    @abstractmethod
    def on_train_batch_end(self, batch, logs=None):
        pass

    @abstractmethod
    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently, no data is passed to this argument for this method
              but that may change in the future.
        """
        pass

    @abstractmethod
    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.
        Args:
            logs: Dict. Currently, the output of the last call to `on_epoch_end()`
              is passed to this argument for this method but that may change in
              the future.
        """
        pass

    @abstractmethod
    def set_model(self, model):
        self.model = model

    @abstractmethod
    def set_param(self, param):
        self.params = param
