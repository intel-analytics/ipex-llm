from abc import abstractmethod


class Callback(object):
    def __init__(self):
        self.model = None
        self.params = None

    @abstractmethod
    def on_batch_begin(self, batch):
        """
        Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        @param batch: Integer, index of batch within the current epoch.
        """
        pass

    @abstractmethod
    def on_batch_end(self, batch):
        """
        Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        @param batch: Integer, index of batch within the current epoch.
        """
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch):
        """
        Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        @param epoch: Integer, index of epoch.
        @param logs: Dict. Currently, saved stats in last epoch has been passed to this argument
        for this method but may change in the future.
        """
        pass

    @abstractmethod
    def on_epoch_end(self, epoch):
        """
        Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        @param epoch:  Integer, index of epoch.
        """
        pass

    @abstractmethod
    def on_train_begin(self):
        """
        Called at the beginning of training.
        Subclasses should override for any actions to run.
        @param logs: Dict. Currently, no data is passed to this argument for this method
          but that may change in the future.
        """
        pass

    @abstractmethod
    def on_train_end(self):
        """
        Called at the end of training.
        Subclasses should override for any actions to run.
        """
        pass

    def set_model(self, model):
        self.model = model

    def set_param(self, param):
        self.params = param
