class Callback(object):
    def __init__(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    @abstractmethod
    def on_train_batch_begin(self):
        pass

    @abstractmethod
    def on_train_batch_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def set_model(self):
        pass

    def set_param(self):
        pass
