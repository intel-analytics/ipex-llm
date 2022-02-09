from bigdl.orca.learn.pytorch.callbacks import Callback
import os
import re
import warnings


class ModelCheckpoint(Callback):
    def __init__(self,
                 filepath=None,
                 save_weights_only=False,
                 ):
        """
        ModelCheckpoint callback is used in conjunction with training using estimator.fit() to save
        a model or weights (in a checkpoint file) at some interval, so the model or weights can be
        loaded later to continue the training from the state saved.
        Example:
            >>> checkpoint_callback = ModelCheckpoint(
            ...     filepath='my/path/sample-mnist-{epoch:02d}',
            ... )
        :param filepath: path to save the model file.
        """
        super().__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.last_ckpt_path = ""

    def on_batch_begin(self, batch):
        """
        Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        @param batch: Integer, index of batch within the current epoch.
        """
        pass

    def on_batch_end(self, batch):
        """
        Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.
        @param batch: Integer, index of batch within the current epoch.
        """
        pass

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

    def on_epoch_end(self, epoch):
        """
        Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.
        @param epoch:  Integer, index of epoch.
        """
        stats = {"epoch": self.trainer.epochs}
        last_ckpt_path = self._format_checkpoint_name(self.filepath, stats)
        self.trainer.save_checkpoint(last_ckpt_path, self.save_weights_only)

    def on_train_begin(self):
        """
        Called at the beginning of training.
        Subclasses should override for any actions to run.
        @param logs: Dict. Currently, no data is passed to this argument for this method
          but that may change in the future.
        """
        # todo: support resume training
        from bigdl.orca.learn.pytorch.utils import get_filesystem
        dirname = os.path.dirname(self.filepath)
        fs = get_filesystem(dirname)
        if fs.exists(dirname):
            files = [os.path.basename(f["name"]) for f in fs.listdir(dirname)]
            files = [x for x in files if "ckpt" in x]
            if len(files) == 0:
                return None
            raise ValueError(f"Find non-empty dirname with filepath of {self.filepath}.")
        else:
            fs.mkdirs(dirname)

    def on_train_end(self):
        """
        Called at the end of training.
        Subclasses should override for any actions to run.
        """
        stats = {"epoch": self.trainer.epochs}
        last_ckpt_path = self._format_checkpoint_name(self.filepath, stats, last=True)
        previous, self.last_ckpt_path = self.last_ckpt_path, last_ckpt_path
        self.trainer.save_checkpoint(last_ckpt_path, self.save_weights_only)
        if previous and previous != last_ckpt_path:
            self.trainer.remove_checkpoint(previous)

    def set_model(self, model):
        self.model = model

    def set_param(self, param):
        self.params = param

    @staticmethod
    def _format_checkpoint_name(filepath, stats, last=False):
        """
        checkpoint name is in the format of 'epoch={epoch}.ckpt'
        """
        filename = os.path.basename(filepath) if not last else "last"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            for group in groups:
                name = group[1:]

                if "epoch" not in name:
                    warnings.warn("We only support filepath with {epoch} for now.")

                filename = filename.replace(group, name + "={" + name)

                if name not in stats:
                    stats[name] = 0
            filename = filename.format(**stats)
        ckpt_name = f"{filename}.ckpt"
        ckpt_path = os.path.join(os.path.dirname(filepath), ckpt_name)
        return ckpt_path
