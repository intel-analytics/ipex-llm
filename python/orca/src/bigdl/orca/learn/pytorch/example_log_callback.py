from python.orca.src.bigdl.orca.learn.pytorch.callbacks import Callback
import logging
logger = logging.getLogger(__name__)


class ExampleLogCallback(Callback):
    def on_batch_end(self, batch):
        logger.info("on_batch_end called")
        logger.info(f"batch got here: {batch}")

    def on_epoch_begin(self, epoch):
        logger.info("on_epoch_begin called")
        logger.info(f"epoch got here: {epoch}")

    def on_epoch_end(self, epoch):
        logger.info("on_epoch_end called")
        logger.info(f"epoch got here: {epoch}")

    def on_train_begin(self):
        logger.info("on_train_begin called")

    def on_train_end(self):
        logger.info("on_train_end called")

    def on_batch_begin(self, batch):
        logger.info("On_batch_begin called")
        logger.info(f"batch got here {batch}")

