import logging
import time
import sys

def logger_creator(log_dir="./orca.log"):
    # initialize logging stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler_std = logging.StreamHandler(sys.stdout)
    handler_std.setLevel(logging.INFO)
    handler_std.setFormatter(formatter)

    handler_log = logging.FileHandler(log_dir)
    handler_log.setLevel(logging.INFO)
    handler_log.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler_log)
        logger.addHandler(handler_std)

    return logger

if __name__ == '__main__':
    orca_logger = logger_creator()
