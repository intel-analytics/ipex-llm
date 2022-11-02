import ray
from ray.exceptions import RayActorError
from bigdl.dllib.utils.log4Error import *

def get_driver_node_ip():
    """
    Returns the IP address of the current node.

    :return: the IP address of the current node.
    """
    return ray._private.services.get_node_ip_address()

def check_for_failure(remote_values):
    """Checks remote values for any that returned and failed.
    :param remote_values: List of object IDs representing functions
            that may fail in the middle of execution. For example, running
            a SGD training loop in multiple parallel actor calls.
    :return Bool for success in executing given remote tasks.
    """
    unfinished = remote_values
    try:
        while len(unfinished) > 0:
            finished, unfinished = ray.wait(unfinished)
            finished = ray.get(finished)
        return True
    except RayActorError as exc:
        logger.exception(str(exc))
    return False
