import warnings
from functools import wraps

def deprecated(message=""):
    def deprecated_decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn("{} will be deprecated in future release. {}"
                          .format(function.__name__, message),
                          category=DeprecationWarning)
            warnings.simplefilter('default', DeprecationWarning)
            return function(*args, **kwargs)
        return wrapped
    return deprecated_decorator
