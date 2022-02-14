import functools


def proxy_method(cls, name):
    # This unbound method will be pulled from the superclass.
    assert(hasattr(cls,name))
    proxyed = getattr(cls, name)
    @functools.wraps(proxyed)
    def wrapper(self, *args, **kwargs):
        return self._proxy(name, proxyed.__get__(self, cls), *args, **kwargs)
    return wrapper


def proxy_methods(cls):
    for name in cls.PROXYED_METHODS:
        setattr(cls, name, proxy_method(cls, name))
    return cls