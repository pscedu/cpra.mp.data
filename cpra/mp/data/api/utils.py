from cpra.mp.data.config import DEBUG
from enum import Enum
from functools import wraps
from time import time


def enum_value(inst):
    """Get the value from an enum."""
    return inst.value if isinstance(inst, Enum) else inst


def timing(func):
    """Time an asynchronous function call."""

    @wraps(func)
    async def wrapped(*args, **kwargs):
        start = time()
        result = await func(*args, **kwargs)
        if DEBUG:
            print(
                "func:%r args:[%r, %r] took: %2.4f sec"
                % (func.__name__, args, kwargs, time() - start)
            )
        return result

    return wrapped
