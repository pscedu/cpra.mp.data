from cpra.mp.data.config import DEBUG
from enum import Enum
from functools import wraps
from time import time


def enum_value(inst):
    """Get the value from an enum."""
    return inst.value if isinstance(inst, Enum) else inst


def timing(f):
    """Time a function call in debug mode."""
    # Adapted from: https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator

    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        if DEBUG:
            print(
                "func:%r args:[%r, %r] took: %2.4f sec"
                % (f.__name__, args, kw, time() - start)
            )
        return result

    return wrap
