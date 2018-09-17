"""
helper/time.py
"""

from functools import wraps
from sys import stdout
from time import perf_counter


def timer(message: str = None):
    """Print the runtime of the decorated function"""

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if message:
                print(message, end="", file=stdout)
            start_time = perf_counter()
            value = func(*args, **kwargs)
            end_time = perf_counter()
            run_time = end_time - start_time
            print(f"\t-> done in {run_time:.3f}s")
            return value

        return wrapper

    return inner
