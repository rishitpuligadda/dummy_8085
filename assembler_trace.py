import functools

TRACE_ENABLE =  True
def trace(func):
    """A decorator that prints the function signature and return value."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if TRACE_ENABLE:
            args_str = ", ".join(repr(arg) for arg in args)
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            signature = ", ".join([args_str, kwargs_str]) if kwargs else args_str
            print(f"Calling {func.__name__}({signature})")
            result = func(*args, **kwargs)
            print(f"{func.__name__} returned {result!r}")
            return result
    return wrapper

