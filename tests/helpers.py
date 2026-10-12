"""Tiny test helpers so tests run under pytest *and* the bundled runner."""
from contextlib import contextmanager


@contextmanager
def raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")
