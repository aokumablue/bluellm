"""Shared pytest configuration for the bluellm test suite.

Its presence at the tests root fixes the rootdir so pytest prepends this
directory to ``sys.path``; combined with ``pythonpath = ["tests"]`` in
``pyproject.toml`` it makes ``from helpers import ...`` resolve robustly,
including when tests are invoked from outside the repository root. Shared
fixtures, when added, belong here.
"""
