"""Package to install: CLI `smartrain`, workspace from the current directory."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smartrain")
except PackageNotFoundError:
    __version__ = "0.0.0"
