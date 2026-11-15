from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("FSCGNSConverter")
except PackageNotFoundError:
    __version__ = "2026.11"
