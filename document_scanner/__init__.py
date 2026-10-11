from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("document-scanner")
except _PackageNotFoundError:
    # Running from a source checkout without installing
    __version__ = "unknown"
