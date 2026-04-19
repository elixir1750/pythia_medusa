"""Allow running the src-layout package without installation."""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE_DIR = _PACKAGE_DIR.parent / "src" / "pythia_medusa"

if _SRC_PACKAGE_DIR.exists():
    __path__ = [str(_SRC_PACKAGE_DIR)]
else:
    __path__ = []
