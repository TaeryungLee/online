from __future__ import annotations

import os
from importlib import resources

def get_diffuser_params_path() -> str:
    """Return absolute path to diffuser_params.yaml bundled in the package."""
    # importlib.resources.files is available in Python 3.9+
    try:
        return str(resources.files("stablemofusion.config").joinpath("diffuser_params.yaml"))
    except Exception:
        # Fallback: relative to this file
        here = os.path.dirname(__file__)
        return os.path.join(os.path.dirname(here), "config", "diffuser_params.yaml")
