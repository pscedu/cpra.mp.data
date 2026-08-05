import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TRUE_VALUES = ("1", "t", "true", "y", "yes")


def get_config(name, default_value):
    """Get a config variable from the environment."""
    return os.environ.get(f"CPRA_MP_DATA_{name}", default_value)


BASE_PATH = Path(get_config("BASE_PATH", "/ocean/projects/bcs200002p/shared/data"))
CROSSWALK_PATH = Path(get_config("CROSSWALK_PATH", Path(__file__).parent / "crosswalk"))
GEOMETRY_PATH = Path(__file__).parent / "geometry"
TILES_PATH = Path(get_config("TILES_PATH", "/ocean/projects/bcs200002p/shared/tiles"))
DEBUG = get_config("DEBUG", "False").lower() in TRUE_VALUES
MPD_URI = get_config("MPD_URI", "")
STREAM_THRESHOLD = int(get_config("STREAM_THRESHOLD", "25"))


def get_base_path(base_path=None):
    """Resolve a base path, falling back to the configured default.

    Read at call time rather than bound as a default argument value so that
    overriding BASE_PATH (in tests, or between calls) takes effect.
    """
    return Path(base_path) if base_path is not None else BASE_PATH


# Map tile and style URLs
GLYPHS_BASE_URL = get_config("GLYPHS_BASE_URL", "https://api.mpdp.coastal.la.gov/glyphs")
SPRITES_BASE_URL = get_config("SPRITES_BASE_URL", "https://api.mpdp.coastal.la.gov/sprites")
STATIC_TILES_BASE_URL = get_config("STATIC_TILES_BASE_URL", "https://api.mpdp.coastal.la.gov/tiles")
