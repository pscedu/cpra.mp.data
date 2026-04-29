import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TRUE_VALUES = ("1", "t", "true", "y", "yes")

os.environ.setdefault("PYGEOAPI_SERVER_URL", "https://mpdp.coastal.la.gov/v1/ogc")
os.environ.setdefault("PYGEOAPI_CONFIG", "cpra/mp/data/ogc_config.yaml")
os.environ.setdefault("PYGEOAPI_OPENAPI", "cpra/mp/data/ogc_openapi.yaml")

def get_config(name, default_value):
    """Get a config variable from the environment."""
    return os.environ.get(f"CPRA_MP_DATA_{name}", default_value)


BASE_PATH = Path(get_config("BASE_PATH", "/ocean/projects/bcs200002p/shared/data"))
GRIDS_PATH = Path(get_config("GRIDS_PATH", "/ocean/projects/bcs200002p/shared/grids"))
DEBUG = get_config("DEBUG", "False").lower() in TRUE_VALUES
MPD_URI = get_config("MPD_URI", "")
STREAM_THRESHOLD = int(get_config("STREAM_THRESHOLD", "25"))

# Map tile and style URLs
GLYPHS_BASE_URL = get_config("GLYPHS_BASE_URL", "https://api.mpdp.coastal.la.gov/glyphs")
SPRITES_BASE_URL = get_config("SPRITES_BASE_URL", "https://api.mpdp.coastal.la.gov/sprites")
STATIC_TILES_BASE_URL = get_config("STATIC_TILES_BASE_URL", "https://api.mpdp.coastal.la.gov/tiles")
