import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

TRUE_VALUES = ("1", "t", "true", "y", "yes")


def get_config(name, default_value):
    """Get a config variable from the environment."""
    return os.environ.get(f"CPRA_MP_DATA_{name}", default_value)


BASE_PATH = Path(get_config("BASE_PATH", "/ocean/projects/bcs200002p/shared/data-long"))
DEBUG = get_config("DEBUG", "False").lower() in TRUE_VALUES
