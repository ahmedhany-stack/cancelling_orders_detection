from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT_DIR / "configs"


def read_yaml(path: Path) -> dict:
    """
    Read a YAML file.

    Args:
        path (Path): YAML file path.

    Returns:
        dict: Parsed YAML content.
    """

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = read_yaml(CONFIG_DIR / "config.yaml")

MODEL = read_yaml(CONFIG_DIR / "model.yaml")

PATHS = read_yaml(CONFIG_DIR / "paths.yaml")