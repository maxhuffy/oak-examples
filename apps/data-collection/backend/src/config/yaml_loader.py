import yaml
from pathlib import Path


class YamlLoader:
    """YAML file loader that converts dicts to objects with dot access."""

    def __init__(self, file_path: str | Path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._data = self._wrap(data)

    def _wrap(self, obj) -> any:
        if isinstance(obj, dict):
            return type("YamlNamespace", (), {k: self._wrap(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [self._wrap(x) for x in obj]
        return obj

    def __getattr__(self, name: str) -> any:
        return getattr(self._data, name)
