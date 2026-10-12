"""Sport registry."""
from ..config import Config
from ..exceptions import ConfigError
from .base import SportsAnalyzer
from .football import FootballAnalyzer

_REGISTRY: dict[str, type[SportsAnalyzer]] = {"football": FootballAnalyzer}


def get_sport_analyzer(config: Config) -> SportsAnalyzer:
    try:
        return _REGISTRY[config.sport](config)
    except KeyError as exc:
        raise ConfigError(f"Unsupported sport '{config.sport}'. Available: {sorted(_REGISTRY)}") from exc


__all__ = ["SportsAnalyzer", "FootballAnalyzer", "get_sport_analyzer"]
