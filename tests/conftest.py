import pytest
import sc2reader


@pytest.fixture
def pvp_replay() -> sc2reader.resources.Replay:
    """Replay fixture for testing."""
    return sc2reader.load_replay("tests/data/PvP_replay.SC2Replay")

@pytest.fixture
def zvp_replay() -> sc2reader.resources.Replay:
    """Replay fixture for testing."""
    return sc2reader.load_replay("tests/data/ZvP_replay.SC2Replay")


@pytest.fixture
def team_game_replay() -> sc2reader.resources.Replay:
    """Replay fixture for testing."""
    return sc2reader.load_replay("tests/data/team_game_replay.SC2Replay")
