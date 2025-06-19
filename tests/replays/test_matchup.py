import pytest

from starcraft_predictor.replays.matchup import Matchup


def test_matchup_from_replay(pvp_replay):
    """Test that the matchup is correctly extracted from a replay."""
    matchup = Matchup.from_replay(pvp_replay)
    assert matchup == Matchup.PVP

def test_matchup_orders_races_from_replay(zvp_replay):
    """Test that the matchup orders the races correctly."""
    assert zvp_replay.players[0].play_race == "Zerg"
    assert zvp_replay.players[1].play_race == "Protoss"
    matchup = Matchup.from_replay(zvp_replay)
    assert matchup.race1 == "Protoss"
    assert matchup.race2 == "Zerg"

def test_matchup_from_str():
    """Test that the matchup is correctly extracted from a string."""
    matchup = Matchup("Protoss vs Terran")
    assert matchup == Matchup.PVT

def test_matchup_from_str_invalid():
    """Test that an invalid matchup raises a ValueError."""
    with pytest.raises(ValueError):
        Matchup("Protoss vs Terran vs Zerg")
