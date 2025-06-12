from enum import Enum

import sc2reader


class Matchup(Enum):
    """Possible combinations for matchups in StarCraft II.

    Note: These matchups are treated as combinations, not permutations. TvZ and ZvT are considered the same matchup, and
    the replay ingestion process will reverse the players in ZvT to match the TvZ Matchup.

    """

    PVP = "Protoss vs Protoss"
    PVT = "Protoss vs Terran"
    PVZ = "Protoss vs Zerg"
    TVT = "Terran vs Terran"
    TVZ = "Terran vs Zerg"
    ZVZ = "Zerg vs Zerg"

    @property
    def race1(self) -> str:
        """The first race in the matchup."""
        return self.value.split(" ")[0]

    @property
    def race2(self) -> str:
        """The second race in the matchup."""
        return self.value.split(" ")[2]

    @property
    def races(self) -> list[str]:
        """The races in the matchup."""
        return [self.race1, self.race2]

    @classmethod
    def from_replay(cls, replay: sc2reader.resources.Replay) -> "Matchup":
        """Create a Matchup from a replay."""
        races = [x.play_race for x in replay.players]
        races.sort()
        return cls(f"{races[0]} vs {races[1]}")
