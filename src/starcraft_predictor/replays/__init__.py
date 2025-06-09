from enum import Enum

import sc2reader

TRACKED_UNIT_TYPES = {
    "Protoss": [
        "Probe",
        "Zealot",
        "Stalker",
        "Sentry",
        "Adept",
        "High Templar",
        "Dark Templar",
        "Immortal",
        "Colossus",
        "Disruptor",
        "Archon",
        "Carrier",
        "Tempest",
    ],
    "Terran": [
        "SCV",
        "MULE",
        "Marine",
        "Marauder",
        "Reaper",
        "Ghost",
        "Hellion",
        "Hellbat",
        "Widow Mine",
        "Cyclone",
        "Thor",
        "Liberator",
        "Viking",
        "Banshee",
        "Medivac",
    ],
    "Zerg": [
        "Drone",
        "Overlord",
        "Overseer",
        "Zergling",
        "Baneling",
        "Roach",
        "Ravager",
        "Hydralisk",
        "Lurker",
        "Infestor",
        "Ultralisk",
        "Queen",
    ],
}

TRACKED_UPGRADE_TYPES = {
    "Protoss": [],
    "Terran": [],
    "Zerg": [],
}


EVENT_DATA_FIELDS = [
    "food_made",
    "food_used",
    "minerals_collection_rate",
    "minerals_lost",
    "minerals_lost_army",
    "minerals_used_current",
    "minerals_used_current_army",
    "minerals_used_current_economy",
    "vespene_collection_rate",
    "vespene_lost",
    "vespene_lost_army",
    "vespene_used_current",
    "vespene_used_current_army",
    "vespene_used_current_economy",
    "vespene_used_current_technology",
    "workers_active_count",
]


class Matchup(Enum):
    """
    Possible combinations for matchups in StarCraft II.

    Note: These are NOT permutations. TvZ and ZvT are considered the same matchup, and the replay process will reverse
    the players in ZvT to match the TvZ Matchup.

    """

    PVP = "Protoss vs Protoss"
    PVT = "Protoss vs Terran"
    PVZ = "Protoss vs Zerg"
    TVT = "Terran vs Terran"
    TVZ = "Terran vs Zerg"
    ZVZ = "Zerg vs Zerg"

    @property
    def race1(self):
        return self.value.split(" ")[0]

    @property
    def race2(self):
        return self.value.split(" ")[2]

    @classmethod
    def from_replay(cls, replay: sc2reader.resources.Replay) -> "Matchup":
        races = [x.pick_race for x in replay.players]
        races.sort()
        return cls(f"{races[0]} vs {races[1]}")
