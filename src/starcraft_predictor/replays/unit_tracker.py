import logging
from collections import Counter

import sc2reader
from sc2reader.events.tracker import (
    UnitBornEvent,
    UnitDiedEvent,
    UnitDoneEvent,
    UnitInitEvent,
    UnitTypeChangeEvent,
    UpgradeCompleteEvent,
)

logger = logging.getLogger(__name__)


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
        "Observer",
        "Warp Prism",
        "Phoenix",
        "Void Ray",
        "Oracle",
        "Carrier",
        "Tempest",
        "Mothership Core",
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
        "Siege Tank",
        "Cyclone",
        "Widow Mine",
        "Thor",
        "Viking",
        "VikingFighter",
        "VikingAssault",
        "Medivac",
        "Liberator",
        "Raven",
        "Banshee",
        "Battlecruiser",
    ],
    "Zerg": [
        "Drone",
        "Queen",
        "Zergling",
        "Baneling",
        "Roach",
        "Ravager",
        "Hydralisk",
        "Lurker",
        "Infestor",
        "Swarm Host",
        "Ultralish",
        "Overlord",
        "Overseer",
        "Mutalisk",
        "Corruptor",
        "Brood Lord",
        "Viper",
    ],
}

TRACKED_UPGRADE_TYPES = {
    "Protoss": [
        "WarpGateResearch",
        "BlinkTech",
        "Charge",
        "ProtossGroundWeaponsLevel1",
        "ProtossGroundWeaponsLevel2",
        "ProtossGroundWeaponsLevel3",
        "PsiStormTech",
        "ExtendedThermalLance",
        "ProtossGroundArmorsLevel1",
        "ProtossGroundArmorsLevel2",
        "ProtossGroundArmorsLevel3",
        "ProtossShieldsLevel1",
        "ProtossShieldsLevel2",
        "ProtossShieldsLevel3",
        "GraviticDrive",
        "DarkTemplarBlinkUpgrade",
        "ObserverGraviticBooster",
        "ProtossAirWeaponsLevel1",
        "ProtossAirWeaponsLevel2",
        "ProtossAirWeaponsLevel3",
        "AdeptPiercingAttack",
        "TempestGroundAttackUpgrade",
    ],
    "Terran": [
        "Stimpack",
        "ShieldWall",
        "PunisherGrenades",
        "InterferenceMatrix"
        "TerranInfantryWeaponsLevel1",
        "TerranInfantryWeaponsLevel2",
        "TerranInfantryWeaponsLevel3",
        "TerranInfantryArmorsLevel1",
        "TerranInfantryArmorsLevel2",
        "TerranInfantryArmorsLevel3",
        "TerranVehicleWeaponsLevel1",
        "TerranVehicleWeaponsLevel2",
        "TerranVehicleWeaponsLevel3",
        "TerranShipWeaponsLevel1",
        "TerranShipWeaponsLevel2",
        "TerranShipWeaponsLevel3",
        "TerranVehicleAndShipArmorsLevel1",
        "TerranVehicleAndShipArmorsLevel2",
        "TerranVehicleAndShipArmorsLevel3",
        "BansheeCloak",
        "CycloneLockOnDamageUpgrade",
        "PersonalCloaking",
        "HighCapacityBarrels",
        "TerranBuildingArmor",
        "LiberatorAGRangeUpgrade",
        "SmartServos",
        "DrillClaws",
        "HiSecAutoTracking",
        "MedivacCaduceusReactor",
        "BansheeSpeed",
        "BattlecruiserEnableSpecializations",
    ],
    "Zerg": [
        "zerglingmovementspeed",
        "ZergMeleeWeaponsLevel1",
        "ZergMeleeWeaponsLevel2",
        "ZergMeleeWeaponsLevel3",
        "GlialReconstitution",
        "overlordspeed",
        "ZergGroundArmorsLevel1",
        "ZergGroundArmorsLevel2",
        "ZergGroundArmorsLevel3",
        "ZergMissileWeaponsLevel1",
        "ZergMissileWeaponsLevel2",
        "ZergMissileWeaponsLevel3",
        "CentrificalHooks",
        "EvolveGroovedSpines",
        "zerglingattackspeed",
        "EvolveMuscularAugments",
        "Burrow",
        "LurkerRange",
        "DiggingClaws",
        "ChitinousPlating",
        "Frenzy",
        "ZergFlyerWeaponsLevel1",
        "ZergFlyerWeaponsLevel2",
        "ZergFlyerWeaponsLevel3",
        "AnabolicSynthesis",
        "NeuralParasite",
        "ZergFlyerArmorsLevel1",
        "ZergFlyerArmorsLevel2",
        "ZergFlyerArmorsLevel3",
        "TunnelingClaws",
    ],
}


class UnitTracker:
    """Class for tracking units in a StarCraft II replay.

    This class handles all unit-related events and maintains the current state of units for both players.
    It tracks units that are born, die, change type, or are initialized and completed.

    A note on event.unit_type_name and event.unit.name:
    - event.unit_type_name is the name of the unit type produced by that event. Whereas event.unit.name is
      current type in the unit object. Because sc2reader.load_replay() ingestst ALL events before we start
      processing them, event.unit.name is always the type of the unit at the end of the game, even if we are looking at
      the UnitBornEvent which created it. For example, below is a process flow of a unit's life from a fully loaded
      replay:

        1. UnitBornEvent
            - The event.unit_type_name is "Zergling", since the event produced a zergling.
            - The event.unit.name is "Baneling" since, at some point later in the game, this unit morphed to a baneling.
        2. UnitTypeChangeEvent
            - The event.unit_type_name is "Baneling" since the event produced a baneling.
            - The event.unit.name is "Baneling" since, as stated above, the unit object has been updated to a baneling.
              This is the event which transformed unit.name from "Zergling" to "Baneling" during the load of the replay.
        3. UnitDiedEvent
            - The event has no property event.unit_type_name, since it does not produce a unit.
            - The event.unit.name is "Baneling", and we can use this type as it is the last type of the unit, since it
              is dead.

    This has consequences for which attribute we use when tracking units from various events.

    Attributes
    ----------
    players : List[sc2reader.objects.Participant]
        List of players in the replay.
    tracked_units : Dict[int, Dict[int, str]]
        Dictionary containing the currently tracked units for both players, indexed by player ID.
        Format: {player_id: {unit_id: unit_type_name, ...}}
    tracked_initialisations : Dict[int, Dict[int, str]]
        Dictionary containing the currently tracked initialisations for both players, indexed by player ID.
        Format: {player_id: {unit_id: unit_type_name, ...}}
    tracked_upgrades : Dict[int, Dict[int, str]]
        Dictionary containing the currently tracked upgrades for both players, indexed by player ID.
        Format: {player_id: {upgrade_id: upgrade_type_name, ...}}

    """

    def __init__(self, players: list[sc2reader.objects.Participant]) -> None:
        """Initialize the UnitTracker.

        Parameters
        ----------
        players : List[sc2reader.objects.Participant]
            List of players in the replay.

        """
        self.players = players
        self.tracked_units = {0: {}, 1: {}}
        self.tracked_initialisations = {0: {}, 1: {}}
        self.tracked_upgrades = {0: {}, 1: {}}

    def handle_unit_born(self, event: UnitBornEvent) -> None:
        """Handle a UnitBornEvent and update the units dictionary.

        Parameters
        ----------
        event : UnitBornEvent
            The UnitBornEvent to process.

        """
        player = event.unit.owner
        if player is None:
            return

        if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
            self.tracked_units[player.pid - 1][
                event.unit_id
            ] = event.unit_type_name

    def handle_unit_died(self, event: UnitDiedEvent) -> None:
        """Handle a UnitDiedEvent and update the units dictionary.

        UnitDiedEvents can remove units that are either tracked or units that are being initialised.

        Parameters
        ----------
        event : UnitDiedEvent
            The UnitDiedEvent to process.

        """
        player = event.unit.owner
        if player is None:
            return

        if event.unit.name in TRACKED_UNIT_TYPES[player.play_race]:
            removed_unit = self.tracked_units[player.pid - 1].pop(event.unit_id, None)
            removed_initialisation = self.tracked_initialisations[player.pid - 1].pop(event.unit_id, None)
            if not any([removed_unit, removed_initialisation]):
                logger.warning(
                    "Unit not found in tracked units for both players.",
                    extra={"unit_id": event.unit_id, "unit_name": event.unit.name},
                )

    def handle_unit_type_change(self, event: UnitTypeChangeEvent) -> None:
        """Handle a UnitTypeChangeEvent and update the units dictionary.

        UnitTypeChangeEvents modify the unit_type_name for an already existing unit_id.

        Parameters
        ----------
        event : UnitTypeChangeEvent
            The UnitTypeChangeEvent to process.

        """
        player = event.unit.owner
        if player is None:
            return

        if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
            try:
                self.tracked_units[player.pid - 1][event.unit_id] = event.unit_type_name
            except KeyError:
                logger.warning(
                "Warning: UnitTypeChangeEvent unit not found in tracked units for both players.",
                extra={"unit_id": event.unit_id},
            )

    def handle_unit_init(self, event: UnitInitEvent) -> None:
        """Handle a UnitInitEvent and update the initialisations dictionary.

        UnitInitEvents are for units which are not born, but are initialized, eg: warped in units.

        Parameters
        ----------
        event : UnitInitEvent
            The UnitInitEvent to process.

        """
        player = event.unit.owner
        if player is None:
            return

        if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
            self.tracked_initialisations[player.pid - 1][
                event.unit_id
            ] = event.unit_type_name

    def handle_unit_done(self, event: UnitDoneEvent) -> None:
        """Handle a UnitDoneEvent and update the units dictionary.

        UnitDoneEvents are for finished initialisations, eg: finished warping in a unit.

        Parameters
        ----------
        event : UnitDoneEvent
            The UnitDoneEvent to process.

        """
        player = event.unit.owner
        if player is None:
            return

        if event.unit.name in TRACKED_UNIT_TYPES[player.play_race]:
            try:
                unit_type_name = self.tracked_initialisations[player.pid - 1].pop(event.unit_id)
                self.tracked_units[player.pid - 1][event.unit_id] = unit_type_name
            except KeyError:
                logger.warning(
                    "Warning: UnitDoneEvent for not found in tracked units for both players.",
                    extra={"unit_id": event.unit_id, "unit_name": event.unit.name},
                )

    def handle_upgrade_complete(self, event: UpgradeCompleteEvent) -> None:
        """Handle a UpgradeCompleteEvent and update the upgrades dictionary.

        Parameters
        ----------
        event : UpgradeCompleteEvent
            The UpgradeCompleteEvent to process.

        """
        player = event.player
        if player is None:
            return

        if event.upgrade_type_name in TRACKED_UPGRADE_TYPES[player.play_race]:
            self.tracked_upgrades[player.pid - 1][
                event.upgrade_type_name
            ] = 1

    def get_current_units(self) -> dict[str, int]:
        """Get the current units for both players based on their tracked units.

        Returns
        -------
        Dict[str, int]
            Dictionary containing the current unit counts for both players.
            Format: {f"player_{i}_{unit_type}": count, ...}

        """
        output_dict = {}
        for i, player in enumerate(self.players):
            player_units = Counter(self.tracked_units[i].values())
            player_units = {
                f"player_{i+1}_{x}": player_units.get(x, 0)
                for x in TRACKED_UNIT_TYPES[player.play_race]
            }

            player_upgrades = {
                f"player_{i+1}_{x}": self.tracked_upgrades[i].get(x, 0)
                for x in TRACKED_UPGRADE_TYPES[player.play_race]
            }

            output_dict = {**output_dict, **player_units, **player_upgrades}

        return output_dict
