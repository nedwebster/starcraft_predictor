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
        "Medivac",
        "Liberator",
        "Ravem",
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
        if event.control_pid == 0:  # Some events during the game setup are not assigned to player 1 or player 2
            return

        player = self.players[event.control_pid - 1]

        if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
            self.tracked_units[event.control_pid - 1][
                event.unit_id
            ] = event.unit_type_name

    def handle_unit_died(self, event: UnitDiedEvent) -> None:
        """Handle a UnitDiedEvent and update the units dictionary.

        UnitDiedEvents do not have a player id, so we attempt to remove the unit from both players' tracked units.

        Parameters
        ----------
        event : UnitDiedEvent
            The UnitDiedEvent to process.

        """
        try:
            self.tracked_units[0].pop(event.unit_id, None)
        except KeyError:
            try:
                self.tracked_units[1].pop(event.unit_id, None)
            except KeyError:
                logger.warning("Unit not found in tracked units for both players.", extra={"unit_id": event.unit_id})

    def handle_unit_type_change(self, event: UnitTypeChangeEvent) -> None:
        """Handle a UnitTypeChangeEvent and update the units dictionary.

        UnitTypeChangeEvents do not have a player id, so we attempt to update the unit type for both players' tracked
        units. UnitTypeChangeEvents modify the unit_type_name for an already existing unit_id.

        Parameters
        ----------
        event : UnitTypeChangeEvent
            The UnitTypeChangeEvent to process.

        """
        not_found_unit = 0

        for i, player in enumerate(self.players):
            if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
                try:
                    self.tracked_units[i][event.unit_id] = event.unit_type_name
                except KeyError:
                    not_found_unit += 1

        if not_found_unit > 1:
            logger.warning(
                "Warning: UnitTypeChangeEvent unit not found in tracked units for both players.",
                extra={"unit_id": event.unit_id},
            )

    def handle_unit_init(self, event: UnitInitEvent) -> None:
        """Handle a UnitInitEvent and update the initialisations dictionary.

        Parameters
        ----------
        event : UnitInitEvent
            The UnitInitEvent to process.

        """
        if event.control_pid == 0:
            return

        player = self.players[event.control_pid - 1]

        if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
            self.tracked_initialisations[event.control_pid - 1][
                event.unit_id
            ] = event.unit_type_name

    def handle_unit_done(self, event: UnitDoneEvent) -> None:
        """Handle a UnitDoneEvent and update the units dictionary.

        Parameters
        ----------
        event : UnitDoneEvent
            The UnitDoneEvent to process.

        """
        try:
            unit_type_name = self.tracked_initialisations[0].pop(event.unit_id)
            self.tracked_units[0][event.unit_id] = unit_type_name
        except KeyError:
            try:
                unit_type_name = self.tracked_initialisations[1].pop(event.unit_id)
                self.tracked_units[1][event.unit_id] = unit_type_name
            except KeyError:
                logger.debug(
                    "Warning: UnitDoneEvent for not found in tracked units for both players.",
                    extra={"unit_id": event.unit_id},
                )

    def handle_upgrade_complete(self, event: UpgradeCompleteEvent) -> None:
        """Handle a UpgradeCompleteEvent and update the upgrades dictionary.

        Parameters
        ----------
        event : UpgradeCompleteEvent
            The UpgradeCompleteEvent to process.

        """
        if event.pid == 0:
            return

        player = self.players[event.pid - 1]

        if event.upgrade_type_name in TRACKED_UPGRADE_TYPES[player.play_race]:
            self.tracked_upgrades[event.pid - 1][
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
