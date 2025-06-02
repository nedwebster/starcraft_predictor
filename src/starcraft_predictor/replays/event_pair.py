from pydantic import BaseModel, ConfigDict, model_validator
from sc2reader.events.tracker import PlayerStatsEvent


class EventPair(BaseModel):
    """Class to represent a pair of PlayerStatsEvents from a replay, correspnding to two players at the same game time."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    event_1: PlayerStatsEvent
    event_2: PlayerStatsEvent

    @model_validator(mode="after")
    def validate_events(self):
        if self.event_1.player == self.event_2.player:
            raise ValueError("Events must be from alternating players")
        if self.event_1.second != self.event_2.second:
            raise ValueError("Events must be from the same game time")
        return self


def build_event_pairs(events: list[PlayerStatsEvent]) -> list[EventPair]:
    """
    Convert a list of events into a list of EventPairs.

    Eg.
    [event1, event2, event3, event4]
    -->
    [EventPair(event1, event2), EventPair(event3, event4)]

    """
    event_pairs = []
    for i in range(0, len(events), 2):
        event_pair = EventPair(event_1=events[i], event_2=events[i + 1])
        event_pairs.append(event_pair)
    return event_pairs
