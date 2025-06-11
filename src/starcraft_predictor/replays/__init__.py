import sc2reader

from starcraft_predictor.errors import ReplayIngestionError


def load_replay(
    replay_path: str | None = None,
    replay: sc2reader.resources.Replay | None = None,
) -> sc2reader.resources.Replay:
    """Load a replay from a path, or return an already loaded replay object.

    Parameters
    ----------
    replay_path : str | None
        The path to the replay file.
    replay : sc2reader.resources.Replay | None
        An already loaded replay object.

    """
    if replay_path is not None and replay is not None:
        raise ReplayIngestionError
    if replay_path is not None:
        return sc2reader.load_replay(replay_path)
    return replay
