import pandas as pd
from abc import ABC, abstractmethod

from starcraft_predictor import Replay


class BaseEventProcessor(ABC):

    @classmethod
    @abstractmethod
    def process_events(cls, replay: Replay) -> pd.DataFrame:
        """
        Process all events of chosen type for a single replay. The role of
        this method is to convert important information from the sc2reader
        classes and store it in a pandas.dataframe ready for modeling.
        """
        pass
