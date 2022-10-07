import numpy as np
import shap


class StarcraftShap:

    def __init__(self, processed_replay, features, predictions, model):

        self.predictions = predictions
        self.processed_replay = processed_replay
        self.features = features
        self.model = model
        self.moment_index = self._get_moment_index()

    def _get_min_max_indexes(self):
        """
        Iteratively go through a list and
        identify all the local maxima and local minima,
        then return the index pairs of the each line segment.

        eg: given predictions = [1, 3, 5, 3, 2, 8]

        The function would return: [[0, 2], [2, 4], [4, 5]]

        """

        min_max_points = []
        direction = ""

        for i, pred in enumerate(self.predictions):

            # append the first value as a local maxima/minima
            if i == 0:
                pred_value = pred
                min_max_points.append(0)

            # determine the direction at the start
            if i == 1:
                if pred < pred_value:
                    direction = "down"
                    pred_value = pred
                else:
                    direction = "up"
                    pred_value = pred
            # iterate through all points, taking note of when
            # direction changes (ie, local maxima/minima)
            else:
                if direction == "down":
                    if pred < pred_value:
                        pred_value = pred
                    else:
                        min_max_points.append(i-1)
                        direction = "up"
                        pred_value = pred
                elif direction == "up":
                    if pred > pred_value:
                        pred_value = pred
                    else:
                        min_max_points.append(i-1)
                        direction = "down"
                        pred_value = pred

        # append the last point as a local maxima/minima
        min_max_points.append(len(self.predictions) - 1)

        min_max_pairs = [
            [min_max_points[j], min_max_points[j+1]]
            for j in range(len(min_max_points) - 1)
        ]

        return min_max_pairs

    def _get_difference(self, index_pair):
        """
        For a given index pair, get the difference in
        prediction values.

        """

        difference = (
            self.predictions[index_pair[1]]
            - self.predictions[index_pair[0]]
        )

        return difference

    def _get_moment_index(self):
        """
        Get the index for the games 'moment', defined
        as the point in the game where the probability
        monotonically shifts the most. This can happen
        over any length of game time.

        """

        min_max_indexes = self._get_min_max_indexes()

        differences = [
            abs(self._get_difference(index_pair))
            for index_pair in min_max_indexes
        ]

        return min_max_indexes[differences.index(max(differences))]

    def _get_shap_values(self):

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(
            self.processed_replay[self.features]
        )

        return shap_values

    def _get_max_shap_change(self, shap_values, index_pair):

        abs_diff = list(abs(
            np.array(shap_values[index_pair[0]])
            - np.array(shap_values[index_pair[1]])
        ))

        max_index = abs_diff.index(max(abs_diff))

        return max_index

    def _get_feature_difference(
        self, feature, index_pair
    ):

        first_index_value = self.processed_replay.loc[index_pair[0], feature]
        second_index_value = self.processed_replay.loc[index_pair[1], feature]

        feature_change = abs(first_index_value - second_index_value)

        return feature_change

    def get_moment(self):

        moment_index = self._get_moment_index()
        shap_values = self._get_shap_values()
        feature_index = self._get_max_shap_change(shap_values, moment_index)
        feature = self.features[feature_index]
        feature_difference = self._get_feature_difference(
            feature,
            moment_index,
        )

        return (moment_index, feature, feature_difference)
