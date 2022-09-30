import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatches
import seaborn as sns


def get_font_prop():

    PACKAGE_INSTALLATION_PATH = os.path.dirname(
        os.path.abspath(__file__)
    )

    font_prop = font_manager.FontProperties(
        fname=PACKAGE_INSTALLATION_PATH + "/fonts/starcraft_font.ttf"
    )

    return font_prop


class PlotEngine:
    """
    Class for creating visualizations.

    Visualizations are created as part of the training process
    and prediction process.
    """

    RACE_COLORS = {
        "p": "#0C48CC",
        "t": "#F40404",
        "z": "#88409C",
    }

    # set global setting for background
    plt.rcParams['font.family'] = 'monospace'
    FONT_PROP = get_font_prop()

    @staticmethod
    def _find_crossing_pairs(y, threshold):
        """
        Function to find the pair of y values that cross a given threshold.
        """

        pairs = []

        y_temp = list(np.subtract(y, threshold))

        for i in range(len(y_temp) - 1):
            if (np.sign(y_temp[i]) == 0) or (np.sign(y_temp[i+1]) == 0):
                pass
            elif np.sign(y_temp[i]) != np.sign(y_temp[i + 1]):
                pairs.append((i, i+1))

        return pairs

    @staticmethod
    def _insert_new_values(x, y, crossing_index, threshold):
        """
        Function to insert a new value inbetween the crossing pair,
        that sits on the threshold line.

        """

        rate_of_change = (
            (y[crossing_index[1]] - y[crossing_index[0]])
            / (x[crossing_index[1]] - x[crossing_index[0]])
        )

        x_movement = (threshold - y[crossing_index[0]]) / rate_of_change

        new_x = x[crossing_index[0]] + x_movement

        x.insert(crossing_index[1], new_x)
        y.insert(crossing_index[1], threshold)

        return x, y

    def _add_threshold_points(self, x, y, threshold):
        """
        Wrapper to apply the insert_new_values function to
        all crosisng pairs in x and y

        """

        crossing_pairs = self._find_crossing_pairs(y, threshold)

        for i, pair in enumerate(crossing_pairs):

            pair = tuple(np.add(pair, i))
            x, y, = self._insert_new_values(x, y, pair, threshold)

        return x, y

    def _threshold_plot(self, ax, x, y, threshv, color, overcolor):
        """
        Helper function to plot points above a threshold in a different color.

        This is a pretty hacky way to assign the colours with if/elif
        statements. Should find a way to make this less verbose.

        Parameters
        ----------
        ax : Axes
            Axes to plot to
        x, y : array
            The x and y values

        threshv : float
            Plot using overcolor above this value

        color : color
            The color to use for the lower values

        overcolor: color
            The color to use for values over threshv

        """

        # splits x and y into individual line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # for each line segment, assess what colour it should be
        for i, segment in enumerate(segments):
            if segment[:, 1][0] < 0.5:
                plot_color = color
            elif segment[:, 1][0] > 0.5:
                plot_color = overcolor
            elif segment[:, 1][1] < 0.5:
                plot_color = color
            else:
                plot_color = overcolor

            # plot that individual line segment
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=plot_color,
                linewidth=3,
            )

        return ax

    def win_probability_plot(
        self,
        df: pd.DataFrame,
        p1_race: str,
        p2_race: str,
        match_id: str = "TESTID",
        p1_handle: str = "Player 1",
        p2_handle: str = "Player 2",
    ):
        """
        Win probability plot in starcraft theme

        """

        # create fig and axis
        fig, ax = plt.subplots(figsize=(20, 10), facecolor="#00182c")

        # set facecolor
        ax.set_facecolor("#010713")

        x, y = self._add_threshold_points(
            list(df["seconds"].values),
            list(df["win_prob"].values),
            threshold=0.5,
        )

        # plot the win probability line
        self._threshold_plot(
            ax,
            x,
            y,
            .5,
            self.RACE_COLORS[p1_race],
            self.RACE_COLORS[p2_race],
        )

        # plot 50% line
        sns.lineplot(
            x=df["seconds"],
            y=0.5,
            color="white",
            ls="dashed",
            lw=0.5,
        )

        # set ticks and tick parameters (font, color, etc.)
        ax.set_xticks(df["seconds"].values[::6])
        ax.set_xticklabels([
            int(val) for val in df["seconds"].values[::6] / 60
        ])
        ax.tick_params(axis='x', colors='#62afd4', labelsize=15)
        ax.tick_params(axis='y', colors='#62afd4', labelsize=15)

        # set x and y axis limits
        ax.set_ylim(0, 1)
        ax.set_xlim(df["seconds"].values[0], df["seconds"].values[-1])

        # add border colour to plot
        ax.spines['bottom'].set_color('#62afd4')
        ax.spines['top'].set_color('#62afd4')
        ax.spines['left'].set_color('#62afd4')
        ax.spines['right'].set_color('#62afd4')

        # add custom legend with player colours and names
        patch_1 = mpatches.Patch(
            color=self.RACE_COLORS[p1_race],
            label=p1_handle,
        )
        patch_2 = mpatches.Patch(
            color=self.RACE_COLORS[p2_race],
            label=p2_handle,
        )
        plt.legend(
            handles=[patch_1, patch_2],
            loc="upper left",
            facecolor="#00182c",
            labelcolor="#62afd4",
        )

        # add title and labels to plots
        plt.title(
            "Win Probability Plot\n",
            color="#62afd4",
            fontsize=30,
            fontproperties=self.FONT_PROP,
        )
        ax.set_ylabel("Win Probability\n", fontsize=20, color="#62afd4")
        ax.set_xlabel("\nMinutes", fontsize=20, color="#62afd4")

        return fig
