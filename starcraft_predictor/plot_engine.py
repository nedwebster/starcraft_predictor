import pandas as pd
import numpy as np
from numpy.random import rand
from plotnine import (ggplot, aes, geom_line,
                      theme, xlab, ylab, geom_hline,
                      scale_color_manual, guides, ggtitle,
                      scale_x_continuous)
from plotnine.themes import (element_line, element_text,
                             element_blank, element_rect)
from plotnine.options import get_option


# code for ThemeStarcraft modified from:
# https://plotnine.readthedocs.io/en/stable/_modules/plotnine/themes/theme_gray.html
class ThemeStarcraft(theme):
    """
    Custom theme for Starcraft 2 flavored plots.
    """
    def __init__(self, base_size=11, base_family=None):
        base_family = base_family or get_option('base_family')
        half_line = base_size/2
        background_color = "#d1d7de"

        theme.__init__(
            self,
            line=element_line(color='black', size=1,
                              linetype='solid', lineend='butt'),
            rect=element_rect(fill=background_color, color='black',
                              size=1, linetype='solid'),
            text=element_text(family=base_family, style='normal',
                              color='black', size=base_size,
                              linespacing=0.9, ha='center',
                              va='center', rotation=0, margin={}),
            aspect_ratio=get_option('aspect_ratio'),

            axis_line=element_line(),
            axis_line_x=element_blank(),
            axis_line_y=element_blank(),
            axis_text=element_text(size=base_size*.8,
                                   color='#4D4D4D'),
            axis_text_x=element_text(
                va='top', margin={'t': half_line*0.8/2}),
            axis_text_y=element_text(
                ha='right', margin={'r': half_line*0.8/2}),
            axis_ticks=element_line(color='#333333'),
            axis_ticks_length=0,
            axis_ticks_length_major=half_line/2,
            axis_ticks_length_minor=half_line/4,
            axis_ticks_minor=element_blank(),
            axis_ticks_direction='out',
            axis_ticks_pad=2,
            axis_title_x=element_text(
                va='top', margin={'t': half_line*0.8}),
            axis_title_y=element_text(
                angle=90, va='bottom', margin={'r': half_line*0.8}),

            dpi=get_option('dpi'),
            figure_size=get_option('figure_size'),

            # legend, None values are for parameters where the
            # drawing routines can make better decisions than
            # can be pre-determined in the theme.
            legend_background=element_rect(color='None'),
            legend_entry_spacing_x=5,
            legend_entry_spacing_y=2,
            legend_key=element_rect(fill=background_color,
                                    colour='None'),
            legend_key_size=base_size*0.8*1.8,
            legend_key_height=None,
            legend_key_width=None,
            legend_margin=0,     # points
            legend_spacing=10,   # points
            legend_text=element_text(
                size=base_size*0.8, ha='left',
                margin={'t': 3, 'b': 3, 'l': 3, 'r': 3,
                        'units': 'pt'}),
            legend_text_legend=element_text(va='baseline'),
            legend_text_colorbar=element_text(va='center'),
            legend_title=element_text(ha='left',
                                      margin={'t': half_line*0.8,
                                              'b': half_line*0.8,
                                              'l': half_line*0.8,
                                              'r': half_line*0.8,
                                              'units': 'pt'}),
            legend_title_align=None,
            legend_position='right',
            legend_box=None,
            legend_box_margin=10,    # points
            legend_box_just=None,
            legend_box_spacing=0.1,  # In inches
            legend_direction=None,

            panel_background=element_rect(fill=background_color),
            panel_border=element_blank(),
            panel_grid_major=element_line(color='white', size=1),
            panel_grid_minor=element_line(color='white', size=0.5),
            panel_spacing=0.07,
            panel_spacing_x=0.07,
            panel_spacing_y=0.07,
            panel_ontop=True,

            strip_background=element_rect(fill=background_color, color='None'),
            strip_margin=0,
            strip_margin_x=None,
            strip_margin_y=None,
            strip_text=element_text(color='#1A1A1A', size=base_size*0.8,
                                    linespacing=1.0),
            strip_text_x=element_text(
                margin={'t': half_line/2, 'b': half_line/2}),
            strip_text_y=element_text(
                margin={'l': half_line/2, 'r': half_line/2},
                rotation=-90),

            plot_background=element_rect(color='white'),
            plot_title=element_text(size=base_size*1.2,
                                    margin={'b': half_line*1.2,
                                            'units': 'pt'},
                                    ha="center",
                                    weight="heavy"),
            plot_margin=None,

            complete=True)


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

    def __add_threshold_crossings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds points in the data where the line crosses the 0.5 threshold.
        """
        # first step is to find rows (i, i+1) where the win_prob crosses 0.5
        # create a shifted win_prob column such that each row i has the i+1th
        # win_prob in the shifted_win_prob column
        df["seconds_shifted"] = df["seconds"].shift(-1)
        df["win_prob_shifted"] = df["win_prob"].shift(-1)

        # now determine whether the values cross the 0.5 threshold
        threshold_crossed_rows = np.where(
            ((df["win_prob"] < 0.5) & (df["win_prob_shifted"] > 0.5)) |
            ((df["win_prob"] > 0.5) & (df["win_prob_shifted"] < 0.5))
        )

        for row in threshold_crossed_rows[0]:

            # find the % from point i to i+1
            xp = [df.iloc[row]["win_prob"], df.iloc[row]["win_prob_shifted"]]
            fp = [df.iloc[row]["seconds"], df.iloc[row]["seconds_shifted"]]

            # np.interp requires monotonic increasing xp array
            if xp[0] > xp[-1]:
                xp = xp[::-1]
                # fp = fp[::-1]

            # determine the crossing point in seconds
            crossing_point = np.interp(0.5, xp, fp)

            # build new row for the interpolated point
            new_row = pd.DataFrame({
                "seconds": [crossing_point],
                "win_prob": [0.5],
                "seconds_shifted": [df.iloc[row]["seconds_shifted"]],
                "win_prob_shifted": [df.iloc[row]["win_prob_shifted"]]
                })

            # update original row
            df.at[row, "win_prob_shifted"] = 0.5
            df.at[row, "seconds_shifted"] = crossing_point

            # add new row to df
            df = df.append(new_row)

        return df

    def __stack_df(
        self, df: pd.DataFrame,
        p1_handle: str = "Player 1", p2_handle: str = "Player 2"
    ) -> pd.DataFrame:

        # create a binary "predicted_winner" column
        conditions = [
            (df["win_prob"] < 0.5) & (df["win_prob_shifted"] <= 0.5),
            (df["win_prob"] > 0.5) & (df["win_prob_shifted"] >= 0.5),
            (df["win_prob"] == 0.5) & (df["win_prob_shifted"] < 0.5),
            (df["win_prob"] == 0.5) & (df["win_prob_shifted"] > 0.5),
        ]
        values = [0, 1, 0, 1]
        df["predicted_winner"] = np.select(conditions, values)

        # create a copy of the data and reverse the predicted winner column
        df_p2 = df.copy()
        df_p2["predicted_winner"] = abs(df["predicted_winner"] - 1)

        # create player columns
        # TODO: Update player to player IDs
        df["Player"] = p1_handle
        df_p2["Player"] = p2_handle

        # return appended df
        return df.append(df_p2)

    def win_probability_plot(
        self, df: pd.DataFrame, p1_race: str, p2_race: str,
        match_id: str = "TESTID",
        p1_handle: str = "Player 1", p2_handle: str = "Player 2"
    ) -> ggplot:
        """
        Plots a win probability line graph.

        mappings:
            x axis: Game time in 10 second intervals.
            y axis: Win probability for player 1.
            color: Player.

        geoms:
            line: A line showing the win probability.

        Returns:
            A plotnine ggplot object.
        """

        # add any points where the win_probs crosses the 0.5 threshold
        df = self.__add_threshold_crossings(df=df)

        # stack data ready for plot
        df = self.__stack_df(df=df, p1_handle=p1_handle, p2_handle=p2_handle)

        # build ggplot object
        p = (
            ggplot(df, aes(
                x="seconds",
                y="win_prob",
                color="Player",
                alpha="predicted_winner"
                )
            )
            + geom_hline(yintercept=0.5, linetype="dashed", color="grey")
            + geom_line()
            + xlab("Game Time (seconds)")
            + ylab("Win Probability for Player 1 (%)")
            + ggtitle(f"Win Probability for Match ID: {match_id}")
            + ThemeStarcraft()
            + scale_color_manual(values={
                p1_handle: self.RACE_COLORS[p1_race],
                p2_handle: self.RACE_COLORS[p2_race]
            })
            + scale_x_continuous(breaks=np.arange(start=0, stop=1e6, step=60))
            + guides(alpha=False)
        )

        return p  


if __name__ == "__main__":
    # create data
    test_data = pd.DataFrame(
        data={
            "seconds": np.arange(start=0, stop=600, step=10),
            "win_prob": np.clip(
                np.arange(start=0.3, stop=0.6, step=0.005) + (rand(60)/20),
                a_min=0,
                a_max=1
            )
        }
    )
    pe = PlotEngine()
    # test_data2 = pe.add_threshold_crossings(df=test_data)
    # test_data2 = pe.stack_df(df=test_data2)
    # print(test_data2)
    p = pe.win_probability_plot(
        df=test_data,
        p1_race="t",
        p2_race="z",
        match_id="111111",
        p1_handle="i_play_a_skill_race",
        p2_handle="i_play_a_noob_race"
        )
    print(p)
    # plot.save(filename='test.png', dpi=1000)
