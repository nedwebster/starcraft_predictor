import matplotlib.pyplot as plt
from starcraft_predictor import load_scorer


def score_replay(replay_path: str):
    """
    Score a replay file and return the predicted probabilities.
    """
    replay_scorer = load_scorer()
    fig = replay_scorer.score_replay(replay_path)
    return fig


def main():
    replay_path = "/Users/nedwebster/Documents/python_projects/personal_projects/starcraft_predictor/example_data/example_replay.SC2Replay"
    score_replay(replay_path)
    plt.savefig("figure.png")


if __name__ == "__main__":
    main()
