![CICD](https://github.com/nedwebster/starcraft_predictor/actions/workflows/cicd.yml/badge.svg)

# starcraft_predictor
`starcraft_predictor` contains a pre-trained ML model that can be used to generate win probabilities throughout a Starcraft 2 game. The package uses `.SC2Replay` files to load in game metadata and generate predictions based on the state of the game at every 10 second interval.


## Setup
The package is not currently on PyPi so the best way to use it locally is to clone the repo

```
git clone https://github.com/nedwebster/starcraft_predictor.git
```

The project uses `uv` for package management and project builds. Install the dependencies and the package using:

```
uv sync --reinstall
```

## Usage

To analyse a replay, load the `replay_scorer` and use the `score_replay()` method. The `load_scorer()` function loads the pre-trained model, and can score replays given a path.

Run the following code in a notebook:

```
from starcraft_predictor import load_scorer

replay_scorer = load_scorer()

replay_scorer.score_replay("path/to/replay.SC2Replay")
```

![](example_data/sc2_plot.png)


## Web App
The project uses streamlit to deploy the application as a simple web app. To run the streamlit app, run:

```
uv run streamlit run app.py
```

From there you can upload replays from your local machine and get win probability prediction plots.

## Folder Structure
```
.
├── example_data/                        <-- example data to use in the tutorials 
├── example_notebooks/                   <-- notebook tutorials for app functionality
├── ml_analysis/                         <-- adhoc analysis notebooks that guides parts of the ML project
├── scripts/                             <-- python scripts used to train the model and score replays 
├── app.py                               <-- streamlit app file for running the streamlit web app.
└── src/starcraft_predictor/             <-- main app folder/
    ├── modelling/                       <-- code for the ml model
    ├── plots/                           <-- code for plotting replay predictions
    ├── processing/                      <-- code for processing data ready for the ML model
    ├── replays/                         <-- code 
```
