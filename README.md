![CICD](https://github.com/nedwebster/starcraft_predictor/actions/workflows/cicd.yml/badge.svg)

# starcraft_predictor
`starcraft_predictor` contains a pre-trained XGBoost model that can be used to generate win probabilities throughout a Starcraft 2 game. The package uses `.SC2Replay` files to load in game metadata and generate predictions based on the state of the game at every 10 second interval.


## Setup
The package is not currently on PyPi so the best way to use it locally is to clone the repo

```
git clone https://github.com/nedwebster/starcraft_predictor.git
```

Then navigate into the directory and run 

```
pip install .
```

## Usage

To analyse a replay, use the `score_replay` function from `starcraft_predictor`.

Run the following code in a notebook:

```
from starcraft_predictor import score_replay

score_replay("path/to/replay.SC2Replay")
```

![](example_data/sc2_plot.png)


## Folder Structure
```
.
├── example_data/                        <-- example data to use in the tutorials 
├── example_notebooks/                   <-- notebook tutorials for app functionality
├── ml_analysis/                         <-- adhoc analysis that guides parts of the ML project
├── scripts/                             <-- python scripts used to train the model and score replays 
└── src/starcraft_predictor/             <-- main app folder/
    ├── modelling/                       <-- code for the ml model
    ├── plots/                           <-- code for plotting replay predictions
    ├── processing/                      <-- code for processing data ready for the ML model
    ├── replays/                         <-- code 
```

## Testing
The package uses `pytest` as it's testing framework. It also uses the `pytest-mpl` addon for testing plots (documentation can be found here: https://github.com/matplotlib/pytest-mpl).

Baseline plots have been created for `pytest-mpl` with the command:
```pytest --mpl-generate-path=tests/baseline```
This command does not need to be re-run by the user, unless changes are made that alter the output of the plots.

To run the tests locally, use the following command:

```pytest --mpl tests/.```

## TODO
- Additional feature to model
- Explore alternative modelling techniques
- Conformal predictors for assessing model confidence