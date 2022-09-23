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

To analyse a replay, use the `api` module from `starcraft_predictor``

```
from starcraft_predictor import api

api.evaluate_replay("path/to/replay.SC2Replay")
```
