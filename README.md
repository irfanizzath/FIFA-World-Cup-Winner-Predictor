# FIFA World Cup 2026 Match Prediction Model

A machine learning project that predicts the outcome of football matches and simulates the FIFA World Cup 2026 knockout bracket using a Monte Carlo approach.

My model predicts that France will defeat Argentina in the finals with a 53% win probability making France the 2026 World Cup champions. 

Semi-Finals:

-France vs Spain (France advances (50%))
-England vs Argentina (Argentina advances (50%))

<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/f6dd5256-4f7f-4d57-bfd7-cf0e34019c28" />


## What it does

Given two national teams, the model estimates the probability of a home win, draw, or away win on neutral ground. Those probabilities are then used to simulate the 2026 World Cup bracket thousands of times to estimate each team's chance of winning the tournament outright.

The pipeline:

1. Loads World Cup match history and qualifier results, plus a global Elo ratings table
2. Builds a time aware, match by match dynamic Elo rating for every team, with no data leakage
3. Computes rolling form features (win rate, goals, shooting accuracy, discipline) over each team's last 5 matches
4. Tunes tournament sample weights via walk forward cross validation
5. Trains and compares three models: Logistic Regression, Random Forest, and XGBoost
6. Trains the best performing model on the full historical dataset
7. Exposes a `predict_match(team_a, team_b)` function for any matchup
8. Runs a Numba accelerated Monte Carlo simulation (500,000 runs) over the real FIFA 2026 bracket structure
9. Prints a deterministic projected bracket from the Round of 32 through to the Final

## Data sources

- World Cup match data (2014, 2018, 2022, 2026 group stage) and qualifier results, provided as an Excel workbook with one sheet per tournament
- Football Elo ratings table from [eloratings.net](https://www.eloratings.net/) downloaded just beforre the Round of 32 matches.

Both files are included in this repo and are read directly by the notebook.

## Getting started

Clone the repo and install the dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Then open `FIFA_World_Cup_Winner_Predictor.ipynb` in Jupyter or Google Colab and run the cells in order. If running in Colab, update the file paths in the "Load Data" cell to point to wherever you've uploaded the two Excel files.

## Requirements

- pandas
- numpy
- scikit-learn
- xgboost
- numba
- matplotlib
- openpyxl (for reading Excel files)

## Repository structure

```
.
├── FIFA_World_Cup_Winner_Predictor.ipynb   # main notebook
├── World Cup matches 2026.xlsx      # match history and qualifiers
├── Football Elo Ratings.xlsx # Elo ratings table
└── README.md
```

## Example usage

```python
predict_match("Argentina", "France")
```

This returns a home win, draw, and away win probability, and (if `verbose=True`) prints a readable breakdown.

## Results

Top 20 teams by simulated win probability, from 500,000 Monte Carlo runs:

![Win probability rankings](assets/win_probabilities.png)

Deterministic bracket walkthrough (most likely winner at each stage):

![Predicted bracket results](assets/bracket_predictions.png)

## Limitations

The model has no awareness of injuries, suspensions, tactical context, or squad changes. It is trained on a relatively small number of major tournaments, so the exact probabilities should be treated as a reasonable baseline rather than a precise forecast. The walk forward validation setup also only provides one meaningful test split given how few World Cups exist in the data, so I would not read too much into small accuracy differences between models.
