# Nim: Minimax, Alpha–Beta Pruning & EDA

A compact Python project exploring the impartial game **Nim**. It implements Minimax and Alpha–Beta pruning, compares search effort on the starting position `(3, 4, 5)`, simulates 200 seeded games (optimal A vs. random B), and analyzes the results with pandas and Matplotlib. A notebook walks through the game, search, simulation, exploratory data analysis (EDA), and an animated sample game.

## Quick start

Requirements: Python 3.8+ and `numpy`, `pandas`, `matplotlib`.

```bash
python -m pip install numpy pandas matplotlib
python nim_minimax_eda.py
```

The notebook can also be run interactively:

```bash
jupyter notebook lab3_nim_minimax_eda.ipynb
```

Run its cells from top to bottom. The simulation uses a fixed random seed for reproducibility and writes `nim_games.csv`; the notebook creates the plots and animation.

## Key results: search nodes on `(3, 4, 5)`

All methods return `+1` (A wins with perfect play). Node counts are recursive game positions visited; Alpha–Beta depends on move ordering.

| Method | Move ordering | Nodes visited | Fewer nodes than Minimax |
|---|---|---:|---:|
| Minimax | Natural (small pile/take first) | 1,038,768 | — |
| Alpha–Beta | Natural | 33,916 | 96.7% |
| Alpha–Beta | Big-first | 4,332 | 99.6% |

## Analytic insights (200 games)

| Observation | Reason | Conclusion |
|---|---|---|
| A wins 87.5% of the starting positions under perfect play. | 87.5% of sampled positions have a nonzero Nim-sum; this exactly matches the optimal first-player result in the dataset. | A nonzero Nim-sum marks a winning position for the player to move under optimal play. |
| Alpha–Beta visits only 2.5% as many nodes as Minimax in aggregate. | Alpha–Beta cuts off branches that cannot change the decision; good move ordering enables earlier cutoffs. | Search ordering matters: on `(3, 4, 5)`, big-first reduces visits from 1,038,768 to 4,332. |
| Smart A wins 98% of simulated games against random B. | A selects an optimal move while B chooses a legal move at random. | Random play is usually exploitable, although the result depends on the sampled starts and random choices. |

## Project files

- `nim_minimax_eda.py` — Python implementation and analysis script.
- `nim_games.csv` — 200 simulated games and their recorded features/results.
- `nim_eda.png` — EDA charts.
- `nim_game.gif` — Nim game animation.
- `lab3_nim_minimax_eda.ipynb` — Interactive walkthrough and analysis notebook.

*The results above are calculated from the supplied implementation and `nim_games.csv`. The CSV has 200 rows; repeated rows can occur because sampled games may share the same recorded values.*
