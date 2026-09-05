# Paris Residential AVM

Predicts the asking price of a Paris apartment from its listing details. Built from a 100,000 row extract of real Paris listings. That extract isn't included in this repo, only a script that generates a fake one with the same shape, see Running it below.

## Input and output

Input: one listing per row. After cleaning and feature engineering there are 32 columns feeding the model, built from an original 35 column raw file.

| Column | Type | Meaning |
|---|---|---|
| area | float | living area, sqm |
| room_count | float | rooms |
| bedroom_count | float | bedrooms |
| bathroom_count | float | bathrooms |
| toilet_count | float | separate toilets |
| floor | float | floor of the unit |
| floor_count | float | floors in the building |
| floor_ratio | float | floor divided by floor_count |
| build_year | float | year built |
| property_age | float | years since built |
| arrondissement | int | 1 to 20 |
| rive_gauche | 0/1 | left bank flag |
| prime_arrondissement | 0/1 | flag for the priciest districts |
| has_passenger_lift | 0/1 | has a lift |
| has_cellar | 0/1 | has a cellar |
| balcony_count | float | balconies |
| terrace_count | float | terraces |
| outdoor_area | float | terrace plus balcony area, sqm |
| total_parking | float | parking spots |
| energy_score_ordinal | int | energy rating, 1 worst to 7 best |
| area_per_room | float | area divided by room_count |
| area_per_bedroom | float | area divided by bedroom_count |
| listing_month, listing_quarter, listing_dow | int | when it was listed |
| type_*, subtype_* | 0/1 | property type, one hot encoded |

Output: `price`, the predicted asking price in EUR.

Full raw schema and how each feature gets built: [docs/data_dictionary.md](docs/data_dictionary.md).

## Results

Tested on a held out 20% of the data, 18,336 listings. These numbers are from the real extract, a local run on the generated synthetic data will land in a similar range but not match exactly.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | EUR 44,957 | EUR 94,047 | 0.974 |
| XGBoost | EUR 67,865 | EUR 112,275 | 0.963 |
| Gradient Boosting | EUR 86,347 | EUR 136,226 | 0.946 |
| Linear Regression | EUR 119,859 | EUR 184,636 | 0.900 |
| Lasso | EUR 119,866 | EUR 184,660 | 0.900 |
| Ridge | EUR 119,867 | EUR 184,661 | 0.900 |

A few things worth knowing:

Area alone accounts for about 90% of what the Random Forest relies on, then arrondissement and a flag for the priciest districts.

Error in euros grows with price (about EUR 25,000 on the cheapest fifth of listings, EUR 77,000 on the priciest), even though the percentage error actually shrinks.

About 1 row in 12 in the raw extract got dropped entirely: duplicates, rows that weren't homes at all, values way outside a believable range. Everything else with a bad field just got that field blanked, keeping the row.

## Running it

```bash
pip install -e .
python scripts/make_synthetic_data.py --rows 100000
python scripts/run_pipeline.py --data data/listings_synthetic.csv
```

That installs the package properly, so nothing needs a manual path hack, then generates a synthetic extract shaped like the real one, then runs the whole pipeline: clean the data, build features, train six models, and write out diagnostics and figures.

Everything lands in `reports/`: `metrics.json`, `model_comparison.csv`, `feature_importance.csv`, `cleaning_log.csv`, and eleven figures under `reports/figures/`. The trained model gets saved to `models/best_model.joblib`.

To point it at a real file instead, drop it in `data/` and pass it with `--data`. It reads `.xlsx`, `.csv`, and `.parquet`.

`make install` runs the setup above with the notebook and test extras included, and `make run`, `make tune`, `make test` wrap the commands here.

The leaderboard above comes from sensible defaults, not a tuned model. `scripts/tune_hyperparameters.py` fills that gap. It runs a randomized search, twenty draws with cross validation across three folds by default, around fifteen to twenty minutes on the full extract, over the Random Forest's settings, then reports the tuned result against the default side by side.

```bash
python scripts/tune_hyperparameters.py --data data/listings_synthetic.csv
python scripts/tune_hyperparameters.py --model "XGBoost" --n-iter 60 --save
```

Results get written to `reports/hyperparameter_search.json`. The save flag is what actually replaces `models/best_model.joblib`, so a search can be reviewed before it changes the deployed model.

## Layout

```
src/paris_avm/
├── config.py       every threshold and rule in one place
├── data.py         loading and the initial data audit
├── cleaning.py     the nine cleaning steps, each one logged
├── features.py     the fourteen engineered features
├── modeling.py     split, target check, six models, leaderboard
├── evaluation.py   error by segment and group, importance, intervals
└── plots.py        eleven matplotlib figures, saved as PNG

scripts/
├── make_synthetic_data.py    builds a fake extract with the real schema
├── run_pipeline.py           runs everything end to end
└── tune_hyperparameters.py   randomized search over the tree models

notebooks/
├── 00_original_exploration.ipynb   the original notebook, outputs cleared
└── 01_analysis_walkthrough.ipynb   a shorter version that calls the package

reports/            a sample run's output, figures and a slide deck

tests/              17 tests pinning the cleaning rules and the leakage guards
```

Thresholds live in `config.py` instead of being scattered through the code, so the rules can be argued about, and changed, without touching the code that applies them.
