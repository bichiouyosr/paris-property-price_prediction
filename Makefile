.PHONY: install data run tune test notebook clean

install:
	pip install -e ".[notebook,dev]"

data:
	python scripts/make_synthetic_data.py --rows 100000

run:
	python scripts/run_pipeline.py --data data/listings_synthetic.csv

tune:
	python scripts/tune_hyperparameters.py --data data/listings_synthetic.csv

test:
	pytest

notebook:
	jupyter lab notebooks/

clean:
	rm -rf reports/figures/*.png reports/*.csv reports/metrics.json models/*.joblib
	find . -name __pycache__ -type d -exec rm -rf {} +
