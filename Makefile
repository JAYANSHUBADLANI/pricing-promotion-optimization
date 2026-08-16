.PHONY: help install test lint pipeline clean app

help:
	@echo "install      install dependencies"
	@echo "pipeline     run the full pipeline end to end"
	@echo "test         run the test suite"
	@echo "app          launch the scenario tool"
	@echo "clean        remove generated outputs and caches"

install:
	pip install -r requirements.txt

pipeline:
	python run_pipeline.py

test:
	pytest

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf outputs/figures/*.png outputs/tables/*.csv outputs/tables/*.json
	rm -rf outputs/models/*.nc outputs/models/*.npy
	rm -rf data/interim/*.parquet data/processed/*.parquet
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage
