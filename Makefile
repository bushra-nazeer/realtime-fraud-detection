.PHONY: install train evaluate stream-demo serve test lint format clean

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

train:
	uv run python -m fraud.train

evaluate:
	uv run python -m fraud.evaluate

# Runs an in-memory streaming simulation end-to-end (no Kafka broker needed).
stream-demo:
	uv run python -m fraud.stream.scorer

serve:
	uv run uvicorn fraud.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf models/*.pkl models/metadata.json mlruns reports/figures .pytest_cache .ruff_cache
