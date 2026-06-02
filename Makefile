.PHONY: install data db train serve test docker-up docker-down clean

install:
	pip install -r requirements.txt

data:
	python data/generate_data.py --rows 50000

db:
	python scripts/setup_database.py --rows 50000

train:
	python -m src.train

train-csv:
	python -m src.train --from-csv data/credit_risk.csv

serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ .pytest_cache mlruns models/*.joblib models/metrics.json
