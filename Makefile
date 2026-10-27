# Linux/macOS convenience targets. Windows users: see the PowerShell commands in README.md.
PYTHON ?= .venv/bin/python
PORT ?= 8000

.PHONY: venv install install-spark data train evaluate benchmark api smoke stream test docker-build docker-run clean

venv:
	python3 -m venv .venv

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --no-deps -e .

install-spark:
	$(PYTHON) -m pip install -r requirements-spark.txt

data:
	$(PYTHON) scripts/generate_data.py

train:
	$(PYTHON) scripts/train_model.py

evaluate:
	$(PYTHON) scripts/evaluate_model.py

benchmark:
	$(PYTHON) scripts/benchmark_inference.py

api:
	PORT=$(PORT) $(PYTHON) scripts/run_api.py

smoke:
	$(PYTHON) scripts/api_smoke_test.py --url http://localhost:$(PORT)

stream:
	$(PYTHON) scripts/run_streaming.py --reset

test:
	$(PYTHON) -m pytest

docker-build:
	docker build -t vehicle-telemetry-anomaly-api:latest .

docker-run:
	docker run --rm -p $(PORT):8000 -e PORT=8000 vehicle-telemetry-anomaly-api:latest

clean:
	rm -rf data/processed data/streaming_output data/checkpoints .pytest_cache
