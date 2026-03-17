PYTHON = python
PIP = pip
VENV = .venv

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip && $(PIP) install -e . && $(PIP) install pre-commit && pre-commit install

lint:
	ruff check .
	black --check .
	isort --check-only .

test:
	pytest --cov=src tests/

format:
	black .
	isort .
	ruff check . --fix

up:
	docker-compose up -d

down:
	docker-compose down

validate-data:
	$(PYTHON) src/data_validation.py

train:
	$(PYTHON) src/train.py
