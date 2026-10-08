PYTHON ?= python
VENV ?= .venv
PIP := $(VENV)/Scripts/pip.exe
PYTHON_VENV := $(VENV)/Scripts/python.exe

.PHONY: install setup test lint api dashboard docker-up docker-down

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup: install

api:
	$(PYTHON_VENV) -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON_VENV) -m streamlit run app/dashboard/app.py --server.address 0.0.0.0 --server.port 8501

test:
	$(PYTHON_VENV) -m pytest -q

lint:
	$(PYTHON_VENV) -m ruff check src tests

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
