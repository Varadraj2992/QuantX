# QuantX operations guide

## Local development

1. Create and activate a virtual environment.
2. Install dependencies.
3. Start the API and dashboard separately or with Docker Compose.

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
streamlit run app/dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

## Docker

```bash
docker compose up --build
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- PostgreSQL: localhost:5432

## Validation

```bash
pytest -q
ruff check src tests
```

## Environment variables

Use the project root `.env` file based on `.env.example` and keep any secrets out of source control.
