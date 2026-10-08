# QuantX

QuantX is a production-style quantitative finance analytics platform designed to support data collection, validation, strategy research, portfolio analysis, risk monitoring, and ML signal generation for institutional-style research workflows.

## Phase status

This repository is currently in Phase 15: Final GitHub preparation.

The project structure is being prepared to support:
- market data ingestion and validation
- quant strategy research and backtesting
- portfolio optimization and risk analytics
- ML signal generation and model evaluation
- FastAPI service exposure
- Streamlit dashboarding
- Power BI-ready analytical exports
- Dockerized local execution

## Architecture overview

```text
Market Data
  ↓
Data Ingestion Layer
  ↓
Data Validation & Cleaning
  ↓
PostgreSQL
  ↓
Feature Engineering
  ↓
Strategy Engine / ML Engine
  ↓
Backtesting Engine
  ↓
Portfolio & Risk Analytics
  ↓
Optimization
  ↓
FastAPI + Streamlit
  ↓
Power BI-ready exports
```

## Repository structure

```text
QuantX/
├── app/
│   ├── api/
│   └── dashboard/
├── src/
│   ├── data/
│   ├── features/
│   ├── strategies/
│   ├── backtesting/
│   ├── portfolio/
│   ├── risk/
│   ├── optimization/
│   └── ml/
├── database/
│   ├── models/
│   └── migrations/
├── docs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── exports/
├── tests/
├── scripts/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── README.md
└── ...
```

## Technical stack

- Python 3.11+
- Pandas, NumPy, SciPy
- Scikit-learn, XGBoost
- PostgreSQL + SQLAlchemy
- FastAPI
- Streamlit + Plotly
- Pytest
- Docker + Docker Compose

## Environment setup

1. Create a virtual environment.
2. Copy `.env.example` to `.env`.
3. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Running locally

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
streamlit run app/dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

## Docker

```bash
docker compose up --build
```

For app startup and validation commands, see [docs/operations.md](docs/operations.md).

## Important financial assumptions

This project is designed for research and analytics workflows. It does not provide investment advice or guarantee future returns. All strategy evaluation must be interpreted with caution, including consideration of market regime change, transaction costs, slippage, and survivorship bias.

## Disclaimer

QuantX is a research project for education and portfolio analytics. It is not a substitute for professional investment analysis or a guarantee of financial performance.

## Current focus

The current phase focuses on final repository polish, CI/CD readiness, and GitHub hygiene so the project is cleanly prepared for team review and upstream collaboration.
