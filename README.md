# QuantX

![CI](https://github.com/Varadraj2992/QuantX/actions/workflows/ci.yml/badge.svg)

QuantX is a production-inspired quantitative finance research and analytics platform designed for strategy backtesting, machine learning-based market signals, portfolio optimization, benchmark comparison, and risk analytics.

The platform combines data engineering, quantitative analysis, machine learning, portfolio management, API development, and interactive visualization into a modular end-to-end system.

---

## 1. Overview

QuantX provides an integrated environment for researching and evaluating quantitative investment strategies.

The platform supports:

- Market data ingestion and validation
- Feature engineering and financial indicators
- Quantitative strategy development
- Historical strategy backtesting
- Machine learning-based market signals
- Portfolio optimization
- Portfolio performance analysis
- Risk analytics and monitoring
- Benchmark comparison
- FastAPI-based service exposure
- Interactive Streamlit dashboard
- Power BI-ready analytical exports
- Automated testing and code quality checks
- Docker-based local deployment

QuantX is designed as a research and analytics platform rather than a live trading system.

---

## 2. Key Features

### 2.1 Market Data & Data Engineering

- Market data ingestion and preprocessing
- Data validation and quality checks
- Structured data storage
- Feature engineering for quantitative analysis

### 2.2 Strategy Research & Backtesting

- Modular strategy architecture
- Historical strategy simulation
- Portfolio equity curve analysis
- Performance comparison against benchmarks
- Strategy performance metrics

### 2.3 Machine Learning Signals

- ML-based market signal generation
- Classification models for signal prediction
- Model evaluation and validation
- Signal probability and confidence analysis

Example:

```text
Asset: AAPL
Signal: HOLD
Probability: 50%
Model: Logistic Regression
Confidence: Medium
```

### 2.4 Portfolio Analytics

QuantX provides portfolio-level analytics to evaluate asset allocation, portfolio performance, diversification, and risk-adjusted returns.

#### Portfolio Performance

- Portfolio return and cumulative performance analysis
- Equity curve visualization
- Daily and cumulative returns
- Performance comparison against market benchmarks

#### Portfolio Optimization

- Data-driven asset allocation
- Risk-return analysis
- Optimization of portfolio composition based on defined objectives

#### Diversification Analysis

- Asset-level contribution analysis
- Portfolio exposure analysis
- Allocation distribution
- Diversification assessment across selected assets

#### Risk-Adjusted Analysis

- Volatility measurement
- Drawdown analysis
- Risk-adjusted performance metrics
- Portfolio risk monitoring

#### Benchmark Comparison

QuantX compares portfolio performance against a selected benchmark to evaluate the strategy's risk-adjusted performance relative to the market.

### 2.5 Risk Analytics

QuantX provides risk analytics to measure portfolio risk, downside exposure, volatility, and potential losses.

#### Risk Metrics

- Portfolio volatility analysis
- Maximum drawdown measurement
- Value-at-Risk (VaR) analysis
- Risk-adjusted performance metrics
- Asset-level risk contribution

#### Drawdown Analysis

- Peak-to-trough drawdown analysis
- Maximum drawdown identification
- Drawdown duration analysis
- Recovery period evaluation

#### Risk Monitoring

- Portfolio risk monitoring
- Asset exposure analysis
- Benchmark-relative risk analysis
- Identification of high-risk portfolio conditions

These analytics help evaluate the stability and risk characteristics of quantitative strategies before comparing their historical performance.

### 2.6 Interactive Dashboard

QuantX includes an interactive Streamlit dashboard that provides a centralized interface for exploring quantitative research and portfolio analytics.

#### Executive Overview

- Portfolio performance overview
- Strategy performance summary
- Benchmark comparison
- Key portfolio metrics

#### Strategy Analytics

- Strategy performance visualization
- Asset-level analysis
- Historical performance trends
- Backtesting results

#### ML Signal Center

- Machine learning-generated market signals
- Signal probability
- Model information
- Confidence level
- Asset-specific signal analysis

#### Portfolio & Risk Visualization

- Portfolio allocation
- Performance and equity curves
- Risk metrics
- Drawdown visualization
- Benchmark comparison

The dashboard allows users to interactively explore model outputs, portfolio performance, strategy results, and risk analytics without directly interacting with the underlying Python modules.

---

## 3. System Architecture

```text
Market Data
     ↓
Data Ingestion
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
Portfolio Optimization
     ↓
FastAPI + Streamlit
     ↓
Power BI-Ready Exports
```

---

## 4. Repository Structure

```text
QuantX/
│
├── app/
│   ├── api/
│   │   └── main.py
│   └── dashboard/
│       └── app.py
│
├── src/
│   └── quantx/
│       ├── data/
│       ├── features/
│       ├── strategies/
│       ├── backtesting/
│       ├── portfolio/
│       ├── risk/
│       ├── optimization/
│       └── ml/
│
├── database/
├── data/
│   └── exports/
├── docs/
├── scripts/
├── tests/
│
├── .github/
│   └── workflows/
│
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 5. Technical Stack

### Programming & Data

- Python 3.11+
- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-learn
- XGBoost

### Database

- PostgreSQL
- SQLAlchemy

### API & Dashboard

- FastAPI
- Streamlit
- Plotly

### Testing & Code Quality

- Pytest
- Ruff
- GitHub Actions

### Deployment & Environment

- Docker
- Docker Compose
- Git
- GitHub

---

## 6. Environment Setup

### 6.1 Clone the Repository

```bash
git clone https://github.com/Varadraj2992/QuantX.git
cd QuantX
```

### 6.2 Create a Virtual Environment

```bash
python -m venv .venv
```

### 6.3 Activate the Virtual Environment

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 6.4 Install Dependencies

```bash
pip install -r requirements.txt
```

### 6.5 Configure Environment Variables

Windows:

```powershell
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Update the required configuration values in `.env`.

---

## 7. Running Locally

### 7.1 Start the FastAPI Service

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 7.2 Start the Streamlit Dashboard

```bash
streamlit run app/dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Dashboard:

```text
http://localhost:8501
```

FastAPI:

```text
http://localhost:8000
```

---

## 8. Running with Docker

Build and start the application:

```bash
docker compose up --build
```

Stop the services:

```bash
docker compose down
```

---

## 9. Testing

QuantX uses Pytest for automated testing and Ruff for code quality validation.

Run the test suite:

```bash
pytest -q
```

Run linting:

```bash
ruff check src tests
```

Current local test status:

```text
37 tests passed
```

---

## 10. CI/CD

QuantX includes GitHub Actions for automated validation.

The CI pipeline performs:

1. Python environment setup
2. Dependency installation
3. Ruff linting
4. Automated test execution

The CI workflow helps ensure that code changes maintain project quality and do not introduce regressions.

---

## 11. Dashboard Screenshots

## Dashboard

QuantX provides an interactive Streamlit dashboard for quantitative research, portfolio analytics, strategy evaluation, benchmark comparison, risk monitoring, and ML-based market signals.

### Executive Overview

![QuantX Executive Overview](docs/screenshots/executive-overview.png)

### Benchmark Comparison

![QuantX Benchmark Comparison](docs/screenshots/benchmark-comparison.png)

### Strategy & Portfolio Analytics

![QuantX Strategy Analytics](docs/screenshots/strategy-analytics.png)

### Risk Dashboard

![QuantX Risk Dashboard](docs/screenshots/risk-dashboard.png)

### ML Signal Center

![QuantX ML Signal Center](docs/screenshots/ml-signal-center.png)
---

## 12. Project Highlights

- Modular quantitative finance architecture
- End-to-end quantitative research workflow
- ML-driven market signal generation
- Strategy backtesting framework
- Portfolio optimization and risk analytics
- Benchmark performance comparison
- REST API using FastAPI
- Interactive analytics dashboard using Streamlit
- PostgreSQL database integration
- Dockerized application environment
- Automated testing with Pytest
- Automated code quality checks with Ruff
- GitHub Actions CI pipeline
- Power BI-ready analytical exports

---

## 13. Project Status

QuantX is an actively maintained quantitative finance research and portfolio analytics platform.

The current implementation focuses on:

- Quantitative strategy research
- Historical backtesting
- Machine learning signals
- Portfolio analytics
- Risk analysis
- Dashboard visualization
- API integration
- Automated testing
- CI/CD validation

---

## 14. Financial Assumptions

QuantX is designed for research and analytical workflows.

Backtesting and model results are historical or simulated and should not be interpreted as guaranteed future performance.

Results may be affected by:

- Market regime changes
- Transaction costs
- Slippage
- Liquidity constraints
- Data quality
- Survivorship bias
- Model assumptions

---

## 15. Disclaimer

QuantX is an educational and research-oriented quantitative finance project.

It does not provide financial advice, investment recommendations, or guarantees of future returns. The platform should not be used as a substitute for professional financial analysis or investment advice.

---

## 16. Author

**Varadraj Bhandare**

M.Sc. Data Science & Big Data Analytics

Areas of interest:

- Data Analytics
- Data Science
- Machine Learning
- Quantitative Finance
- Business Intelligence
- Big Data Analytics

---

## 17. License

This project is intended for educational, research, and portfolio purposes.

---

## 18. Project Goals

QuantX aims to provide a modular foundation for quantitative research by combining financial data processing, machine learning, strategy evaluation, portfolio analytics, and visualization in a single platform.

Future development may include additional strategies, expanded model evaluation, improved data pipelines, and more advanced portfolio and risk analytics.

---

## 19. Development Practices

The project follows software engineering practices including:

- Modular application design
- Separation of application and domain logic
- Automated testing
- Static code quality checks
- Environment-based configuration
- Version control with Git
- Continuous integration with GitHub Actions
- Containerized execution with Docker

---

## 20. Repository

GitHub Repository:

https://github.com/Varadraj2992/QuantX

QuantX is developed as a portfolio project demonstrating practical skills in Python, quantitative finance, machine learning, data analytics, software engineering, and business intelligence.
