# QuantX Architecture

## Layered structure

The project is organized into a clean separation of concerns:

- Data: downloads, validation, and feature generation
- Strategy: signal generation and parameterization
- Backtesting: trade execution and performance analytics
- Portfolio: asset weighting and allocation logic
- Risk: volatility, VaR, drawdown, and stress testing
- ML: classification/regression signals with time-aware validation
- API: FastAPI service exposure and JSON contracts
- Dashboard: Streamlit applications for interactive reporting
- BI: Power BI-ready analytical tables and DAX measures

## Design principles

- modular and testable components
- time-series-safe evaluation methods
- no hard-coded secrets
- reproducible configuration via environment variables
- professional transaction-cost assumptions and logging
