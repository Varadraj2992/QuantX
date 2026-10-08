from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from quantx.data.ingestion import generate_sample_market_data

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_market_data.csv"


def main() -> None:
    data = generate_sample_market_data(
        tickers=["AAPL", "MSFT", "NVDA", "AMZN", "SPY"],
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(data)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
