FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["bash", "-lc", "uvicorn app.api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard/app.py --server.address 0.0.0.0 --server.port 8501"]
