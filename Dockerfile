FROM python:3.11-slim

# xgboost wheels need OpenMP runtime
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Model artifacts live in repo under outputs/
COPY outputs ./outputs
COPY api ./api

ENV MODEL_REPORT_PATH=outputs/pref_xgb_fe_report.json \
    RATING_SCALE_PATH=outputs/pref_rating_scale_10pt.json \
    TZ=Asia/Shanghai

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
