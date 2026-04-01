FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY config/ config/
COPY prompts/ prompts/

# FIX #18: Non-editable install for production
# NOTE: src/ must be copied BEFORE install — hatchling needs it to build
RUN pip install --no-cache-dir "."
RUN python -m spacy download en_core_web_lg

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
