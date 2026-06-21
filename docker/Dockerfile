FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY flashrag/ flashrag/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e ".[all]"

COPY configs/ configs/
COPY examples/ examples/

ENTRYPOINT ["flashrag"]
CMD ["check"]
