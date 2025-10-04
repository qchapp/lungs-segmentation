# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal OS deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Copy project metadata first (better build cache)
COPY pyproject.toml setup.cfg README.md LICENSE ./

# Copy source BEFORE installing so packaging sees the code
COPY src ./src

# Install the project and its deps
RUN pip install --upgrade pip && pip install .

# Default entrypoint: single-image prediction CLI
ENTRYPOINT ["uls_predict_image"]
CMD ["--help"]
