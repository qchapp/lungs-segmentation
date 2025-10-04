FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies (git sometimes required by huggingface_hub)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy only project metadata first (better build cache)
COPY pyproject.toml setup.cfg README.md LICENSE ./

# Install the project (this pulls dependencies declared in setup.cfg)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy source (in case editable non-packaged assets are needed at runtime)
COPY src ./src

ENV PYTHONUNBUFFERED=1

# Default entrypoint: single-image prediction CLI. Pass -i argument at runtime.
ENTRYPOINT ["uls_predict_image"]
CMD ["--help"]
