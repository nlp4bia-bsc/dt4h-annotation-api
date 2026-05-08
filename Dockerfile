FROM python:3.10-slim

WORKDIR /NERL_API
# Install uv
RUN pip install uv
# Copy dependency metadata first (better layer caching)
COPY pyproject.toml uv.lock ./
# Install deps using uv
RUN uv sync

COPY app /NERL_API/app

# Instalar dependencias necesarias y Docker CLI
RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/*


ENV FLASK_APP=app
ENV FLASK_ENV=development

CMD uv run python -m app.initializer && uv run flask run --host=0.0.0.0 --port=5000 --reload