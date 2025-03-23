#FROM python:3.12-slim AS builder
#WORKDIR /app
#COPY pyproject.toml uv.lock ./
#RUN uv sync --frozen --no-install-project
#COPY . .
#RUN uv sync --frozen

FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/0.6.9/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ADD . /app
WORKDIR /app

RUN uv sync --frozen

#FROM python:3.12-slim
#COPY --from=builder /app /app
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]