FROM astral/uv:python3.12-bookworm AS builder

# 1. Install the Rust toolchain and C build tools
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="~/.cargo/bin:${PATH}"

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"

# 3. Copy ALL necessary files: dependency definitions AND the Rust source code
COPY pyproject.toml uv.lock ./
COPY lib/ ./lib/

RUN uv sync --frozen


FROM python:3.12-bookworm

COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

WORKDIR /app
