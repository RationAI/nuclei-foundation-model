FROM rocm/dev-ubuntu-24.04 AS builder


# 1. Install the Rust toolchain and C build tools
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="~/.cargo/bin:${PATH}"

# 2. Install Uv from the prebuilt image
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT="/usr/"

# 3. Copy ALL necessary files: dependency definitions AND the Rust source code
COPY pyproject.toml uv.lock ./
COPY lib/ ./lib/

RUN uv sync --frozen


FROM rocm/dev-ubuntu-24.04

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/

WORKDIR /app
