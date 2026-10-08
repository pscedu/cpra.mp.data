FROM python:3.12.6-slim
LABEL maintainer="mjyoder@psc.edu"
LABEL org.opencontainers.image.source="https://github.com/pscedu/cpra.mp.data"

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /bin/uv

# Compile bytecode at build time, since the pod runs with a read-only root
# filesystem and can't write .pyc files later. Copy rather than hardlink out of
# the cache mount, which isn't part of the image. Use the image's Python, never
# a uv-managed download.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /usr/src/app

# Dependencies first, from uv.lock alone, so this layer is reused until the
# lockfile or pyproject.toml changes. --locked fails the build if the two have
# drifted apart instead of silently re-resolving (which is how pip ended up
# installing a newer major Polars than the lock allows).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --extra api --extra scripts

COPY pyproject.toml \
    README.md \
    uv.lock ./

COPY cpra/ ./cpra

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra api --extra scripts

ENV PATH="/usr/src/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["python3", "-u", "-m", "cpra.mp.data.api"]
