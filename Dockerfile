# Reproducible environment for AFTERIMAGE.
#
# The image holds the locked dependencies only. Run it with the repository
# mounted, so git metadata (needed by the confirmatory guards) and the code
# under test are exactly the checkout:
#
#   docker build --build-arg TORCH_EXTRA=cu130 -t afterimage:cu130 .
#   docker run --rm --gpus all -v "$PWD":/work afterimage:cu130 replay --config configs/pilot.toml
#
# CUDA wheels bundle the CUDA runtime; the host needs only an NVIDIA driver
# and the container toolkit. Use TORCH_EXTRA=cpu for CPU-only checks.

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

ARG TORCH_EXTRA=cu130
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_FROZEN=1 \
    AFTERIMAGE_TORCH=${TORCH_EXTRA}

WORKDIR /build
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --extra "${TORCH_EXTRA}" --no-install-project
# Build the project once so the build backend is cached for the editable
# install against the mounted checkout at run time.
COPY src ./src
RUN uv sync --extra "${TORCH_EXTRA}" && rm -rf /build

WORKDIR /work
RUN git config --system --add safe.directory /work
ENTRYPOINT ["/bin/sh", "-c", "exec uv run --extra \"$AFTERIMAGE_TORCH\" afterimage \"$@\"", "--"]
