FROM python:3.10-slim-bookworm

ARG TORCH_BACKEND=cpu

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/artifacts/cache/torch \
    HF_HOME=/app/artifacts/cache/huggingface \
    YOLO_CONFIG_DIR=/app/artifacts/cache/ultralytics \
    MPLCONFIGDIR=/app/artifacts/cache/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app
RUN case "$TORCH_BACKEND" in \
      cpu) python -m pip install torch==2.9.1 torchvision==0.24.1 \
             --index-url https://download.pytorch.org/whl/cpu ;; \
      cu128) python -m pip install torch==2.9.1 torchvision==0.24.1 \
             --index-url https://pypi.org/simple ;; \
      *) echo "TORCH_BACKEND must be cpu or cu128" >&2; exit 1 ;; \
    esac \
    && python -c 'import os, torch; expected = "12.8" if os.environ["TORCH_BACKEND"] == "cu128" else None; assert torch.version.cuda == expected, (torch.version.cuda, expected)'
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt \
    && python -m pip check

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/reference_dataset/ ./data/reference_dataset/
RUN python -m pip install --no-deps --no-build-isolation . \
    && mkdir -p data/forest_vehicles data/visdrone \
       artifacts/weights artifacts/reference_indexes artifacts/cache/ultralytics outputs

ENTRYPOINT ["python", "-m", "vehicle_recognition"]
CMD ["--help"]
