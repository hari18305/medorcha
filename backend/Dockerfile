FROM python:3.11-slim

WORKDIR /app

# System deps: pymupdf/torch/open_clip pull in a few native libs that
# aren't in the slim base. Kept minimal and combined into one layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch + torchvision first (they must come from the same index, or torchvision's compiled ops
# fail against torch), from the CPU-only wheel index: the plain `pip install torch` on Linux pulls the CUDA build
# plus ~3 GB of nvidia-* libraries, which this container can never use (compose requests no GPU). Once
# they are installed, requirements.txt's unpinned `torch` (and open_clip's torchvision) are already satisfied.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend code is bind-mounted over this at dev time (see
# docker-compose.yml's `volumes:`), but we still COPY it so the image is
# runnable standalone (e.g. `docker run` without compose, or a future
# non-dev/prod compose profile without the bind mount).
COPY backend/ ./backend/
COPY data/ ./data/

EXPOSE 8000

# --reload is fine for this prototype's dev-oriented compose setup (bind
# mount + live reload matches how the backend is run outside Docker too,
# per README). Swap for a plain `uvicorn backend.main:app --host 0.0.0.0
# --port 8000` if/when a non-dev deployment profile is added.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
