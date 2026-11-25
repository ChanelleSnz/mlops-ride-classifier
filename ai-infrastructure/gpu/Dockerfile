# CPU vs GPU benchmark image for Vertex AI custom jobs.
# Small on purpose: code + data + model only (~50 MB). The pinned dependencies (PyTorch 2.15.0 CUDA build from
# PyPI, same versions as uv.lock) are installed when the job starts, inside Google's network (see run.sh).
# Built by build.sh, which prepares the build context.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    # Ask the NVIDIA container runtime to mount the GPU and its driver (nvidia/cuda images set these;
    # without them the T4 is attached to the machine but invisible inside the container)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Vertex AI mounts the driver (nvidia-smi, libcuda) in /usr/local/nvidia, like Google's own GPU images
    PATH=/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib

COPY . /app
ENTRYPOINT ["bash", "/app/run.sh"]
