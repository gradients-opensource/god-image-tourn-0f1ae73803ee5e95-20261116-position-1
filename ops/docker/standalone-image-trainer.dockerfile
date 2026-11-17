# The validator builds this file for every image task: python 3.11, ComfyUI 694815f4, torch 2.9.1 cu128 and the
# evaluator's own packages (locked: ops/docker/constraints-2nov.txt) + the ComfyUI-native trainer in comfytrain/.
# Baked: only the evaluator's own public files, sha256-pinned (ops/docker/bake_comfy_assets.py); base model and data
# come from the validator-staged /cache.





FROM python:3.11-slim@sha256:ae952e0a57138a4317b9b61caaa19a8217c2f356803f8d7e63aad08769950937
ARG COMFYUI_COMMIT=694815f498295080a0e15a1502edc9dba841b110
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 \
    COMFY_ROOT=/opt/ComfyUI G56CT_BAKED=/opt/g56/baked G56CT_CKPT_ROOT=/app/checkpoints UV_HTTP_TIMEOUT=120
WORKDIR /app
# Every package version is locked to the tested image (ops/docker/constraints-2nov.txt).

COPY ops/docker/constraints-2nov.txt /tmp/g56-constraints.txt
# Best effort: a C compiler for torch.compile (Triton builds its launcher at run time); bounded,
# and the RUN always exits 0 (without it the trainer trains eagerly).





RUN for i in 1 2; do \
        timeout -k 10 80 sh -c 'export DEBIAN_FRONTEND=noninteractive; o="-o Acquire::Retries=1 -o Acquire::http::Timeout=20"; \
            apt-get $o update && apt-get $o install -y --no-install-recommends gcc libc6-dev' && break; \
        echo "gcc install attempt $i failed or timed out"; [ "$i" = 2 ] || sleep 5; \
    done; rm -rf /var/lib/apt/lists/*; command -v gcc || echo "no C compiler in the image (G56CT_COMPILE_BLOCKS trains eagerly)"
# Every network step retries: any error of a cold build fails the task.

RUN pip install --no-cache-dir uv==0.8.22 && \
    for i in 1 2 3; do \
        uv pip install --system --constraint /tmp/g56-constraints.txt --index-url=https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 && break; \
        [ $i = 3 ] && exit 1; echo "torch install attempt $i failed, retrying"; sleep 20; \
    done
RUN for i in 1 2 3 4; do \
    python -c "import io, tarfile, urllib.request, os; \
u = 'https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/${COMFYUI_COMMIT}'; \
b = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'g56-build'}), timeout=300).read(); \
t = tarfile.open(fileobj=io.BytesIO(b)); root = t.getmembers()[0].name.split('/')[0]; t.extractall('/opt'); \
os.rename('/opt/' + root, '/opt/ComfyUI')" && break; \
    [ $i = 4 ] && exit 1; echo "ComfyUI download attempt $i failed, retrying"; rm -rf /opt/ComfyUI /opt/ComfyUI-*; sleep 15; \
    done && \
    for i in 1 2 3; do \
        uv pip install --system --constraint /tmp/g56-constraints.txt -r /opt/ComfyUI/requirements.txt \
            diffusers==0.39.0 transformers==5.10.2 huggingface-hub==1.18.0 safetensors==0.8.0 pydantic==2.13.4 accelerate==1.6.0 && break; \
        [ $i = 3 ] && exit 1; echo "requirements install attempt $i failed, retrying"; sleep 20; \
    done && \
    python -c "import torch, diffusers, transformers, comfy_kitchen; assert torch.__version__.startswith('2.9.1'), torch.__version__" && \
    echo "g56 package lock: $(pip freeze 2>/dev/null | sort | md5sum | cut -c1-32) (tested 0d8a0d37bab26cd798f3dc18863b71af)"
COPY ops/docker/bake_comfy_assets.py /tmp/bake_comfy_assets.py
RUN python /tmp/bake_comfy_assets.py /opt/g56/baked --flux-te /app/flux && mkdir -p /app/checkpoints /opt/comfy-venv/bin && \
    ln -s /usr/local/bin/python /opt/comfy-venv/bin/python
# flux: the evaluator's own fp16 text encoders (pinned revision) are baked in /app/flux.


COPY comfytrain /workspace/comfytrain
ENTRYPOINT ["python", "/workspace/comfytrain/g56_comfy_train.py"]
