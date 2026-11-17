#!/bin/sh
# Installs Musubi Tuner 30c658c4 (hidream-o1 edit: the PR #1420 evaluator's pin) into its own venv /opt/musubi for
# comfytrain/g56_newfam.py. Called by ops/docker/standalone-image-toolkit-trainer.dockerfile under `timeout`; ANY failure exits
# non-zero and the Dockerfile then removes /opt/musubi (fail-soft: hidream-o1 edit keeps only the zero LoRA, every other family is
# unaffected). The venv sees the image's torch 2.9.1 (--system-site-packages); every other package comes from
# ops/docker/musubi-lock-30c658c4.txt, installed with --no-deps (nothing else is resolved; the system site-packages stay).
set -eu
LOCK=${1:-/tmp/musubi-lock-30c658c4.txt}
COMMIT=30c658c4f4b0bf05038b3346eff9670259b10fc7
i=1
while :; do
    if python - "$COMMIT" <<'PY'
import io, os, sys, tarfile, urllib.request
u = "https://codeload.github.com/kohya-ss/musubi-tuner/tar.gz/" + sys.argv[1]
b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "g56-build"}), timeout=300).read()
t = tarfile.open(fileobj=io.BytesIO(b))
root = t.getmembers()[0].name.split("/")[0]
t.extractall("/opt/musubi-tmp")
os.makedirs("/opt/musubi", exist_ok=True)
os.rename("/opt/musubi-tmp/" + root, "/opt/musubi/src")
PY
    then break; fi
    [ "$i" -ge 4 ] && exit 1
    echo "musubi-tuner download attempt $i failed, retrying"; rm -rf /opt/musubi-tmp /opt/musubi/src; sleep 15; i=$((i + 1))
done
rm -rf /opt/musubi-tmp
python -m venv --system-site-packages /opt/musubi
i=1
until uv pip install --python /opt/musubi/bin/python --no-deps -r "$LOCK" && \
      uv pip install --python /opt/musubi/bin/python --no-deps /opt/musubi/src; do
    [ "$i" -ge 3 ] && exit 1
    echo "musubi requirements attempt $i failed, retrying"; sleep 20; i=$((i + 1))
done
# the venv's own versions (not the system's): Musubi pins transformers 4.57.6 / diffusers 0.32.1; torch is the shared 2.9.1
/opt/musubi/bin/python -c "import torch, transformers, diffusers, accelerate, cv2, musubi_tuner, musubi_tuner.networks.lora_hidream_o1; \
assert torch.__version__.startswith('2.9.1'), torch.__version__; assert transformers.__version__ == '4.57.6', transformers.__version__; \
assert diffusers.__version__ == '0.32.1', diffusers.__version__"
[ -x /opt/musubi/bin/accelerate ]
echo "musubi environment ready"
