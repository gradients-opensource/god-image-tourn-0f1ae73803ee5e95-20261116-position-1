#!/bin/sh
# Installs AI Toolkit e9c72b76 (the PR #1420 evaluator's pin) into its own venv /opt/aitk for comfytrain/g56_newfam.py.
# Called by ops/docker/standalone-image-toolkit-trainer.dockerfile under `timeout`; ANY failure exits non-zero and the
# Dockerfile then removes /opt/aitk (fail-soft: the PR #1420 families ship the zero LoRA, every other family is unaffected).
# The venv sees the image's torch 2.9.1 (--system-site-packages); every package AI Toolkit pins comes from
# ops/docker/aitk-lock-e9c72b76.txt, installed with --no-deps (nothing else is resolved; the system site-packages stay).
set -eu
LOCK=${1:-/tmp/aitk-lock-e9c72b76.txt}
COMMIT=e9c72b762748e4d31837fbeb3b87d01a5b143321
i=1
while :; do
    if python - "$COMMIT" <<'EOF'
import io, os, sys, tarfile, urllib.request
u = "https://codeload.github.com/ostris/ai-toolkit/tar.gz/" + sys.argv[1]
b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "g56-build"}), timeout=300).read()
t = tarfile.open(fileobj=io.BytesIO(b))
root = t.getmembers()[0].name.split("/")[0]
t.extractall("/opt/aitk-tmp")
os.makedirs("/opt/aitk", exist_ok=True)
os.rename("/opt/aitk-tmp/" + root, "/opt/aitk/src")
EOF
    then break; fi
    [ "$i" -ge 4 ] && exit 1
    echo "ai-toolkit download attempt $i failed, retrying"; rm -rf /opt/aitk-tmp /opt/aitk/src; sleep 15; i=$((i + 1))
done
rm -rf /opt/aitk-tmp
# Stable Diffusion LDM key maps (unused by these families) and the repository's sample picture (no code opens it): no weight- or data-like file of the source tree in the image
rm -f /opt/aitk/src/toolkit/keymaps/*.safetensors /opt/aitk/src/assets/VAE_test1.jpg
python -m venv --system-site-packages /opt/aitk
i=1
until uv pip install --python /opt/aitk/bin/python --no-deps -r "$LOCK"; do
    [ "$i" -ge 3 ] && exit 1
    echo "ai-toolkit requirements attempt $i failed, retrying"; sleep 20; i=$((i + 1))
done
# cv2: AI Toolkit imports it on every job (jobs/process -> toolkit/image_utils.py:11); without it the runner dies at import.
/opt/aitk/bin/python -c "import torch, transformers, diffusers, optimum.quanto, torchao, bitsandbytes, cv2; \
assert torch.__version__.startswith('2.9.1'), torch.__version__; assert transformers.__version__ == '5.5.3', transformers.__version__"
echo "ai-toolkit environment ready"
