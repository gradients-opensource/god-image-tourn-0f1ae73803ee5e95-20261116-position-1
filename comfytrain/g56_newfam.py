#!/usr/bin/env python3
"""

























"""
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import zipfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g56_lorafmt as LF  # noqa: E402

AITK_PY = Path(os.environ.get("G56NF_AITK_PY", "/opt/aitk/bin/python"))   # tests point these at a fake runner
AITK_ROOT = Path(os.environ.get("G56NF_AITK_ROOT", "/opt/aitk/src"))
RUNNER = Path(os.environ.get("G56NF_RUNNER", str(HERE / "g56_aitk_runner.py")))
#
MUSUBI_PY = Path(os.environ.get("G56NF_MUSUBI_PY", "/opt/musubi/bin/python"))     # tests point these at a fake Musubi
MUSUBI_ROOT = Path(os.environ.get("G56NF_MUSUBI_ROOT", "/opt/musubi/src"))       # the Musubi tree (scripts in src/musubi_tuner)
MUSUBI_ACCEL = Path(os.environ.get("G56NF_MUSUBI_ACCEL", str(MUSUBI_PY.parent / "accelerate")))
CACHE = Path(os.environ.get("CACHE_ROOT", "/cache"))
CKPT_ROOT = Path(os.environ.get("G56CT_CKPT_ROOT", "/app/checkpoints"))
WORK = Path(os.environ.get("G56NF_WORK", "/tmp/g56nf"))
LOGFILE = os.environ.get("G56CT_LOGFILE", "/tmp/g56ct.log")
VERBOSE = os.environ.get("G56CT_VERBOSE", "0") == "1"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")   # the EXTENSIONS of the evaluator's dataset reader (+ bmp)

#
#
#
#
#
#
#
#
#
#
#
#
ARCH = {"flux2-klein-4b": "flux2_klein_4b", "qwen-image-edit-2511": "qwen_image_edit_plus", "hidream-o1": "hidream_o1",
        "moss-image-base": "moss_image:base"}
MODEL_BASE = {  # template model sections (quantize / qtype / model_kwargs), per (family, mode)
    ("flux2-klein-4b", LF.T2I): dict(quantize=False, quantize_te=False),
    ("flux2-klein-4b", LF.EDIT): dict(quantize=False, quantize_te=False, model_kwargs=dict(match_target_res=True)),
    ("qwen-image-edit-2511", LF.EDIT): dict(quantize=True, qtype="qfloat8", quantize_te=True, qtype_te="qfloat8", low_vram=False),
    ("hidream-o1", LF.T2I): dict(quantize=False, quantize_te=False, model_kwargs=dict(noise_scale=8.0, noise_scale_inference=8.0)),
    ("moss-image-base", LF.T2I): dict(quantize=True, qtype="convrot8", quantize_te=True, qtype_te="convrot8",
                                      model_kwargs=dict(uncond_skip_layers=[22])),
    ("moss-image-base", LF.EDIT): dict(arch="moss_image:base_edit", quantize=True, qtype="convrot8", quantize_te=True,
                                       qtype_te="convrot8", model_kwargs=dict(edit=True, kv_cache=True, match_target_res=True,
                                                                              uncond_skip_layers=[22])),
}
TRAIN_LR = {"hidream-o1": 4e-05, "qwen-image-edit-2511": 0.0002, "flux2-klein-4b": 0.00014142}          # template lr; 1e-4 for the others
TRAIN_TMAX = {"hidream-o1": 50}          # evaluator max_denoising_steps; 250 for the others
CACHE_TE = {("flux2-klein-4b", LF.T2I): True, ("flux2-klein-4b", LF.EDIT): True, ("moss-image-base", LF.T2I): True}
#
#
#
GC_DEFAULT = {("flux2-klein-4b", LF.T2I): False, ("hidream-o1", LF.T2I): True,
              #
              #
              ("moss-image-base", LF.T2I): True}
#
#
#
#
#
GC_OFF_MAX_TOKENS = {("moss-image-base", LF.T2I): 5000}
#
#
#
#
#
#
#
#
#
GC_STEP_FENCE = {("moss-image-base", LF.T2I): 4601}
GC_STEP_COUNTER = frozenset({("moss-image-base", LF.T2I)})
MOSS_PATCH_PX = 16            # FLUX.2 VAE x8, packed 2 x 2: one image token per 16 x 16 pixels
MOSS_MAX_TEXT_TOKENS = 4096   # AI Toolkit moss_image.py MAX_TEXT_TOKENS: the tokenizer truncates a prompt there
MOSS_TEMPLATE_PAD = 32
CAPTION_DROPOUT = 0.5
RESOLUTION = [1024]
#
#
#
#
#
#
#
#
TEMPLATE_STEPS = {("flux2-klein-4b", LF.T2I): 1000, ("flux2-klein-4b", LF.EDIT): 1000, ("qwen-image-edit-2511", LF.EDIT): 500,
                  ("moss-image-base", LF.EDIT): 500}
TEMPLATE_STEPS[("hidream-o1", LF.T2I)] = 1000
# (the two lines stay one line apart: either value changes, or is picked, without touching the other)
TEMPLATE_STEPS[("moss-image-base", LF.T2I)] = 500
#
#
#
#
#
#
#
MEMBERS_MAX = {("flux2-klein-4b", LF.T2I): 10, ("flux2-klein-4b", LF.EDIT): 1, ("qwen-image-edit-2511", LF.EDIT): 1,
               ("hidream-o1", LF.T2I): 1, ("moss-image-base", LF.T2I): 8, ("moss-image-base", LF.EDIT): 1}
# member length by task size (optional): (steps per training image, lower clamp, upper clamp) -> step_cap() = c x N clamped;
# no entry = TEMPLATE_STEPS
STEPS_PER_IMAGE = {("flux2-klein-4b", LF.T2I): (38, 250, 1000), ("flux2-klein-4b", LF.EDIT): (20, 250, 1000), ("moss-image-base", LF.EDIT): (10, 250, 500), ("moss-image-base", LF.T2I): (42, 378, 1134)}
SOUP_FORM = "svd_full"   # the form of a mean file (g56_lorafmt.SOUP_FORMS): svd_full | svd32 (rank 32, not exact: the fallback form) | concat
#
#
#
#
#
#
#
#
EXPORT_SCALE = {("flux2-klein-4b", LF.T2I): {1: 0.75, 3: 0.75, 4: 0.75}, ("flux2-klein-4b", LF.EDIT): 1.0, ("qwen-image-edit-2511", LF.EDIT): 1.0,
                ("hidream-o1", LF.T2I): {1: 0.9, 5: 1.05}, ("hidream-o1", LF.EDIT): {1: 0.8, 6: 1.0}, ("moss-image-base", LF.T2I): {1: 0.85, 3: 0.9, 4: 0.85},
                ("moss-image-base", LF.EDIT): 1.0}
SEED = 20261104
SNAPSHOT_EVERY_S = 600.0   # kill safety: a trained snapshot replaces the shipped file at least this often
FIRST_SNAPSHOT_STEPS = 1
#
#
#
#
MUSUBI_STEPS = 375
MUSUBI_SAVE_EVERY = 10
MUSUBI_LR = 4e-5
#
#
#
#
#
#
#
#
#
#
#
#
MUSUBI_TBAND = (850, 1000)
MUSUBI_NOTEXT = 0.5
#
#
#
#
#
#
#
#
#
#
#
MUSUBI_CPU_THREADS = 1
MUSUBI_GC = False
MUSUBI_TIMING = ""
#
#
#
#
#
#
#
MUSUBI_PATCHLINEAR = True
#
#
#
#
#
#
#
#
#
#
MUSUBI_TEMB = False
#
#
#
#
#
#
#
#
#
#
MUSUBI_MEMBERS = 12
TEMB_MODULE = "g56_lora_hidream_o1_temb"
MUSUBI_NET_DIR = HERE / "musubi_net"
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
TAILMEAN = {("hidream-o1", LF.EDIT): None, ("hidream-o1", LF.T2I): (0.5, 1.0), ("qwen-image-edit-2511", LF.EDIT): (0.8, 0.9, 1.0), ("moss-image-base", LF.T2I): (0.85, 0.92, 1.0)}
TAILMEAN_WINDOW = (0.5, 0.75, 1.0)
TAILMEAN_COST_X = 3.0
TAILMEAN_MARGIN_S = 15.0
#
#
#
#
#
#
#
#
#
#
EDIT_MASK = {("flux2-klein-4b", LF.EDIT): False, ("qwen-image-edit-2511", LF.EDIT): False, ("moss-image-base", LF.EDIT): False,
             ("hidream-o1", LF.EDIT): False}
MUSUBI_RUNNER = HERE / "g56_musubi_runner.py"
WALL_FRAC, RESERVE_S = 0.97, 120.0   # the ComfyUI families' _FD_COMMON budget: stop training 3 % + 120 s before the kill
POST_S = 90.0              # the runner's last snapshot must be on disk this long before the training deadline ends
WATCHDOG_S = 60.0          # hard stop this long before the validator's kill (the shipped file is always complete)
KILL_GRACE_S = 30.0        # the child is stopped when it runs this long past the training deadline (it stops itself first)
#
#
#
RES_RUNGS = (1024, 768, 512)
RELAUNCH_LOAD_S = 300.0
RELAUNCH_MARGIN_S = 120.0
#
#
#
#
MOSS_EDIT_PROCESSOR_CONST = True
QWEN35_PREPROCESSOR = {"size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16, "temporal_patch_size": 2,
                       "merge_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                       "processor_class": "Qwen3VLProcessor", "image_processor_type": "Qwen2VLImageProcessorFast"}


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
CFG_KNOBS = ("GRAD_ACCUM", "BATCH", "TS_BIAS", "OPTIMIZER", "WEIGHT_DECAY", "MAX_GRAD_NORM", "RANK", "FLIP", "TWO_ENTRY")
CFG_OPTIMIZERS = ("adamw8bit", "adamw", "adam")   # the first one = the job's own
#
#
#
#
#
#
#
#
#
#
#
TS_STRATA_K = 8
#
#
#
#
#
#
#
TSTRAT = {}
#
#
#
#
SDPA_BACKEND = {}
#
#
#
#
#
#
#
#
#
#
#
#
#
FREEZE_BLOCKS = {}
FREEZE_CELLS = {("qwen-image-edit-2511", LF.EDIT): 60}
#
#
#
#
#
#
#
#
#
#
#
LORA_NOOP = {("flux2-klein-4b", LF.T2I): True, ("moss-image-base", LF.T2I): True, ("hidream-o1", LF.T2I): True, ("qwen-image-edit-2511", LF.EDIT): True}
BNB_NOSYNC = {}
#
#
#
#
#
RANK = {("flux2-klein-4b", LF.T2I): 16, ("flux2-klein-4b", LF.EDIT): 16}
#
#
#
#
#
CAPTION_DROPOUT_CELL = {("hidream-o1", LF.T2I): 0.2}
#
#
#
#
#
MAX_GRAD_NORM_CELL = {("moss-image-base", LF.T2I): 3.0, ("qwen-image-edit-2511", LF.EDIT): 1.5}
#
#
#
#
#
TRAIN_LR_CELL = {}
#
#
#
#
#
#
#
#
LPT_RADIUS = {("hidream-o1", LF.T2I): 4.0}
#
#
#
#
#
#
TF32 = {("moss-image-base", LF.T2I): True}


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
def knob(name, default, cast=float):
    v = os.environ.get("G56NF_" + name)
    if v is None or v == "":
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes", "on")
    if cast is list:
        return [int(x) for x in v.split(",") if x.strip()]
    return cast(v)


def recipe(fam, mode):
    """The values build_config uses for one task (the constants above, each overridable by its G56NF_ knob)."""
    return dict(caption_dropout=knob("CAPTION_DROPOUT", CAPTION_DROPOUT_CELL.get((fam, mode), CAPTION_DROPOUT)),
                resolution=knob("RESOLUTION", list(RESOLUTION), list),
                lr=knob("LR", TRAIN_LR_CELL.get((fam, mode), TRAIN_LR.get(fam, 1e-4))),
                tmin=knob("TMIN", 0, int), tmax=knob("TMAX", TRAIN_TMAX.get(fam, 250), int),
                timestep_type=knob("TIMESTEP_TYPE", "linear", str),
                cache_te=knob("CACHE_TE", CACHE_TE.get((fam, mode), False), bool))


def config_knobs(fam, mode):
    """
    
    
    """
    ck = {}

    def read(name, parse, what):
        v = (os.environ.get("G56NF_" + name) or "").strip()
        if not v:
            return None
        try:
            got = parse(v)
        except Exception:  # noqa: BLE001
            got = None
        if got is None:
            log(f"WARN G56NF_{name} {v[:20]!r} ignored ({what})")
        return got

    def whole(lo, hi):
        return lambda v: int(v) if re.fullmatch(r"[0-9]{1,4}", v) and lo <= int(v) <= hi else None

    def number(good):
        def parse(v):
            x = float(v)
            return x if x == x and abs(x) != float("inf") and good(x) else None
        return parse

    def word(*names):
        return lambda v: v.lower() if v.lower() in names else None

    def flag(v):
        return {"1": True, "true": True, "on": True, "yes": True, "0": False, "false": False, "off": False, "no": False}.get(v.lower())

    g = read("GRAD_ACCUM", whole(1, 8), "a whole number 1..8")
    if g is not None and g > 1:
        ck["grad_accum"] = g
    b = read("BATCH", whole(1, 8), "a whole number 1..8")
    if b is not None and b > 1:
        ck["batch"] = b
    t = read("TS_BIAS", word("content", "style", "balanced"), "content or style")
    if t in ("content", "style"):
        ck["ts_bias"] = t
    o = read("OPTIMIZER", word(*CFG_OPTIMIZERS), f"{' | '.join(CFG_OPTIMIZERS)}: the member reset knows no other")
    if o is not None and o != CFG_OPTIMIZERS[0]:
        ck["optimizer"] = o
    w = read("WEIGHT_DECAY", number(lambda x: 0.0 <= x <= 1.0), "a number 0..1")
    if w is not None:
        ck["weight_decay"] = w
    c = read("MAX_GRAD_NORM", number(lambda x: x > 0.0), "a number > 0")
    if c is None:   # no G56NF_MAX_GRAD_NORM (or one that did not parse): the release table's clip of this (family, mode), where it holds one
        c = MAX_GRAD_NORM_CELL.get((fam, mode))
    if c is not None:
        ck["max_grad_norm"] = c
    r = read("RANK", whole(1, 128), "a whole number 1..128")
    if r is None:   # no G56NF_RANK (or one that did not parse): the release table's rank of this (family, mode), where it holds one
        r = RANK.get((fam, mode))
    if r is not None and r != LF.EVAL_RANK:
        ck["rank"] = r
    if read("FLIP", flag, "1 or 0"):
        if mode == LF.EDIT:
            log("WARN G56NF_FLIP ignored on an edit task (AI Toolkit mirrors the target, not the raw control images: wrong pairs)")
        else:
            ck["flip"] = True
    if read("TWO_ENTRY", flag, "1 or 0"):
        ck["two_entry"] = True
        if os.environ.get("G56NF_CAPTION_DROPOUT"):
            log("WARN G56NF_CAPTION_DROPOUT is not used with G56NF_TWO_ENTRY (the two entries train caption : empty 1 : 1)")
    return ck


def config_knobs_line(ck, n, cap):
    """One private line for the 'configuration only' knobs that are on: each value, and what they do to the counts - items per
    pass (N x 2 mirrored x 2 entries), items per optimizer step (GRAD_ACCUM x BATCH), the step cap (optimizer steps, unchanged:
    step_cap() counts the N training images) and how many passes over the items that is."""
    per_pass = max(1, n) * (2 if ck.get("flip") else 1) * (2 if ck.get("two_entry") else 1)
    per_step = ck.get("grad_accum", 1) * ck.get("batch", 1)
    return ("config knobs: " + ", ".join(f"{k} {v}" for k, v in sorted(ck.items())) + f"; items per pass {per_pass} (N {n}"
            + (" x 2 mirrored" if ck.get("flip") else "") + (" x 2 entries: caption / empty" if ck.get("two_entry") else "")
            + f"), items per optimizer step {per_step}; step cap {cap} optimizer steps (a run / a member) = "
            + f"{cap * per_step / per_pass:.1f} passes" + ("" if per_step == 1 else ", the cap is not divided by the items per step"))


def config_effect_line(m):
    """What the run really trained on with a 'configuration only' knob on (the runner's last status entry 'cfg') -> the private
    log: the counts over the TRAINED steps - optimizer steps, batches, items, mirrored items, items per caption-dropout entry -
    and what AI Toolkit's own objects held at the first step."""
    log(f"config effect: steps {m.get('steps', 0)}, batches {m.get('batches', 0)}, items {m.get('items', 0)}, mirrored "
        f"{m.get('mirrored', 0)}, by caption dropout {m.get('by_dropout')}; AI Toolkit held: optimizer {m.get('optimizer')}, weight "
        f"decay {m.get('weight_decay')}, rank {m.get('rank')}, max grad norm {m.get('max_grad_norm')}, timestep draw "
        f"{m.get('content_or_style')}, accumulation {m.get('gradient_accumulation')}, batch size {m.get('batch_size')}"
        + (f"; count error: {m['error']}" if m.get("error") else ""))


def export_scale(fam, mode, k_done=1):
    """
    
    
    """
    v = EXPORT_SCALE.get((fam, mode), 1.0)
    if isinstance(v, dict):
        ks = [k for k in v if k <= k_done]
        v = v[max(ks)] if ks else 1.0
    try:
        s = float(knob("EXPORT_SCALE", v))
    except Exception:  # noqa: BLE001
        s = float("nan")
    if not 0.0 < s < float("inf"):
        log(f"WARN export scale {s!r} for {fam} / {mode} ignored: 1.0")
        return 1.0
    return s


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
EXPORT_GROUP_SCALE = {("hidream-o1", LF.EDIT): {6: "lora_unet_model_visual_.*=0.8;lora_unet_model_language_model_layers_(1[89]|2[0-9]|3[0-5])_.*=1.25"}, ("hidream-o1", LF.T2I): {1: "diffusion_model\\.language_model\\.layers\\.(\\d+)\\.self_attn\\.(v|o)_proj=1.375;diffusion_model\\.language_model\\.layers\\.(\\d+)\\.mlp\\.down_proj=0.625;diffusion_model\\.language_model\\.layers\\.([0-8])\\.(self_attn\\.(q|k|v|o)|mlp\\.(gate|up|down))_proj=1.375;diffusion_model\\.language_model\\.layers\\.(9|1[0-7])\\.(self_attn\\.(q|k|v|o)|mlp\\.(gate|up|down))_proj=0.625;diffusion_model\\.language_model\\.layers\\.(2[7-9]|3[0-5])\\.(self_attn\\.(q|k|v|o)|mlp\\.(gate|up|down))_proj=0.625", 9: "diffusion_model\\.language_model\\.layers\\.(\\d+)\\.self_attn\\.(v|o)_proj=1.25;diffusion_model\\.language_model\\.layers\\.(\\d+)\\.mlp\\.down_proj=0.75;diffusion_model\\.language_model\\.layers\\.([0-8])\\.(self_attn\\.(q|k|v|o)|mlp\\.(gate|up|down))_proj=1.25;diffusion_model\\.language_model\\.layers\\.(9|1[0-7])\\.(self_attn\\.(q|k|v|o)|mlp\\.(gate|up|down))_proj=0.75;diffusion_model\\.language_model\\.layers\\.(2[7-9]|3[0-5])\\.(self_attn\\.(q|k|v|o)|mlp\\.(gate|up|down))_proj=0.75"}, ("moss-image-base", LF.T2I): "diffusion_model\\.layers\\.2[1-7]\\.proj\\.\\w+=1.15"}
EGS_UP = (".lora_B.weight", ".lora_up.weight", ".diff", ".diff_b")   # the tensors a group factor multiplies (DENSE: the delta itself)
EGS_REST = (".lora_A.weight", ".lora_down.weight", ".alpha")         # the rest of a module: never touched
_EGS_SAID = set()   # the WARN reasons said so far (a K-member task ships many files: each reason once)


def egs_warn(reason, name):
    """one WARN line per reason and task: the file `name` (and every later one with the same reason) ships without the group scale"""
    if reason not in _EGS_SAID:
        _EGS_SAID.add(reason)
        log(f"WARN export group scale off for {name}: {reason} - shipped without it (said once per reason)")


def export_group_plan(fam, mode, k_done=1):
    """EXPORT_GROUP_SCALE of a shipped file that holds k_done finished members: [(regex text, compiled regex, factor)] in the order of the
    spec, or None (off): G56NF_EXPORT_GROUP_SCALE when set ('off' / '0' / 'none' / 'false' / 'no' = off), else the (family, mode) entry of
    the table. A spec that does not parse - a part without '=', an empty or repeated regex, one that does not compile, a factor that is
    not a finite number >= 0 - -> one WARN line, None. '{OR}' in a regex = '|' (the queue's escape of its field separator)."""
    env = (os.environ.get("G56NF_EXPORT_GROUP_SCALE") or "").strip()
    if env.lower() in ("0", "off", "none", "false", "no"):
        return None
    v = env or EXPORT_GROUP_SCALE.get((fam, mode))
    if isinstance(v, dict) and v and all(isinstance(k, int) for k in v):   # {member count: spec} - read at the largest count <= k_done
        ks = [k for k in v if k <= k_done]
        v = v[max(ks)] if ks else None
    if not v:
        return None
    try:
        parts = v.items() if isinstance(v, dict) else [p.rpartition("=")[::2] for p in str(v).split(";") if p.strip()]
        plan, seen = [], set()
        for rx, f in parts:
            rx, f = str(rx).replace("{OR}", "|").strip(), float(f)
            if not rx or rx in seen or not 0.0 <= f < float("inf"):
                raise ValueError(f"part {rx[:60]!r} = {f!r}")
            seen.add(rx)
            plan.append((rx, re.compile(rx), f))
        return plan or None
    except Exception as e:  # noqa: BLE001  (a broken spec never stops a task: the files ship without it)
        egs_warn(f"the spec {str(v)[:120]!r} does not parse ({type(e).__name__}: {str(e)[:80]}; 'regex=factor;...', a factor >= 0)",
                 f"{fam} / {mode}")
        return None


def egs_split(key):
    """a key of a shipped file -> (its module, True when the key is a tensor a group factor multiplies); (None, False) for any other key"""
    for suf in EGS_UP:
        if key.endswith(suf):
            return key[: -len(suf)], True
    for suf in EGS_REST:
        if key.endswith(suf):
            return key[: -len(suf)], False
    return None, False


def group_scaled(fam, mode, header, sd, k_done, name, rep=None):
    """EXPORT_GROUP_SCALE on the shipped file `sd` of k_done finished members, AFTER its export scale -> (the file, its loader report).
    Off (export_group_plan None) -> (sd, rep), the very objects it got: the file stays byte for byte. On -> a new state dict whose up
    factors (EGS_UP) of every module a regex of the plan matches WHOLE are float64(tensor) x the product of the factors of the regexes
    that match it, stored in the tensor's own dtype; every other tensor is sd's own object (keys and their order kept), and its
    g56_lorafmt.check report. A regex that matches no module of the file, a result that fails the loader mirror or is not finite ->
    egs_warn, (sd, rep): the file ships without the group scale. One private line when it acts."""
    plan = export_group_plan(fam, mode, k_done)
    if not plan:
        return sd, rep
    try:
        mods = {}   # the file's modules, in the order of its keys
        for k in sd:
            m = egs_split(k)[0]
            if m is not None:
                mods[m] = True
        fac, hits = {}, []
        for text, rx, f in plan:
            hit = [m for m in mods if rx.fullmatch(m)]
            hits.append(len(hit))
            for m in hit:
                fac[m] = fac.get(m, 1.0) * f
        if not all(hits):
            raise LookupError("no module of the file matches " + ", ".join(repr(t[:80]) for (t, _, _), n in zip(plan, hits) if not n))
        out, n_up = {}, 0
        for k, v in sd.items():
            m, up = egs_split(k)
            if up and m in fac:
                out[k] = (v.double() * fac[m]).to(v.dtype)
                n_up += 1
            else:
                out[k] = v
        chk = LF.check(fam, mode, header, out, expect_full=True)
        if not chk["ok"] or not finite({k: out[k] for k in out if out[k] is not sd[k]}):
            raise ValueError("the scaled file fails the loader check or is not finite")
    except Exception as e:  # noqa: BLE001  (the group scale never costs a shipped file)
        egs_warn(f"{type(e).__name__}: {str(e)[:200]}", name)
        return sd, rep
    log(f"export group scale ({name}, {k_done} member(s)): " + "; ".join(f"{t[:80]} x{f:g} {n} modules" for (t, _, f), n in zip(plan, hits))
        + f" -> {len(fac)} of {len(mods)} modules, {n_up} up factors, down factors / alphas untouched")
    return out, chk


def lpt_radius(fam, mode, hours=None):
    """The Gaussian radius in pixels with which prepare_t2i blurs the training images of a task (0 = the raw bytes): the (family, mode)
    entry of LPT_RADIUS - a number, or {hours: number} read at the largest hours <= the task's `hours` (none there, or no hours given -> 0)
    - or G56NF_LPT_RADIUS, which decides over the table. Anything but a finite number >= 0 -> 0 with one WARN line: a broken value never
    changes the data and never stops a task."""
    v = LPT_RADIUS.get((fam, mode), 0.0)
    try:
        if isinstance(v, dict):
            ks = [h for h in v if hours is not None and float(h) <= float(hours) + 1e-6]
            v = v[max(ks, key=float)] if ks else 0.0
        r = float(knob("LPT_RADIUS", v))
    except Exception:  # noqa: BLE001
        r = float("nan")
    if not 0.0 <= r < float("inf"):
        log(f"WARN lpt radius {(os.environ.get('G56NF_LPT_RADIUS') or repr(v))[:20]!r} for {fam} / {mode} ignored: the raw bytes")
        return 0.0
    return r


#
#
#
#
#
#
#
#
#
#
HIDREAM_PATCH_PX = 32
HIDREAM_TEMPLATE_PAD = 16


def caption_bytes(path, nfc=False):
    """The UTF-8 bytes of a caption file as a byte-level BPE can see them at most: the file's own bytes and, with nfc, the bytes of
    its NFC form when that is longer (a tokenizer that normalizes first counts those)."""
    raw = path.read_bytes()
    if not nfc:
        return len(raw)
    import unicodedata
    return max(len(raw), len(unicodedata.normalize("NFC", raw.decode("utf-8", errors="replace")).encode("utf-8")))


def seq_bound(fam, mode, data_dir, resolution):
    """
    
    
    
    
    
    
    """
    if (fam, mode) == ("moss-image-base", LF.T2I):
        px, pad, cap, nfc = MOSS_PATCH_PX, MOSS_TEMPLATE_PAD, MOSS_MAX_TEXT_TOKENS, False
    elif (fam, mode) == ("hidream-o1", LF.T2I):
        px, pad, cap, nfc = HIDREAM_PATCH_PX, HIDREAM_TEMPLATE_PAD, None, True
    else:
        return None
    try:
        image = (max(resolution) // px) ** 2
        longest = max((caption_bytes(f, nfc) for f in Path(data_dir).glob("*.txt")), default=0)
        text = longest + pad if cap is None else min(cap, longest + pad)
        return image + text, image, text
    except Exception as e:  # noqa: BLE001  (no bound = checkpointing stays on where a fence is set)
        log(f"WARN sequence bound: {type(e).__name__}: {str(e)[:120]}")
        return None


def gc_for_task(fam, mode, bound=None):
    """Gradient checkpointing of the task's AI Toolkit job. G56NF_GC (1 / 0) decides when set. Else GC_DEFAULT - except where a
    fence exists for the (family, mode) (GC_OFF_MAX_TOKENS; G56NF_GC_FENCE where a sequence bound exists): there it is OFF exactly
    when the task's sequence bound (seq_bound) is known and <= the fence, ON otherwise."""
    default = knob("GC", GC_DEFAULT.get((fam, mode), True), bool)
    if os.environ.get("G56NF_GC"):
        return default
    fence = GC_OFF_MAX_TOKENS.get((fam, mode))
    if bound is not None and os.environ.get("G56NF_GC_FENCE"):
        try:
            fence = int(float(os.environ["G56NF_GC_FENCE"]))
        except Exception as e:  # noqa: BLE001
            log(f"WARN G56NF_GC_FENCE ignored ({type(e).__name__})")
    if not fence:
        return default
    off = bound is not None and bound[0] <= fence
    log(f"gc fence {fence} tokens: sequence bound {'unknown' if bound is None else f'{bound[0]} (image {bound[1]} + text {bound[2]})'} "
        f"-> gradient checkpointing {'off' if off else 'on'}")
    return not off


def gc_step_plan(fam, mode):
    """
    
    """
    v = GC_STEP_FENCE.get((fam, mode), 0)
    raw = os.environ.get("G56NF_GC_STEP_FENCE")
    if raw:
        try:
            v = int(float(raw))
        except Exception as e:  # noqa: BLE001
            log(f"WARN G56NF_GC_STEP_FENCE ignored ({type(e).__name__})")
    if not v or v <= 0:
        return {}
    if (fam, mode) not in GC_STEP_COUNTER:
        log(f"WARN gc step fence {v}: no token counter for {fam} / {mode} - off")
        return {}
    if os.environ.get("G56NF_GC"):
        log(f"gc step fence {v}: off - G56NF_GC decides the whole run")
        return {}
    log(f"gc step fence {v} tokens: gradient checkpointing per forward - off at <= {v} real tokens (batch x (image + text), counted at "
        f"the transformer), on above; held on from an OOM to the end")
    return {"gc_step_fence": v}


def edit_mask_plan(fam, mode, data_dir, ctrl_dirs):
    """
    
    
    """
    try:
        if mode != LF.EDIT or not ctrl_dirs or not knob("EDIT_MASK", EDIT_MASK.get((fam, mode), False), bool):
            return {}
        import g56_editmask as EM
        mdir = Path(data_dir).parent / "masks"
        got = EM.write_masks(data_dir, ctrl_dirs[0], mdir)
        masked = [v for v in got.values() if v is not None]
        if not masked:
            log(f"edit mask: masked 0/{len(got)}: nothing to weight")
            return {}
        cov, mass = sorted(v[0] for v in masked), sorted(v[1] for v in masked)
        log(f"edit mask: masked {len(masked)}/{len(got)}, coverage median {cov[len(cov) // 2]:.3f} (min {cov[0]:.3f} / "
            f"max {cov[-1]:.3f}), changed mass median {mass[len(mass) // 2]:.3f}")
        return dict(edit_mask_dir=str(mdir))
    except Exception as e:  # noqa: BLE001
        log(f"WARN edit mask not built ({type(e).__name__}: {str(e)[:120]}): the unweighted loss")
        return {}


def edit_mask_line(m):
    """What a runner weighted in a G56NF_EDIT_MASK run (its last status: AI Toolkit runner / Musubi runner) -> the private log."""
    mass = m.get("mass_median")
    log(f"edit mask: weighted steps {m.get('weighted', 0)}/{m.get('steps', 0)} ({m.get('attached', 0)} with a mask), items masked "
        f"{m.get('masked', 0)}/{m.get('items', 0)}, changed mass median {'-' if mass is None else '%.3f' % mass} at the item crop"
        + (f"; OFF: {m['off']}" if m.get("off") else ""))


def tf32_plan(fam, mode):
    """TF32 / G56NF_TF32=1 / 0 -> {} (off: nothing touched) | {'tf32': True} for the runner (its main() switches torch's two TF32
    switches on before any model exists). The knob decides over the table; unset / '' = the table's (family, mode) entry. No line
    here: the private line 'tf32: ...' after the child says what torch held."""
    try:
        on = knob("TF32", bool(TF32.get((fam, mode), False)), bool)
    except Exception:  # noqa: BLE001
        on = False
    return dict(tf32=True) if on else {}


def ts_strata_plan():
    """
    
    """
    v = (os.environ.get("G56NF_TS_STRATA") or "").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return {}
    if v in ("1", "on", "true", "yes"):
        return dict(ts_strata=TS_STRATA_K)
    if re.fullmatch(r"[0-9]{1,2}", v) and 2 <= int(v) <= 64:
        return dict(ts_strata=int(v))
    log(f"WARN G56NF_TS_STRATA {v[:20]!r} ignored (1, or a whole number 2..64)")
    return {}


def sdpa_plan(fam, mode):
    """
    """
    raw = os.environ.get("G56NF_SDPA_BACKEND")
    v = (raw if raw is not None else SDPA_BACKEND.get((fam, mode), "")) or ""
    v = str(v).strip().lower()
    if v in ("", "0", "off", "default"):
        return {}
    if v != "cudnn":
        log(f"WARN G56NF_SDPA_BACKEND {v[:20]!r} ignored (cudnn, or unset)")
        return {}
    log("sdpa backend: cudnn first in torch's attention priority order (then torch's own order: the fallback); every attention call "
        "counted by the backend torch picks")
    return dict(sdpa_backend="cudnn")


def sdpa_line(m):
    """What the runner's attention did in a G56NF_SDPA_BACKEND run (its last status 'sdpa') -> one private line."""
    if not m:
        log("sdpa backend: no reading (the runner wrote none)")
    elif not m.get("applied"):
        log(f"sdpa backend: not applied ({m.get('why')}) - torch's own order")
    else:
        log(f"sdpa backend {m.get('want')}: order {m.get('order')}; attention calls by backend {json.dumps(m.get('calls'), sort_keys=True)}; "
            f"first call {m.get('first')}")


def freeze_plan(fam, mode, hours=None):
    """
    
    
    
    """
    raw = os.environ.get("G56NF_FREEZE_BLOCKS")
    if raw is not None and raw.strip() != "":   # the knob decides, at any hours
        v = raw
    else:
        v = FREEZE_BLOCKS.get((fam, mode), "")
        if isinstance(v, dict):   # {hours: k}: the entry at the largest hours <= the task's (LPT_RADIUS's rule)
            try:
                at = {float(h): k for h, k in v.items()}
                ks = [h for h in at if hours is not None and h <= float(hours) + 1e-6]
                v = at[max(ks)] if ks else ""
            except Exception:  # noqa: BLE001  (hours that are not numbers: the WARN below)
                v = repr(v)
    v = str(v or "").strip().lower()
    if v in ("", "0", "off"):
        return {}
    nb = FREEZE_CELLS.get((fam, mode))
    if nb is None:
        log(f"WARN G56NF_FREEZE_BLOCKS {v[:20]!r} ignored (defined for {', '.join(f'{f} {m}' for f, m in FREEZE_CELLS)} only)")
        return {}
    if not re.fullmatch(r"[0-9]{1,3}", v) or not 1 <= int(v) <= nb - 1:
        log(f"WARN G56NF_FREEZE_BLOCKS {v[:20]!r} ignored (a whole number 1..{nb - 1})")
        return {}
    k = int(v)
    log(f"freeze blocks: on - the LoRA of transformer blocks 0..{k - 1} of {nb} is not trained (its factors switched off after AI "
        f"Toolkit's own switch; their zero delta shipped): the backward stops at block {k}")
    return dict(freeze_blocks=k)


def freeze_line(m, step_s=None):
    """What the runner did in a G56NF_FREEZE_BLOCKS run (its last status 'freeze', and the median step time) -> one private line: the
    split, the first step's blocks (outputs with a gradient; forward calls per block - checkpointing runs a trained block twice, a
    frozen one once), the frozen factors switched off again, the largest lora_up of the frozen (must be 0) and of the trained modules."""
    if not isinstance(m, dict) or "applied" not in m:
        log("freeze blocks: no reading (the runner wrote none)")
        return
    if not m.get("applied"):
        log(f"freeze blocks: not applied ({m.get('why')}) - every block trained")
        return
    k, nb = int(m.get("k") or 0), int(m.get("blocks") or 0)
    og, calls = m.get("out_grad"), m.get("calls")
    if og is None or not calls:
        blk = "first step: no block record"
    else:
        lo, hi = calls[:k], calls[k:]
        blk = (f"first step: block outputs with a gradient {len(og)} of {nb} ({'blocks ' + str(min(og)) + '..' + str(max(og)) if og else 'none'}"
               f"; below {k}: {sum(1 for i in og if i < k)}), forward calls per block below {k} {sorted(set(lo))}, from {k} {sorted(set(hi))}")
    fu, ku = m.get("frozen_up_max"), m.get("kept_up_max")
    log(f"freeze blocks 0..{k - 1} of {nb}: applied; LoRA modules frozen {m.get('frozen_modules')} of "
        f"{int(m.get('frozen_modules') or 0) + int(m.get('kept_modules') or 0)} ({m.get('frozen_params')} factors), re-frozen "
        f"{m.get('refrozen')}; {blk}; lora_up max frozen {fu if fu is None else '%.3g' % fu} / trained {ku if ku is None else '%.3g' % ku}"
        + (f"; step {step_s:.3f} s (median)" if isinstance(step_s, (int, float)) and step_s > 0 else "")
        + (f"; {m['up_error']}" if m.get("up_error") else ""))


def tstrat_plan(fam, mode):
    """."""
    try:
        on = knob("TSTRAT", TSTRAT.get((fam, mode), False), bool)
    except Exception:  # noqa: BLE001
        on = False
    if not on:
        return {}
    log("tstrat: on - the timestep of every step from the member's low-discrepancy sequence (frac(u0 + n / golden ratio), u0 from its "
        "seed) through AI Toolkit's own uniform draw over the band; its randint made and discarded (the noise streams stay paired)")
    return dict(tstrat=True)


def tstrat_line(m):
    """What the runner drew in a G56NF_TSTRAT run (its last status 'tstrat') -> one private line."""
    if not m:
        log("tstrat: no reading (the runner wrote none)")
        return
    log(f"tstrat: on, n steps {m.get('steps', 0)}: drawn {m.get('drawn', 0)} ({m.get('applied', 0)} taken by AI Toolkit), own draws "
        f"{m.get('own', 0)}, band {m.get('lo')}..{m.get('hi')}, distance to the uniform band {m.get('disc')}"
        + (f"; OFF: {m['off']}" if m.get("off") else ""))


def profile_plan():
    """
    
    
    
    """
    v = (os.environ.get("G56NF_PROFILE") or "").strip()
    if not v:
        return {}
    m = re.fullmatch(r"(\d{1,5}),(\d{1,5})", v.replace(" ", ""))
    if m and 1 <= int(m.group(1)) <= int(m.group(2)) <= int(m.group(1)) + 49:
        return dict(profile=[int(m.group(1)), int(m.group(2))])
    log(f"WARN G56NF_PROFILE {v[:30]!r} ignored (a,b: the training steps a..b, 1 <= a <= b <= a + 49)")
    return {}


def profile_line(p):
    """What the runner's profiler read (its last status) -> one private line."""
    if not p:
        log("profile: no reading (the run ended before the profiled steps did)")
        return
    if p.get("error"):
        log(f"profile steps {p.get('steps')}: failed ({p['error']})")
        return
    st = ", ".join(f"{k} {v[0] / max(1, p['steps'][1] - p['steps'][0] + 1):.3f} s" for k, v in list(p.get("stages", {}).items())[:8])
    gr = ", ".join(f"{k} {v / max(1, p['steps'][1] - p['steps'][0] + 1):.0f} ms" for k, v in p.get("groups", {}).items())
    log(f"profile steps {p['steps'][0]}..{p['steps'][1]} ({p.get('device')}): wall {p.get('wall_s_per_step')} s/step with the profiler on, "
        f"kernels {p.get('kernel_ms_per_step')} ms/step; per step - stages (synchronized, inclusive): {st}; groups: {gr}; top: "
        f"{'; '.join(f'{t[0][:60]} {t[2]:.0f} ms' for t in p.get('top', []))}; file {p.get('file')}")


def speed_lines(spec, status):
    """."""
    if spec.get("lora_noop"):
        m = status.get("lora_noop")
        if not m:
            log("lora noop: no reading (the runner wrote none)")
        else:
            orig = m.get("original") or {}
            log(f"lora noop: fast path {m.get('fast', 0)} calls over {m.get('modules', 0)} modules, AI Toolkit's own forward "
                f"{sum(orig.values())} calls {json.dumps(orig, sort_keys=True)}, LoRA term skipped by AI Toolkit {m.get('at_skip', 0)}; "
                f"multiplier tensor read {m.get('multiplier_reads', 0)} time(s); at step {status.get('step')}")
    if spec.get("bnb_nosync"):
        b = status.get("bnb_nosync")
        if not b or b.get("applied") is None:
            log("bnb nosync: no reading (the run ended before its first training step)")
        elif not b.get("applied"):
            log(f"bnb nosync: not applied ({b.get('why')}) - bitsandbytes' own synchronizations")
        else:
            n, s = int(b.get("tensors") or 0), int(b.get("steps") or 0)
            log(f"bnb nosync: applied ({b.get('why')}): synchronizations skipped {b.get('skipped', 0)}, kept {b.get('passed', 0)}; "
                f"{n} tensors x {s} trained steps = {n * s}")


def gc_step_line(m):
    """What the runner decided in a G56NF_GC_STEP_FENCE run (its last status 'gc_step') -> one private line."""
    if not m:
        log("gc step fence: no reading (the runner wrote none)")
    elif not m.get("applied"):
        log(f"gc step fence: not applied ({m.get('why')}) - the job's own gradient checkpointing for the whole run")
    else:
        log(f"gc step fence {m.get('fence')}: forwards with gradients - checkpointing off {m.get('off', 0)}, on above the fence "
            f"{m.get('on', 0)}, on not counted {m.get('unknown', 0)}, held on {m.get('held', 0)}; tokens: largest off {m.get('n_off_max')}, "
            f"smallest above {m.get('n_on_min')}, largest {m.get('n_max')}" + (f"; held on since: {m['hold']}" if m.get("hold") else ""))


def ts_strata_line(m):
    """What the runner drew in a G56NF_TS_STRATA run (its last status) -> the private log: the stratified steps, how many of them
    AI Toolkit really trained at our index, the items, the draws per stratum."""
    log(f"ts strata: K {m.get('k')}, band {m.get('lo')}..{m.get('hi')}, stratified steps {m.get('drawn', 0)}/{m.get('steps', 0)} "
        f"({m.get('applied', 0)} taken by AI Toolkit), items {m.get('items', 0)}, per stratum {m.get('counts')}"
        + (f"; OFF: {m['off']}" if m.get("off") else ""))


def members_max(fam, mode):
    """K_max of the task: MEMBERS_MAX or G56NF_MEMBERS; at least 1; a value that does not parse is 1."""
    try:
        return max(1, knob("MEMBERS", MEMBERS_MAX.get((fam, mode), 1), int))
    except Exception as e:  # noqa: BLE001
        log(f"WARN G56NF_MEMBERS ignored ({type(e).__name__}): 1 member")
        return 1


#
#
#
#
#
#
#
FILL_RHO = {}


def fill_rho(fam, mode):
    """FILL_RHO or G56NF_FILL_RHO for the task's (family, mode) -> {K_fit: rho_min} ({} = off). The knob: 'K:rho,...' with K a whole
    number >= 1 and 0 < rho <= 1; anything else -> one WARN line, off."""
    v = (os.environ.get("G56NF_FILL_RHO") or "").strip()
    if not v:
        return dict(FILL_RHO.get((fam, mode)) or {})
    if v.lower() in ("0", "off", "none"):
        return {}
    try:
        out = {}
        for part in v.split(","):
            k, r = part.split(":")
            k, r = int(k.strip()), float(r.strip())
            if k < 1 or not 0.0 < r <= 1.0 or k in out:
                raise ValueError(part)
            out[k] = r
        if out:
            return out
    except ValueError:
        pass
    log(f"WARN G56NF_FILL_RHO {v[:40]!r} ignored (K:rho pairs, K >= 1, 0 < rho <= 1, e.g. 2:0.5,3:0.85): off")
    return {}


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
PLAN5 = {("flux2-klein-4b", LF.T2I): "end:90,khi:24,loS:0.7,hiS:1,loL:0.85,hiL:1.25", ("moss-image-base", LF.T2I): "end:90,khi:16,lo:1,hi:1.3,tie:stretch,cS:28.5", ("hidream-o1", LF.T2I): "end:240,khi:12,lo:0.75"}
PLAN5_END_S = 90.0
PLAN5_MAX_GB = 8.4
PLAN5_PROBE = 30
PLAN5_CUT = 0.9
PLAN5_POST_A = 15.0    # the prior's fixed part: the last member's save, the driver's poll, the load + check of its saves, a margin
PLAN5_POST_X = 1.25
PLAN5_POLL_S = 2.0     # the driver's poll of a PLAN5 child (5 s otherwise)
#
#
PLAN5_POST_B = {"flux2-klein-4b": 2.4, "moss-image-base": 2.6, "hidream-o1": 3.2, "qwen-image-edit-2511": 10.1}
PLAN5_POST_B_MUSUBI = 2.1
PLAN5_SVD_X, PLAN5_SVD_PAD_S = 1.5, 5.0   # the final soup_form() merge is started when PLAN5_SVD_X x the prior + the pad fits before the end
CAPTYPE_RE = re.compile(r"(\"[^\"]{2,}\"|'[^']{3,}'|\b(text|reads|reading|written|lettering|letters|logo|sign|signage|caption|title|"
                        r"typography|words?|label|font)\b)", re.I)
#
#
PLAN5_ENTRY_TIE = {("moss-image-base", LF.T2I)}
#
#
#
#
#
CAPTYPE_C = {}


def plan5_cell(fam, mode):
    """PLAN5 or G56NF_PLAN5 for the task's (family, mode) -> None (off: unset, '0', 'off') | dict(end, khi, lo, hi, tie, probe, cut,
    maxgb, types = {'S': (lo, hi), 'L': (lo, hi)} | None). A value that does not parse, an unknown key or a number out of range -> one
    WARN line and off: a broken entry never changes a deadline. end below WATCHDOG_S + 15 s is raised to it (WARN)."""
    raw = os.environ.get("G56NF_PLAN5")
    v = str(raw if raw is not None and raw.strip() != "" else PLAN5.get((fam, mode)) or "").strip()
    if v.lower() in ("", "0", "off", "none", "false", "no"):
        return None
    try:
        out = dict(end=PLAN5_END_S, khi=None, lo=1.0, hi=1.0, tie="fill", probe=PLAN5_PROBE, cut=PLAN5_CUT, maxgb=PLAN5_MAX_GB, types=None)
        typed = {}

        def fracs(x, good):
            val = [float(y) for y in x.split("/")]
            if not val or not all(good(y) for y in val):
                raise ValueError(x)
            return val[0] if len(val) == 1 else val
        for part in v.split(","):
            k, x = (y.strip() for y in part.split(":"))
            if k in ("lo", "loS", "loL"):
                (out if k == "lo" else typed)[k] = fracs(x, lambda y: 0.0 < y <= 1.0)
            elif k in ("hi", "hiS", "hiL"):
                (out if k == "hi" else typed)[k] = fracs(x.split("/")[0] if "/" not in x else "x", lambda y: 1.0 <= y <= 8.0)
            elif k in ("cS", "cL") and 0.0 < float(x) <= 1000.0:
                typed[k] = float(x)
            elif k in ("cSmaxN", "cLmaxN", "cSminN", "cLminN") and 1 <= int(x) <= 100000:
                typed[k] = int(x)
            elif k == "end" and 0.0 <= float(x) < 86400.0:
                out[k] = float(x)
            elif k == "khi" and 1 <= int(x) <= 64:
                out[k] = int(x)
            elif k == "tie" and x in ("fill", "stretch"):
                out[k] = x
            elif k == "probe" and 0 <= int(x) <= 1000:
                out[k] = int(x)
            elif k == "cut" and 0.0 <= float(x) <= 1.0:
                out[k] = float(x)
            elif k == "maxgb" and 0.0 < float(x) < 1000.0:
                out[k] = float(x)
            else:
                raise ValueError(part)
        if typed:
            out["types"] = {t: (typed.get("lo" + t, out["lo"]), typed.get("hi" + t, out["hi"])) for t in ("S", "L")}
            out["c"] = {t: typed["c" + t] for t in ("S", "L") if "c" + t in typed}
            out["cmax"] = {t: typed["c" + t + "maxN"] for t in ("S", "L") if "c" + t + "maxN" in typed}
            if any("c" + t + "minN" in typed for t in ("S", "L")):   # (no 'cmin' without one: an entry of before reads as it did)
                out["cmin"] = {t: typed["c" + t + "minN"] for t in ("S", "L") if "c" + t + "minN" in typed}
    except Exception:  # noqa: BLE001
        log(f"WARN G56NF_PLAN5 {v[:80]!r} ignored (key:value pairs of end, khi, lo, hi, loS, hiS, loL, hiL, cS, cL, cSmaxN, cLmaxN, tie, probe, cut, maxgb): off")
        return None
    if out["end"] < WATCHDOG_S + 15.0:
        log(f"WARN plan5: end {out['end']:g} s is below the watchdog's {WATCHDOG_S:g} s + 15 s: {WATCHDOG_S + 15.0:g} s")
        out["end"] = WATCHDOG_S + 15.0
    return out


def caption_type(data_dir):
    """
    
    
    
    """
    try:
        k = n = 0
        for f in sorted(Path(data_dir).glob("*.txt")):
            t = f.read_text(encoding="utf-8", errors="replace")
            if t.strip():
                n += 1
                k += bool(CAPTYPE_RE.search(t))
        if n < 3:
            return "L", k, n, "fewer than 3 non-empty captions"
        return ("S" if k / n < 0.5 else "L"), k, n, ""
    except Exception as e:  # noqa: BLE001
        return "L", 0, 0, f"{type(e).__name__}: {str(e)[:80]}"


def plan5_note(cost, form_s=None, merge_s=None):
    """
    
    """
    if cost is None:
        return
    try:
        for key, v in (("form_s", form_s), ("merge_s", merge_s)):   # the larger of the last two readings (one slow merge does not stay)
            if v is not None:
                cost[key + "_last"] = (cost.get(key + "_last") or [])[-1:] + [float(v)]
                cost[key] = max(cost[key + "_last"])
        tmp = Path(str(cost["path"]) + ".tmp")
        tmp.write_text(json.dumps(dict(form_s=cost["form_s"], merge_s=cost["merge_s"])))
        os.replace(tmp, cost["path"])
    except Exception:  # noqa: BLE001
        pass


def plan5_times(p5, fam, t_kill, work, holder, musubi=False):
    """PLAN5 on: the plan's end (the kill minus end), the driver's poll, the child's own last moment (the end: a child that runs on is
    stopped there, not a grace later), the prior per save of rank 32 and the cost file -> holder; -> the training end the driver's
    other rules read (t_train_end: its - POST_S is the runner's first deadline, the end minus the prior's fixed part)."""
    p5["t_end"] = t_kill - p5["end"]
    p5["post_b32"] = PLAN5_POST_B_MUSUBI if musubi else PLAN5_POST_B.get(fam, 3.0)
    holder.update(p5=p5, poll_s=PLAN5_POLL_S, t_child_kill=p5["t_end"],
                  p5_cost=dict(path=Path(work) / "plan5_cost.json", form_s=0.0, merge_s=0.0, t_stop=t_kill - WATCHDOG_S))
    return p5["t_end"] - PLAN5_POST_A + POST_S


def plan5_cap(p5, fam, mode, n, cap):
    """PLAN5's cS / cL: the step cap of the task's caption type (p5['kind'], set by plan5_kind) -> (the step cap, why): c x N inside
    STEPS_PER_IMAGE's own clamps when the entry names a c for that type, the task is inside that c's evidence - not larger than its
    cSmaxN / cLmaxN, not smaller than its cSminN / cLminN (a type S task outside it takes cL when there is one and N is inside cL's own
    bounds) - and no G56NF_STEPS / G56NF_STEPS_PER_IMAGE decides; else `cap` as it is."""
    kind, cs, cmax = p5.get("kind") or "", p5.get("c") or {}, p5.get("cmax") or {}
    cmin = p5.get("cmin") or {}   # cSminN / cLminN (none named: no lower end)
    rule = STEPS_PER_IMAGE.get((fam, mode))
    if os.environ.get("G56NF_STEPS") or os.environ.get("G56NF_STEPS_PER_IMAGE"):
        return cap, f"the knob's {cap} steps (G56NF_STEPS / G56NF_STEPS_PER_IMAGE decide)"
    if not rule or not n or not cs:
        return cap, f"the table's {cap} steps"

    def inside(t):   # N inside the evidence of type t's c: cTminN <= N <= cTmaxN (a bound the entry does not name: none)
        return t in cs and not (t in cmax and n > cmax[t]) and not (t in cmin and n < cmin[t])
    use, note = kind, ""
    if kind in cs and kind in cmin and n < cmin[kind]:   # below the evidence of that type's c
        use, note = ("L" if kind == "S" and inside("L") else ""), f"N {n} < {cmin[kind]}: not c{kind}; "
    if kind in cs and kind in cmax and n > cmax[kind]:   # outside the evidence of that type's c
        use, note = ("L" if kind == "S" and inside("L") else ""), f"N {n} > {cmax[kind]}: not c{kind}; "
    if use not in cs:
        return cap, f"{note}the table's {cap} steps"
    steps = int(min(rule[2], max(rule[1], round(cs[use] * n))))
    return steps, f"{note}c{use} {cs[use]:g} x N {n} -> {steps} steps (clamp {rule[1]}..{rule[2]}; the table's {cap})"


def plan5_kind(p5, mode, data_dir):
    """PLAN5 with a window / a c by caption type: the task's type, read once (caption_type; text-to-image only) -> p5['kind'] ('' =
    the entry names no type, or an edit task: the plain lo / hi) and p5['kind_text'], the start of the private line 'caption type'."""
    p5["kind"], p5["kind_text"] = "", ""
    if p5["types"] and mode == LF.T2I:
        kind, k, n, why = caption_type(data_dir)
        p5["kind"] = kind
        p5["kind_text"] = f"caption type: {k}/{n} captions with text cues (f={k / n if n else 0.0:.3f}) -> {kind}" + (f" ({why})" if why else "")
    return p5["kind"]


def captype_cell(fam, mode):
    """CAPTYPE_C or G56NF_CAPTYPE_C for a task WITHOUT PLAN5 -> None (off: unset, '0', 'off') | dict(c, cmax[, cmin]) - what plan5_cap
    reads (the caller sets 'kind'). A value that does not parse, an unknown key, a number out of its range or no cS / cL -> one WARN
    line and off: a broken entry never changes a length."""
    raw = os.environ.get("G56NF_CAPTYPE_C")
    v = str(raw if raw is not None and raw.strip() != "" else CAPTYPE_C.get((fam, mode)) or "").strip()
    if v.lower() in ("", "0", "off", "none", "false", "no"):
        return None
    try:
        typed = {}
        for part in v.split(","):
            k, x = (y.strip() for y in part.split(":"))
            if k in ("cS", "cL") and 0.0 < float(x) <= 1000.0:
                typed[k] = float(x)
            elif k in ("cSminN", "cLminN", "cSmaxN", "cLmaxN") and 1 <= int(x) <= 100000:   # (a type's bounds on N: whole numbers)
                typed[k] = int(x)
            else:
                raise ValueError(part)
        if "cS" not in typed and "cL" not in typed:   # (a bound alone is no entry)
            raise ValueError("no cS / cL")
        out = dict(c={t: typed["c" + t] for t in ("S", "L") if "c" + t in typed},
                   cmax={t: typed["c" + t + "maxN"] for t in ("S", "L") if "c" + t + "maxN" in typed})
        if any("c" + t + "minN" in typed for t in ("S", "L")):
            out["cmin"] = {t: typed["c" + t + "minN"] for t in ("S", "L") if "c" + t + "minN" in typed}
        return out
    except Exception:  # noqa: BLE001
        log(f"WARN G56NF_CAPTYPE_C {v[:60]!r} ignored (key:value pairs of cS, cL, cSmaxN, cLmaxN, cSminN, cLminN; a cS or a cL): off")
        return None


def plan5_spec(p5, fam, mode, data_dir, khi, saves, rank, bytes_rank1, holder):
    """PLAN5 on: what the runner is told (spec['plan5'] / the Musubi runner's G56_PLAN5): the end, K_max - the table's, never more
    members than a file of maxgb holds (bytes_rank1 x rank x saves per member) -, the window of the task's caption type where the entry
    has one (and its tie, except on the cells of PLAN5_ENTRY_TIE: the entry's), the prior of the post-training work at this rank. Two
    private lines at most: 'caption type' and 'member fit'."""
    lo, hi, tie = p5["lo"], p5["hi"], p5["tie"]
    kind = p5["kind"] if "kind" in p5 else plan5_kind(p5, mode, data_dir)
    if kind:
        lo, hi = p5["types"][kind]
        if (fam, mode) not in PLAN5_ENTRY_TIE:   # (PLAN5_ENTRY_TIE: the entry's own tie stays)
            tie = "fill" if kind == "S" else "stretch"
    khi = max(1, int(p5["khi"] or khi))
    per_member = float(bytes_rank1) * rank * saves
    if per_member > 0 and khi * per_member > p5["maxgb"] * 1e9:
        cap = max(1, int(p5["maxgb"] * 1e9 // per_member))
        log(f"plan5: K_max {khi} -> {cap}: a file of more members would pass {p5['maxgb']:g} GB ({per_member / 1e6:.0f} MB per member)")
        khi = cap
    p5.update(khi_eff=khi, post_b=p5["post_b32"] * rank / float(LF.EVAL_RANK), saves=saves, per_member=per_member, kind=kind)
    lo_text = "/".join(f"{x:g}" for x in lo) if isinstance(lo, list) else f"{lo:g}"
    log(f"member fit: {'type ' + kind + ', ' if kind else ''}window [{lo_text}, {hi:g}] L, K_max {khi}, tie {tie}, probe {p5['probe']}, cut "
        f"{p5['cut']:g}; plan5 end {p5['end']:g} s before the kill, post-training prior {PLAN5_POST_A:g} s + {p5['post_b']:.2f} s x "
        f"{saves} save(s) per member")
    return dict(end=p5["t_end"], khi=khi, lo=lo, hi=hi, tie=tie, probe=p5["probe"], cut=p5["cut"], post_a=PLAN5_POST_A, post_b=p5["post_b"],
                post_x=PLAN5_POST_X, cost=str(holder["p5_cost"]["path"]), saves=saves)


def plan5_svd_fits(holder, saves):
    """PLAN5: may a file of `saves` saves be written once more in soup_form() before the plan's end? -> (fits, the estimate, seconds left)"""
    p5 = holder["p5"]
    est = PLAN5_SVD_X * p5.get("post_b", p5["post_b32"]) * saves + PLAN5_SVD_PAD_S
    left = p5["t_end"] - time.time()
    return left >= est, est, left


def plan5_final(fam, mode, header, holder, out_dir):
    """PLAN5, after a child whose members' mean was written in the concat form while it trained (no tail mean: tail_final does this
    for the tail groups): the same exact mean once more in soup_form() when its estimate fits before the plan's end - else the concat
    file stays (the same delta, the same keys and shapes). Never raises."""
    mem = holder.get("p5_mem")
    try:
        if not mem or len(mem["sds"]) < 2 or mem.get("form") != "concat" or soup_form() == "concat":
            return
        fits, est, left = plan5_svd_fits(holder, len(mem["sds"]))
        if not fits:
            log(f"plan5: the concat file stays ({soup_form()} estimate {est:.0f} s, {left:.0f} s left before the plan's end)")
            return
        ship_mean(fam, mode, header, mem["sds"], out_dir, cost=holder.get("p5_cost"))
    except Exception as e:  # noqa: BLE001  (the concat file is that mean already)
        log(f"WARN plan5: {type(e).__name__}: {str(e)[:160]} - the concat file stays")


def utilization_line(holder, status, out_dir, t_launch, t_child_end):
    """
    
    
    """
    try:
        p5, now = holder["p5"], time.time()
        t0, w = _T0[0], holder["t_kill"] - _T0[0]
        pl = status.get("plan5") or {}
        t_tr0 = pl.get("t_train0") or t_launch
        ml = [str(x) for x in status.get("member_log") or []]
        steps = [int(m.group(1)) for m in (re.search(r"^member \d+/\d+ (?:done: steps|cut by the deadline at step) (\d+)", x) for x in ml) if m]
        why = ""
        for x in reversed(ml):
            m = re.search(r"next: stop \(([^;)]*)", x) or re.search(r"(cut by the deadline[^:]*)", x)
            if m:
                why = m.group(1)
                break
        size = (Path(out_dir) / "last.safetensors").stat().st_size
        log(f"utilization: window {w:.0f} s, work done at {now - t0:.0f} s = {100.0 * (now - t0) / w:.1f} %; training {t_tr0 - t0:.0f}.."
            f"{t_child_end - t0:.0f} s = {100.0 * (t_child_end - t_tr0) / w:.1f} %, load {t_tr0 - t_launch:.0f} s, post {now - t_child_end:.0f} s; "
            f"members {holder.get('members_done', 0)} of K_max {p5.get('khi_eff')}, steps {steps}; stop: {why or status.get('phase')}; "
            f"deadline {(pl.get('deadline') or 0.0) - t0:.0f} s, plan5 end {p5['t_end'] - t0:.0f} s ({p5['t_end'] - now:.0f} s left; the "
            f"watchdog in {holder['t_kill'] - WATCHDOG_S - now:.0f} s); file {size / 1e6:.0f} MB, upload >= "
            f"{size / 1e6 / (600.0 + holder['t_kill'] - now):.1f} MB/s")
    except Exception as e:  # noqa: BLE001  (a line of the private log never costs the task)
        log(f"utilization: not read ({type(e).__name__}: {str(e)[:120]})")


def soup_form():
    """The form of a mean file: SOUP_FORM or G56NF_SOUP_FORM (a name g56_lorafmt does not know is ignored)."""
    f = knob("SOUP_FORM", SOUP_FORM, str)
    if f not in LF.SOUP_FORMS:
        log(f"WARN G56NF_SOUP_FORM {f!r} ignored: {SOUP_FORM}")
        return SOUP_FORM
    return f


def step_cap(fam, mode, n=None):
    """The length of one member = the step cap of a single run: G56NF_STEPS; else, when a per-image rule exists
    (G56NF_STEPS_PER_IMAGE=c[,lo,hi], else STEPS_PER_IMAGE) and the task size n is known, c x n clamped to [lo, hi]; else
    TEMPLATE_STEPS. n = the training images (edit: pairs) the driver wrote for the trainer (prepare_t2i / prepare_edit: the
    rows it skipped are not counted); the rule's arithmetic goes to the private log."""
    if os.environ.get("G56NF_STEPS"):
        return knob("STEPS", TEMPLATE_STEPS[(fam, mode)], int)
    rule = STEPS_PER_IMAGE.get((fam, mode))
    v = os.environ.get("G56NF_STEPS_PER_IMAGE")
    if v:
        try:
            x = [float(y) for y in v.split(",")]
            rule = (x[0], int(x[1]) if len(x) > 1 else 1, int(x[2]) if len(x) > 2 else 10 ** 9)
        except Exception as e:  # noqa: BLE001
            log(f"WARN G56NF_STEPS_PER_IMAGE ignored ({type(e).__name__})")
    if rule and n:
        c, lo, hi = rule
        steps = int(min(hi, max(lo, round(c * n))))
        log(f"steps per image: c {c:g} x N {n} -> {steps} (clamp {lo}..{hi})")
        return steps
    return TEMPLATE_STEPS[(fam, mode)]


def log(*a, public=False):
    line = f"[nf {time.time() - _T0[0]:7.1f}s] " + " ".join(str(x) for x in a)
    try:
        with open(LOGFILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if VERBOSE or public:
        print(line if VERBOSE else " ".join(str(x) for x in a), flush=True)


_T0 = [time.time()]


def parse(argv):
    import argparse
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-zip", required=True)
    p.add_argument("--model-type", required=True, choices=sorted(LF.FAMILIES))
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", required=True, type=float)
    p.add_argument("--trigger-word", default=None)
    p.add_argument("--training-mode", default=LF.T2I, choices=(LF.T2I, LF.EDIT))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dataset", default=None)
    a, _ = p.parse_known_args(argv)
    return a


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def dataset_zip(a):
    if a.dataset:
        return Path(a.dataset)
    z = CACHE / "datasets" / f"{a.task_id}_tourn.zip"
    if z.exists():
        return z
    if a.dataset_zip and Path(a.dataset_zip).exists():
        return Path(a.dataset_zip)
    raise FileNotFoundError(f"dataset zip not found ({z})")


def _safe_extract(zp, dest):
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            t = (dest / m.filename).resolve()
            if t != root and root not in t.parents:
                raise ValueError(f"unsafe path in the dataset zip: {m.filename}")
        z.extractall(dest)
    return dest


def lpt_png_bytes(raw, radius):
    """
    
    
    
    """
    import io
    from PIL import Image, ImageFilter, ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    im = im.convert("RGB").filter(ImageFilter.GaussianBlur(radius))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def prepare_t2i(zp, out, radius=0.0):
    """The evaluator's own reading (its dataset reader's read_dataset: sorted names, caption = the sibling .txt, raw text) and
    writing (diffusion_objective._write_dataset: <index:06d>.png = the raw bytes, <index:06d>.txt = the caption).
    The ONE writer of the text-to-image training folder (the L2 relaunch, the K members and the two entries train on that folder).
    radius > 0 (lpt_radius: LPT_RADIUS / G56NF_LPT_RADIUS): every image is written as lpt_png_bytes(its bytes, radius) instead - the
    caption, the name, the order and the count as above; an image that cannot be blurred keeps its raw bytes (one WARN line). One
    private line: the radius, the images written blurred and the seconds it took. radius 0 = the raw bytes, as before."""
    out.mkdir(parents=True, exist_ok=True)
    n = blurred = 0
    failed = []
    t0 = time.time()
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        have = set(names)
        for name in sorted(names):
            if not name.lower().endswith(IMAGE_EXT) or "__MACOSX/" in name or Path(name).name.startswith("._"):
                continue
            cap = str(Path(name).with_suffix(".txt"))
            caption = z.read(cap).decode("utf-8", errors="replace") if cap in have else ""
            stem = f"{n:06d}"
            data = z.read(name)
            if radius > 0:
                try:
                    data = lpt_png_bytes(data, radius)
                    blurred += 1
                except Exception as e:  # noqa: BLE001  (an image the blur cannot read keeps its raw bytes: never a lost task)
                    failed.append(type(e).__name__)
            (out / f"{stem}.png").write_bytes(data)
            (out / f"{stem}.txt").write_text(caption, encoding="utf-8")
            n += 1
    if n == 0:
        raise ValueError("no training images in the dataset zip")
    if radius > 0:
        log(f"lpt: radius {radius:g}, {blurred} images, {time.time() - t0:.1f} s")
        if failed:
            log(f"WARN lpt: {len(failed)} of {n} images not blurred ({', '.join(sorted(set(failed)))}): their raw bytes are written")
    return out, [], n


def prepare_edit(zp, out, extract):
    """The edit contract (validator/tasks/datasets/preparation.py:46-82; the evaluator's reader read_generation_dataset):
    one manifest.jsonl, rows {sample_id, instruction, sources: [1 path], target}. Written like the evaluator's
    _write_dataset: targets/<k:06d>.png + .txt (the instruction), controls/0/<k:06d>.png (the source)."""
    _safe_extract(zp, extract)
    mans = [m for m in extract.rglob("manifest.jsonl") if "__MACOSX" not in m.parts]
    if len(mans) != 1:
        raise ValueError(f"edit dataset needs exactly one manifest.jsonl, found {len(mans)}")
    man = mans[0]
    root = man.parent.resolve()
    tdir, cdir = out / "targets", out / "controls" / "0"
    tdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            srcs, tgt, ins = row.get("sources"), row.get("target"), row.get("instruction")
            if not isinstance(srcs, list) or len(srcs) < 1 or not isinstance(tgt, str) or not isinstance(ins, str):
                raise ValueError("bad row")
            sp, tp = (root / srcs[0]).resolve(), (root / tgt).resolve()
            if root not in sp.parents or root not in tp.parents or not sp.is_file() or not tp.is_file():
                raise ValueError("bad path")
        except Exception as e:  # noqa: BLE001  (one bad row never ends the task)
            skipped += 1
            log(f"WARN edit manifest row skipped: {type(e).__name__}")
            continue
        stem = f"{n:06d}"
        (tdir / f"{stem}.png").write_bytes(tp.read_bytes())
        (tdir / f"{stem}.txt").write_text(ins, encoding="utf-8")
        (cdir / f"{stem}.png").write_bytes(sp.read_bytes())
        n += 1
    if n == 0:
        raise ValueError("no usable rows in the edit manifest")
    if skipped:
        log(f"edit manifest: {skipped} rows skipped")
    return tdir, [cdir], n


# ----------------------------------------------------------------------------------------------------------------------
# model folders and offline patches
# ----------------------------------------------------------------------------------------------------------------------
def model_dir(repo):
    """/cache/models/<org--repo>: the task's own snapshot (the validator's downloader, not anonymized for image tasks;
    same rule as the ComfyUI families: never another cache folder)."""
    return CACHE / "models" / repo.replace("/", "--")


STAGED_HUB = CACHE / "hf_cache"
STAGED_MODELS = CACHE / "ai_toolkit_models"   # its Comfy-layout files (AI_TOOLKIT_COMFY_FILES) = the official entrypoint's MODELS_PATH
MOSS_COMFY_TE = "text_encoders/qwen3.5_9b_int8_convrot.safetensors"


def staged_hub_snapshot(repo, files=("config.json",), shards=False):
    """
    
    """
    base = STAGED_HUB / ("models--" + repo.replace("/", "--"))
    cands = []
    try:
        ref = base / "refs" / "main"
        if ref.is_file():
            cands.append(base / "snapshots" / ref.read_text().strip())
        if (base / "snapshots").is_dir():
            cands += sorted(p for p in (base / "snapshots").iterdir() if p not in cands)
    except Exception:  # noqa: BLE001
        return None
    for d in cands:
        try:
            if not all((d / f).is_file() and (d / f).stat().st_size > 0 for f in files):
                continue
            if shards:
                names = set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values())
                if not names or not all((d / s).is_file() and (d / s).stat().st_size > 0 for s in names):
                    continue
            return d
        except Exception:  # noqa: BLE001
            continue
    return None


def staged_runtime(fam, mode):
    """
    
    
    """
    if fam == "flux2-klein-4b":
        te = staged_hub_snapshot("Qwen/Qwen3-4B", ("config.json", "tokenizer.json", "tokenizer_config.json",
                                                   "model.safetensors.index.json"), shards=True)
        vae = staged_hub_snapshot("ai-toolkit/flux2_vae", ("ae.safetensors",))
        return dict(hf_hub_cache=str(STAGED_HUB)) if te and vae else None
    if fam == "moss-image-base":
        te = STAGED_MODELS / MOSS_COMFY_TE
        need = ("config.json",) + (("preprocessor_config.json",) if mode == LF.EDIT else ())
        if te.is_file() and te.stat().st_size > 0 and staged_hub_snapshot("Qwen/Qwen3.5-9B", need):
            return dict(hf_hub_cache=str(STAGED_HUB), models_path=str(STAGED_MODELS))
    return None


def klein_te_dir(snap, work):
    """
    
    """
    d = work / "klein_te"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for sub in ("text_encoder", "tokenizer"):
        src = snap / sub
        if not src.is_dir():
            raise FileNotFoundError(f"{src} missing (FLUX.2 Klein snapshot)")
        for f in sorted(src.iterdir()):
            if f.is_file() and not (d / f.name).exists():
                (d / f.name).symlink_to(f.resolve())
    return d


def moss_processor_dir(work):
    """MOSS edit image processor: a staged copy first (the hub cache layout under /cache/hf_cache, then any cached model
    folder holding a Qwen2VL-type preprocessor_config.json), else MOSS_EDIT_PROCESSOR_CONST."""
    cands = sorted((CACHE / "hf_cache").glob("models--Qwen--Qwen3.5-9B/snapshots/*/preprocessor_config.json"))
    cands += sorted((CACHE / "models").glob("Qwen--Qwen3.5-9B/preprocessor_config.json"))
    for c in cands:
        try:
            if json.loads(c.read_text()).get("image_processor_type", "").startswith("Qwen2VLImageProcessor"):
                return c.parent, "staged"
        except Exception:  # noqa: BLE001
            continue
    if not knob("MOSS_EDIT_PROCESSOR_CONST", MOSS_EDIT_PROCESSOR_CONST, bool):
        return None, "none"
    d = work / "qwen35_processor"
    d.mkdir(parents=True, exist_ok=True)
    (d / "preprocessor_config.json").write_text(json.dumps(QWEN35_PREPROCESSOR, indent=4))
    return d, "const"


def build_config(fam, mode, snap, data_dir, ctrl_dirs, run_dir, te_dir=None, staged=None, bound=None, knobs=None, gc=None):
    """
    
    
    """
    ck = config_knobs(fam, mode) if knobs is None else knobs
    mb = dict(MODEL_BASE[(fam, mode)])
    model = dict(name_or_path=str(snap), arch=mb.pop("arch", ARCH[fam]))
    model.update(mb)
    if fam == "moss-image-base" and not staged:
        model["te_name_or_path"] = str(snap / "text_encoder")   # stock Qwen3.5-9B, loaded as a local folder (_mixin.py:455-545)
    rc = recipe(fam, mode)
    ds = dict(folder_path=str(data_dir), caption_ext="txt", caption_dropout_rate=rc["caption_dropout"], resolution=rc["resolution"])
    if fam != "hidream-o1":
        ds["cache_latents_to_disk"] = True   # HiDream is pixel space (FakeVAE): the template caches no latents
    if ctrl_dirs:
        ds["control_path"] = [str(c) for c in ctrl_dirs]
    if ck.get("flip"):        # G56NF_FLIP (text-to-image only: config_knobs)
        ds["flip_x"] = True
    datasets = [ds]
    if ck.get("two_entry"):   # G56NF_TWO_ENTRY: the same folder twice - every caption kept, every caption dropped
        datasets = [dict(ds, caption_dropout_rate=0.0), dict(ds, caption_dropout_rate=1.0)]
    train = dict(batch_size=ck.get("batch", 1), steps=1_000_000, lr=rc["lr"], optimizer=ck.get("optimizer", CFG_OPTIMIZERS[0]),
                 lr_scheduler="constant",
                 gradient_checkpointing=gc_for_task(fam, mode, bound) if gc is None else gc, noise_scheduler="flowmatch",
                 timestep_type=rc["timestep_type"],
                 min_denoising_steps=rc["tmin"], max_denoising_steps=rc["tmax"], dtype="bf16",
                 cache_text_embeddings=rc["cache_te"], train_unet=True, train_text_encoder=False,
                 skip_first_sample=True, disable_sampling=True)
    if "grad_accum" in ck:      # G56NF_GRAD_ACCUM
        train["gradient_accumulation"] = ck["grad_accum"]
    if "ts_bias" in ck:         # G56NF_TS_BIAS
        train["content_or_style"] = ck["ts_bias"]
    if "weight_decay" in ck:    # G56NF_WEIGHT_DECAY
        train["optimizer_params"] = {"weight_decay": ck["weight_decay"]}
    if "max_grad_norm" in ck:   # G56NF_MAX_GRAD_NORM
        train["max_grad_norm"] = ck["max_grad_norm"]
    rank = ck.get("rank", LF.EVAL_RANK)   # G56NF_RANK: rank = alpha, the scale stays 1
    proc = dict(type="g56_timed_trainer", training_folder=str(run_dir), device="cuda",
                performance_log_every=knob("PERF_LOG_EVERY", 0, int),
                network=dict(type="lora", linear=rank, linear_alpha=rank),
                save=dict(dtype="float32", save_every=10 ** 9, max_step_saves_to_keep=1, save_format="diffusers"),
                datasets=datasets, train=train, model=model)
    return {"job": "extension", "config": {"name": "g56", "process": [proc]}, "meta": {"name": "g56", "version": "1.0"}}


# ----------------------------------------------------------------------------------------------------------------------
# shipping
# ----------------------------------------------------------------------------------------------------------------------
def finite(sd):
    import torch
    return all(bool(torch.isfinite(v).all()) for v in sd.values())


def shipped_form(fam, mode, header, src, normalize=None, what="shipped"):
    """A runner snapshot in the form that leaves the container: normalized (AI Toolkit save -> shipped layout; Musubi saves: keys
    as saved, metadata dropped), loader-checked, finite. -> (state dict, the check report), or None (logged: not `what`)."""
    try:
        sd = (normalize or LF.normalize_aitk)(fam, LF.load_file_tensors(src))
        rep = LF.check(fam, mode, header, sd, expect_full=True)
    except Exception as e:  # noqa: BLE001
        log(f"WARN snapshot {src.name}: not {what} ({type(e).__name__}: {str(e)[:160]})")
        return None
    if not rep["ok"]:
        log(f"WARN snapshot {src.name}: format check failed, not {what}: dropped {len(rep['dropped'])} shape "
            f"{rep['shape_errors'][:2]} cut {len(rep['cut'])} scale {rep['scale_errors'][:2]} coverage {rep['coverage']:.3f}")
        return None
    if not finite(sd):
        log(f"WARN snapshot {src.name}: non-finite values, not {what}")
        return None
    return sd, rep


def scaled_for_export(fam, mode, header, sd, rep, s, name):
    """The export scale on a file that passed shipped_form: (sd, rep) -> (s x its delta, that file's own loader report, a log
    note). s == 1 -> the inputs, untouched. The scaled file is checked like any shipped file; if it does not pass, the file is
    shipped as trained (logged) - the scale never costs a shipped file."""
    if s == 1.0:
        return sd, rep, ""
    try:
        scaled = LF.scale_delta(sd, s)
        rep_s = LF.check(fam, mode, header, scaled, expect_full=True)
        if not rep_s["ok"] or not finite(scaled):
            raise ValueError("the scaled file fails the loader check")
        return scaled, rep_s, f", export scale {s:g}"
    except Exception as e:  # noqa: BLE001
        log(f"WARN export scale {s:g} not applied to {name} ({type(e).__name__}: {str(e)[:120]}): shipped as trained")
        return sd, rep, ""


def _as_saved(fam, sd):
    """the shipped form of a Musubi save: the keys as saved (already the evaluator's layout); shipped_form drops the metadata"""
    return dict(sd)


def tailmean_fracs(fam, mode):
    """TAILMEAN or G56NF_TAILMEAN for the task's (family, mode): the window as sorted fractions of the step cap, or None (off:
    unset, '0', 'off'). '1' / 'on' / 'true' / 'yes' = TAILMEAN_WINDOW. A value that does not parse, or a fraction outside [0.5, 1]
    (desk HiDream's window read: a save at L/4 hurt on every read), -> one WARN line, off."""
    v = (os.environ.get("G56NF_TAILMEAN") or "").strip().lower()
    if not v:
        return TAILMEAN.get((fam, mode)) or None
    if v in ("0", "off", "none", "false", "no"):
        return None
    if v in ("1", "on", "true", "yes"):
        return TAILMEAN_WINDOW
    try:
        fr = tuple(sorted({float(x) for x in v.split(",") if x.strip()}))
        if fr and all(0.5 <= x <= 1.0 for x in fr):
            return fr
    except ValueError:
        pass
    log(f"WARN G56NF_TAILMEAN {v[:40]!r} ignored (fractions of the step cap in [0.5, 1], e.g. 0.5,0.75,1; or 1 = that window): off")
    return None


def tail_plan(fracs, cap, every):
    """The window on the save grid: each fraction x the step cap, to the nearest save (a multiple of the save interval; a tie to the
    earlier save: 3L/4 of 500 = 375 -> 370 with a save every 10 steps; at least the first save). The cap is the run's own last save:
    a window step less than one save interval before it is that save, not kept apart. -> (the kept steps, the window's spacing =
    the smallest gap between consecutive steps of the window with the cap) or (None, why): no step to keep, or two fractions on one
    save."""
    pts = []
    for f in fracs:
        q = f * cap / every
        n = int(q)
        pts.append(max(every, (n + 1 if q - n > 0.5 + 1e-9 else n) * every))
    if len(set(pts)) < len(pts):
        return None, f"two fractions on one save of the grid ({pts}: cap {cap}, a save every {every} steps)"
    kept = sorted(s for s in set(pts) if s <= cap - every)
    if not kept:
        return None, f"no window step at least one save before the step cap {cap}"
    grid = kept + [cap]
    return kept, min(b - a for a, b in zip(grid, grid[1:]))


def tail_need(t_cap, members):
    """the seconds the tail mean is given before it starts: TAILMEAN_COST_X x the slowest kept save's seconds (copy + load + loader
    check) x the members + TAILMEAN_MARGIN_S"""
    return TAILMEAN_COST_X * t_cap * members + TAILMEAN_MARGIN_S


def tail_capture(fam, mode, header, f, step, tail):
    """G56NF_TAILMEAN: the save `f` of a window step -> a copy in the work folder (tail['dir']/s<step>.safetensors), loader-checked;
    its seconds feed the time estimate. A save that fails is not retried and not a member. Never raises."""
    t = time.time()
    try:
        tail["dir"].mkdir(parents=True, exist_ok=True)
        dst, tmp = tail["dir"] / f"s{step}.safetensors", tail["dir"] / f".s{step}.tmp"
        shutil.copyfile(f, tmp)
        os.replace(tmp, dst)
        if shipped_form(fam, mode, header, dst, normalize=_as_saved, what="kept for the tail mean") is None:
            tail["failed"].add(step)
            dst.unlink()
            return
        tail["got"][step] = dst
        tail["t_cap"] = max(tail["t_cap"], time.time() - t)
    except Exception as e:
        tail["failed"].add(step)
        log(f"WARN tail mean: the save at step {step} not kept ({type(e).__name__}: {str(e)[:120]})")


def tail_poll(fam, mode, header, files, steps, tail, tseen, done):
    """G56NF_TAILMEAN, one poll of musubi_child: EVERY save of a window step not kept yet (not only the newest one), once its size
    held still for one poll or the child ended -> tail_capture."""
    for f in files:
        try:
            step, size = musubi_step(f.name, steps), f.stat().st_size
        except Exception:  # noqa: BLE001  (Musubi removed it meanwhile)
            continue
        if step in tail["steps"] and step not in tail["got"] and step not in tail["failed"] and (tseen.get(f.name) == size or done):
            tail_capture(fam, mode, header, f, step, tail)
        tseen[f.name] = size


def tail_ship(fam, mode, header, holder, out_dir, steps):
    """G56NF_TAILMEAN, after the training child (no relaunch) of a run that reached the step cap: the upload file becomes the EXACT
    mean of the deltas of the members (the kept saves + the last shipped save; g56_lorafmt.concat_mean), with the export scale of a
    finished run, once
    it passes concat_mean's structure checks, the loader mirror and finiteness; atomically swapped in. Whatever fails, or when the
    time left before the watchdog's stop is short of tail_need(), the last save stays shipped (one line). Never raises."""
    t0 = time.time()
    try:
        tail, last = holder.get("tail"), holder.get("musubi_last")
        if last is None:
            log("tail mean: no save was shipped: nothing to average")
            return
        final, fpath = last
        if final < tail["cap"]:   # (desk HiDream's 0.5 h read: a stopped run is better off with its last save)
            log(f"tail mean: the run stopped at step {final} before the step cap {tail['cap']} - the last save stays (a mean only for a "
                f"run that reached the cap)")
            return
        use = sorted(s for s in tail["got"] if s < final) + [final]
        head = (f"tail mean (window {tail['fracs']} of the cap {tail['cap']}, kept steps {sorted(tail['got'])}): members at steps {use} "
                f"(the last: {Path(fpath).name})")
        if len(use) < 2:
            log(f"{head}: one member - the last save stays")
            return
        left, need = holder.get("t_kill", float("inf")) - WATCHDOG_S - time.time(), tail_need(tail["t_cap"], len(use))
        if left < need:
            log(f"WARN {head}: {left:.0f} s left before the watchdog's stop < the estimate {need:.0f} s - the last save stays")
            return
        got = shipped_form(fam, mode, header, Path(fpath), normalize=_as_saved, what="averaged") if Path(fpath).is_file() else None
        if got is None:
            log(f"WARN {head}: the last save is not readable any more - it stays shipped")
            return
        sd, rep = LF.concat_mean([LF.load_file_tensors(tail["got"][s]) for s in use[:-1]] + [got[0]])
        s = export_scale(fam, mode, 1) if final >= steps else 1.0   # a finished run's file carries the export scale (ship())
        if s != 1.0:
            sd = LF.scale_delta(sd, s)
        if final >= steps:   # EXPORT_GROUP_SCALE: ship()'s rule - a finished run's file, after its export scale
            sd = group_scaled(fam, mode, header, sd, 1, "the tail mean")[0]
        chk = LF.check(fam, mode, header, sd, expect_full=True)
        bad = rep["problems"][:1] or ([] if chk["ok"] else ["the loader check fails"]) or ([] if finite(sd) else ["non-finite values"])
        if bad:
            log(f"WARN {head}: not shipped ({bad[0]}) - the last save stays")
            return
        LF.save_atomic(sd, Path(out_dir) / "last.safetensors")
        log(f"{head}: shipped - {len(use)} members, ranks {rep['ranks']}, alpha kept" + (f", export scale {s:g}" if s != 1.0 else "")
            + f"; {len(sd)} tensors, coverage {chk['coverage']:.3f}; {time.time() - t0:.1f} s (estimate {need:.0f} s, {left:.0f} s left)")
    except Exception as e:
        log(f"WARN tail mean: {type(e).__name__}: {str(e)[:160]} - the last save stays")


def tail_steps(fracs, L):
    """g56_aitk_runner.tail_window's rule, the driver's own copy (newfam_check case V2 holds them equal): the window steps of a member
    of length L - each fraction x L to the nearest step (a tie to the earlier step), at least 1, below L, once each, in order."""
    out = set()
    for f in fracs:
        q = float(f) * L
        n = int(q)
        st = max(1, n + 1 if q - n > 0.5 + 1e-9 else n)
        if st < L:
            out.add(st)
    return sorted(out)


def tail_new(fam, mode, header, save_root, tails, tail):
    """G56NF_TAILMEAN on the AI Toolkit path, one poll of aitk_child: every tail_<k>_<step>.safetensors of the runner not handled yet ->
    its shipped form (normalized, loader-checked, finite; unscaled) in tails[k][step]; the runner's file is removed. The slowest load +
    check feeds the time rule (tail['t_cap']). Never raises."""
    try:
        files = sorted(save_root.glob("tail_*.safetensors")) if save_root.is_dir() else []
    except Exception:  # noqa: BLE001
        return
    for f in files:
        if f.name in tail["seen"]:
            continue
        tail["seen"].add(f.name)
        t = time.time()
        try:
            _, k, step = f.stem.split("_")
            got = shipped_form(fam, mode, header, f, what="kept for the tail mean")
            if got is not None:
                tails.setdefault(int(k), {})[int(step)] = got[0]
                tail["t_cap"] = max(tail["t_cap"], time.time() - t)
            f.unlink()
        except Exception as e:
            log(f"WARN tail mean: {f.name} not kept ({type(e).__name__}: {str(e)[:120]})")


def tail_group(k, f, sd, tails, tail):
    """G56NF_TAILMEAN on the AI Toolkit path: member k's group = its window saves kept so far (tail_steps of the member file's own step
    = its length, in step order) + its last save sd (the member file f, shipped form). A save of member k at any other step is left
    out (said); member k's saves leave `tails` either way."""
    L = snap_step(f) or 0
    win = set(tail_steps(tail["fracs"], snap_len(f) or L))   # (PLAN5: the member's PLANNED length, when its header holds one)
    got = tails.pop(k, {})
    use, odd = sorted(x for x in got if x in win), sorted(x for x in got if x not in win)
    log(f"tail mean: member {k}: saves at steps {use} and its last ({L})" + (f"; not in its window, left out: {odd}" if odd else ""))
    return [got[x] for x in use] + [sd]


def tail_flat(groups, s, tail, fallback=True):
    """
    
    
    
    """
    K = len(groups)
    flat = [x for g in groups for x in g]
    ws = [s / (K * len(g)) for g in groups for _ in g]
    t_left = tail["t_kill"] - WATCHDOG_S - time.time()
    rate = max(tail["rate"], tail["t_cap"])
    x = tail.get("cost_x", TAILMEAN_COST_X)   # (PLAN5: its own factor - the plan kept the time of a merge of saves x K at that factor)
    t_all = x * rate * len(flat) + TAILMEAN_MARGIN_S
    if t_left >= t_all:
        return flat, ws, f", tail mean {len(flat)} saves ({'+'.join(str(len(g)) for g in groups)})"
    t_last = x * rate * K + TAILMEAN_MARGIN_S
    if fallback and t_left >= t_last:
        log(f"WARN tail mean: {t_left:.0f} s left before the watchdog's stop < the estimate {t_all:.0f} s for {len(flat)} saves - the "
            f"members' last saves alone ({t_last:.0f} s)")
        return [g[-1] for g in groups], None, ", tail mean: no time, the members' last saves"
    raise TimeoutError(f"{t_left:.0f} s left before the watchdog's stop < {t_all if not fallback else t_last:.0f} s"
                       + ("" if not fallback else " even for the members' last saves"))


def tail_final(fam, mode, header, holder, out_dir):
    """G56NF_TAILMEAN on the AI Toolkit path, after a K-member child: the members' tail means were shipped in the concat form while
    the child trained (exact, no QR / SVD beside the training); now the same mean once more in soup_form() when the time rule covers
    it (no fallback: the concat file is that mean already) - else, or when soup_form() is concat itself, the concat file stays.
    Never raises."""
    mem, tail = holder.get("tail_mem"), holder.get("tail_aitk")
    try:
        if not mem or not mem.get("groups") or tail is None or soup_form() == "concat":
            return
        if holder.get("p5"):   # PLAN5: the soup_form() merge only when its estimate fits before the plan's end
            fits, est, left = plan5_svd_fits(holder, sum(len(g) for g in mem["groups"]))
            if not fits:
                log(f"plan5: the concat file stays ({soup_form()} estimate {est:.0f} s, {left:.0f} s left before the plan's end)")
                return
        ship_mean(fam, mode, header, mem["sds"], out_dir, groups=mem["groups"], tail=tail, fallback=False, cost=holder.get("p5_cost"))
    except Exception as e:
        log(f"WARN tail mean: {type(e).__name__}: {str(e)[:160]} - the concat file stays")


def tail_ship_aitk(fam, mode, header, holder, out_dir, spec):
    """G56NF_TAILMEAN on the AI Toolkit path, a single run (K_max 1), after its training child (no relaunch): when its final snapshot
    is at the step cap, the upload file becomes the exact mean of its window saves and that snapshot (ship_mean with one group: each
    save 1 / m, export_scale(1)); a run the deadline stopped before the cap ships its last save as it is (the window read of desk
    HiDream). Never raises."""
    tail, cap = holder["tail_aitk"], int(spec.get("max_steps") or 0)
    try:
        f = Path(spec["save_root"]) / "g56_final.safetensors"
        L = snap_step(f) if f.is_file() else None
        if L is None or L < cap:
            log(f"tail mean: the run stopped at step {L} before the step cap {cap} - the last save stays (a mean only for a run that "
                f"reached the cap)")
            return
        got = shipped_form(fam, mode, header, f, what="averaged")
        if got is None:
            log("WARN tail mean: the final snapshot does not pass the checks - the shipped file stays")
            return
        grp = tail_group(1, f, got[0], holder.get("tails") or {}, tail)
        if len(grp) < 2:
            log("tail mean: no window save was kept - the shipped file stays")
            return
        ship_mean(fam, mode, header, [got[0]], out_dir, groups=[grp], tail=tail)
    except Exception as e:
        log(f"WARN tail mean: {type(e).__name__}: {str(e)[:160]} - the shipped file stays")


def ship(fam, mode, header, src, out_dir, normalize=None, k_done=0, cost=None):
    """A runner snapshot -> its shipped form (shipped_form), atomically shipped. k_done = the finished members it holds: 0 = a
    snapshot of an unfinished run, shipped as trained; 1 = a run that reached its step cap, shipped with export_scale(fam, mode,
    1) (1.0 = as trained). Returns the check report of the file written (None = not shipped). cost (PLAN5; None = nothing): the
    seconds of the load + check and of the write go to the runner's cost file."""
    t = time.time()
    got = shipped_form(fam, mode, header, src, normalize)
    if got is None:
        return None
    t_form = time.time() - t
    s = export_scale(fam, mode, k_done) if k_done else 1.0
    sd, rep, note = scaled_for_export(fam, mode, header, *got, s, src.name)
    if k_done and (s == 1.0 or note):   # EXPORT_GROUP_SCALE: a finished run's file, after its export scale (not when that one failed)
        sd, rep = group_scaled(fam, mode, header, sd, k_done, src.name, rep)
    t = time.time()
    LF.save_atomic(sd, out_dir / "last.safetensors")
    plan5_note(cost, form_s=t_form, merge_s=time.time() - t)
    log(f"shipped {src.name}: {len(sd)} tensors, coverage {rep['coverage']:.3f}, max rel err {rep['max_rel_err']:.1e}{note}")
    return rep


def inside(path, parent):
    """True when `path` is `parent` or lies under it (both resolved)."""
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


def keep_root(work, out_dir):
    """
    
    """
    d = os.environ.get("G56NF_KEEP_DIR")
    if d:
        if Path(d).is_absolute() and not inside(d, out_dir):
            return Path(d)
        log(f"WARN G56NF_KEEP_DIR {d} ignored (not absolute, or inside the upload folder): the work folder is used")
    return Path(work) / "keep"


def keep_plan(work, out_dir):
    """
    
    
    
    """
    try:
        steps = sorted({s for s in knob("KEEP_STEPS", [], list) if s > 0})
    except Exception as e:  # noqa: BLE001
        log(f"WARN G56NF_KEEP_STEPS ignored ({type(e).__name__}: {str(e)[:80]})")
        return {}
    if not steps:
        return {}
    kd = keep_root(work, out_dir)
    if inside(kd, out_dir):
        log(f"WARN G56NF_KEEP_STEPS ignored: the keep folder {kd} is inside the upload folder")
        return {}
    return dict(keep_steps=steps, keep_dir=str(kd))


def keep_new(fam, mode, header, save_root, keep_dir, done):
    """G56NF_KEEP_STEPS: every keep snapshot of the runner not handled yet (`done`: names seen) -> <keep_dir>/s<step>.safetensors in
    the shipped form (shipped_form: what last.safetensors would be had training stopped there; tensors only), the step taken from
    the snapshot's header; the runner's copy is removed. Never raises, never touches the upload folder."""
    try:
        files = sorted(save_root.glob("keep_*.safetensors")) if save_root.is_dir() else []
    except Exception:  # noqa: BLE001
        return
    for f in files:
        if f.name in done:
            continue
        done.add(f.name)
        try:
            step = snap_step(f)
            if step is None:
                log(f"WARN keep {f.name}: no step in its header, not kept")
                continue
            got = shipped_form(fam, mode, header, f, what="kept")
            if got is None:
                continue
            LF.save_atomic(got[0], keep_dir / f"s{step}.safetensors")
            log(f"kept step {step}: {len(got[0])} tensors -> {keep_dir.name}/s{step}.safetensors")
            f.unlink()
        except Exception as e:  # noqa: BLE001
            log(f"WARN keep {f.name}: {type(e).__name__}: {str(e)[:160]}")


def ship_mean(fam, mode, header, sds, out_dir, groups=None, tail=None, form=None, fallback=True, cost=None, gate=True):
    """
    
    
    
    
    
    
    
    
    
    
    """
    k, form = len(sds), form or soup_form()
    s = export_scale(fam, mode, k)
    t = time.time()
    flat, ws, note = list(sds), None, ""
    try:
        if groups is not None:
            flat, ws, note = tail_flat(groups, s, tail, fallback)
            if cost is not None and ws is None:   # PLAN5: said in a line of its own (a reader counts the files the tail mean did not act on)
                log(f"plan5: tail mean dropped ({k} members, {sum(len(g) for g in groups)} saves: no time before the watchdog's stop - the "
                    f"members' last saves alone)")
        n = len(flat)
        if cost is not None and n > 1 and t + PLAN5_POST_X * cost["merge_s"] * n > cost["t_stop"]:
            raise TimeoutError(f"{cost['t_stop'] - t:.0f} s left before the watchdog's stop < {PLAN5_POST_X * cost['merge_s'] * n:.0f} s")
        kw = {} if gate or form != "concat" else {"gate": False}
        sd, rep = LF.mean_delta(flat, scale=s, form=form, **kw) if ws is None else LF.mean_delta(flat, scale=s, form=form, weights=ws, **kw)
        rank = rep["rank"][0] if len(rep["rank"]) == 1 else None
        if n > 1:
            rank = LF.EVAL_RANK if form == "svd32" else rank
            if form != "svd32" and not rep["max_rel_err"] <= LF.MEAN_TOL:
                raise ValueError(f"not the exact mean (max relative error {rep['max_rel_err']:.2e})")
            uni = LF.uniform_rank(sd)
            if not uni["ok"] or uni["rank"] != rank:
                raise ValueError(f"not one rank {rank} without alpha: {uni['problems'] or uni['rank']}")
        sd = group_scaled(fam, mode, header, sd, k, f"the mean of {k} member(s)")[0]   # EXPORT_GROUP_SCALE: after the export scale
        chk = LF.check(fam, mode, header, sd, expect_full=True)
        if not chk["ok"] or not finite(sd):
            raise ValueError("the loader check fails")
    except Exception as e:  # noqa: BLE001
        log(f"WARN members: the mean of {k} members is not shipped ({type(e).__name__}: {str(e)[:160]}); the shipped file stays")
        if cost is not None:   # PLAN5: said in a line of its own (the file that stays holds fewer members, or another form)
            log(f"plan5: merge skipped ({k} members{', their tail mean' if groups is not None else ''}: {type(e).__name__}) - the shipped file stays")
        return False
    tw = time.time()
    LF.save_atomic(sd, out_dir / "last.safetensors")
    tw = time.time() - tw
    dt = time.time() - t
    if tail is not None and n > 1:
        tail["rate"] = max(tail["rate"], dt / n) if cost is None else dt / n   # (PLAN5: the last merge's, not the slowest ever)
    plan5_note(cost, merge_s=dt / max(1, n))
    log(f"merged {k} members: {dt:.1f} s, rank {rank}, scale {s:g}, form {form if n > 1 else 'member'}, "
        f"max rel err {rep['max_rel_err']:.1e}, fallback {len(rep['fallback'])}{note}" + (f", write {tw:.1f} s" if groups is not None else ""))
    log(f"shipped members 1-{k}: {len(sd)} tensors, coverage {chk['coverage']:.3f}")
    return True


def members_new(fam, mode, header, save_root, out_dir, mem, keep_dir=None):
    """K members: every finished member of the runner not handled yet (member_<k>.safetensors, in order) is brought to the shipped
    form - one that does not pass is left out of the mean - and the upload file is rewritten with the exact mean of the members so
    far (ship_mean). mem = dict(sds=[state dicts], seen={file names}). keep_dir (G56NF_KEEP_MEMBERS): each member also goes there,
    unscaled, as m<k>.safetensors. Never raises; whatever fails, the shipped file stays."""
    try:
        files = sorted(f for f in save_root.glob("member_*.safetensors") if f.name not in mem["seen"]) if save_root.is_dir() else []
    except Exception:  # noqa: BLE001
        return
    if files and mem.get("tails") is not None:   # G56NF_TAILMEAN: every window save the runner wrote before these member files
        tail_new(fam, mode, header, save_root, mem["tails"], mem["tail"])   # (a member's window saves land before its own file)
    added = 0
    for f in files:
        mem["seen"].add(f.name)
        try:
            t = time.time()
            got = shipped_form(fam, mode, header, f, what="averaged")
            if got is None:
                continue
            plan5_note(mem.get("cost"), form_s=time.time() - t)
            mem["sds"].append(got[0])
            if mem.get("tails") is not None:
                mem["groups"].append(tail_group(int(f.stem.split("_")[1]), f, got[0], mem["tails"], mem["tail"]))
            added += 1
            if keep_dir is not None:
                LF.save_atomic(got[0], keep_dir / f"m{int(f.stem.split('_')[1])}.safetensors")
            f.unlink()
        except Exception as e:  # noqa: BLE001
            log(f"WARN member {f.name}: {type(e).__name__}: {str(e)[:160]}")
    if added:
        try:
            if mem.get("tails") is None:   # (PLAN5: mem['form'] = concat while the child trains, plan5_final after; else soup_form())
                ship_mean(fam, mode, header, mem["sds"], out_dir, form=mem.get("form"), cost=mem.get("cost"), gate=mem.get("cost") is None)
            else:
                ship_mean(fam, mode, header, mem["sds"], out_dir, groups=mem["groups"], tail=mem["tail"], form="concat",
                          cost=mem.get("cost"), gate=mem.get("cost") is None)
        except Exception as e:  # noqa: BLE001
            log(f"WARN members: {type(e).__name__}: {str(e)[:160]}; the shipped file stays")


def member_lines(status_path, mem):
    """K members: the runner's one line per finished member (status.json 'member_log') copied once to the driver's private log."""
    try:
        lines = json.loads(Path(status_path).read_text()).get("member_log") or []
    except Exception:  # noqa: BLE001
        return
    for line in lines[mem["logged"]:]:
        log(str(line)[:300])
    mem["logged"] = max(mem["logged"], len(lines))


def usable(out_dir):
    f = Path(out_dir) / "last.safetensors"
    try:
        if not f.is_file() or f.stat().st_size == 0:
            return False
        LF.read_header(f)
        return True
    except Exception:  # noqa: BLE001
        return False


def _kill_group(proc, why):
    if proc is None or proc.poll() is not None:
        return
    log(f"stopping the training process ({why})")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass
    for _ in range(40):
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


def _watchdog(t_kill, out_dir, holder):
    """WATCHDOG_S before the validator's kill: the child group stopped, exit 0 with the complete shipped file."""
    def watch():
        while time.time() < t_kill - WATCHDOG_S:
            time.sleep(min(5.0, max(0.1, t_kill - WATCHDOG_S - time.time())))
            if holder.get("done"):
                return
        if holder.get("done"):
            return
        _kill_group(holder.get("proc"), "watchdog")
        try:  # a save interrupted by the hard stop leaves its temp file: never upload it (only last.safetensors)
            (Path(out_dir) / ".last.safetensors.tmp").unlink()
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            log(f"watchdog: could not remove the partial save: {e}")
        log("watchdog: hard stop before the validator's kill, exit 0" if usable(out_dir) else
            "watchdog: no usable adapter", public=True)
        os._exit(0 if usable(out_dir) else 1)
    threading.Thread(target=watch, name="g56nf-watchdog", daemon=True).start()


def snap_step(path):
    """The 'g56_step' a runner snapshot carries in its safetensors header (None when absent or unreadable)."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__") or {}
        return int(meta["g56_step"]) if "g56_step" in meta else None
    except Exception:  # noqa: BLE001
        return None


def snap_len(path):
    """The 'g56_len' a PLAN5 member file carries in its header - the length it was planned at, which a member the deadline cut short
    and PLAN5 kept did not reach (None when absent: every other file)."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__") or {}
        return int(meta["g56_len"]) if "g56_len" in meta else None
    except Exception:  # noqa: BLE001
        return None


def musubi_step(name, max_steps):
    """The step of a Musubi save from its name (g56-step%08d; the end-of-training save g56.safetensors = max_steps)."""
    m = re.match(r"g56-step(\d+)\.safetensors$", name)
    if m:
        return int(m.group(1))
    return max_steps if name == "g56.safetensors" else None


def lower_rung(res):
    """One resolution rung below the largest of `res` ([] when there is none)."""
    return [r for r in RES_RUNGS if r < max(res)][:1]


def count_ooms(path, pattern):
    try:
        return len(re.findall(pattern, Path(path).read_text(encoding="utf-8", errors="replace")))
    except Exception:  # noqa: BLE001
        return 0


def aitk_child(fam, mode, header, spec, spec_path, env, alog, out_dir, holder, t_train_end, floor):
    """ONE AI Toolkit training child: launched on `spec`, its snapshots shipped as they land, stopped at the training deadline. With
    `floor` (the step of the file an earlier child shipped) a snapshot replaces the shipped file only when its step is above it.
    -> (exit code, its status.json, the step of the last file it shipped (else `floor`), seconds from launch to its first snapshot)."""
    spec_path.write_text(json.dumps(spec, indent=1))
    t_launch = time.time()
    with open(alog, "ab") as lf:
        proc = subprocess.Popen([str(AITK_PY), str(RUNNER), str(spec_path)], cwd=str(AITK_ROOT), stdout=lf,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)
    holder["proc"] = proc
    save_root = Path(spec["save_root"])
    shipped_mtime, last_beat, shipped, t_first, kept = 0.0, time.time(), floor, None, set()
    cap = int(spec.get("max_steps") or 0)
    mem = dict(sds=[], seen=set(), logged=0) if int(spec.get("members") or 1) > 1 or spec.get("plan5") else None   # K members
    tails = {} if spec.get("tail_fracs") else None
    if mem is not None and tails is not None:
        mem.update(tails=tails, groups=[], tail=holder["tail_aitk"])
    if mem is not None and spec.get("plan5"):   # PLAN5: the mean in the concat form while the child trains; the costs the runner reads
        mem.update(form="concat", cost=holder.get("p5_cost"))
    while True:
        rc = proc.poll()
        snaps = sorted(save_root.glob("g56_*.safetensors"), key=lambda p: p.stat().st_mtime) if save_root.is_dir() else []
        if snaps and snaps[-1].stat().st_mtime > shipped_mtime:
            s = snaps[-1]
            m0 = s.stat().st_mtime
            if t_first is None:
                t_first = time.time() - t_launch
            step = snap_step(s)
            if mem is not None and mem["seen"]:   # K members: once a member finished, no snapshot replaces the mean
                log(f"members: {s.name} at step {step} ignored (a finished member is shipped)")
            elif floor is not None and (step is None or step <= floor):
                log(f"relaunch: {s.name} at step {step} not above the shipped step {floor}: the shipped file stays")
            elif ship(fam, mode, header, s, out_dir,
                      k_done=int(mem is None and bool(cap) and step is not None and step >= cap),
                      cost=holder.get("p5_cost") if spec.get("plan5") else None) is not None:
                shipped = step if step is not None else 10 ** 9   # unknown step: never replaced by a relaunch
            shipped_mtime = m0   # a bad snapshot is not retried
        if spec.get("keep_dir"):
            keep_new(fam, mode, header, save_root, Path(spec["keep_dir"]), kept)
        if tails is not None:
            tail_new(fam, mode, header, save_root, tails, holder["tail_aitk"])
            if spec.get("plan5"):   # PLAN5: the slowest load + check of a window save
                plan5_note(holder.get("p5_cost"), form_s=holder["tail_aitk"]["t_cap"])
        if mem is not None:
            members_new(fam, mode, header, save_root, out_dir, mem,
                        Path(spec["members_keep_dir"]) if spec.get("members_keep_dir") else None)
            member_lines(spec["status"], mem)
        if rc is not None:
            break
        if time.time() > holder.get("t_child_kill", t_train_end + KILL_GRACE_S):   # (PLAN5: the plan's end, not a grace after it)
            _kill_group(proc, "training deadline")
            continue
        if time.time() - last_beat > 300:
            last_beat = time.time()
            log(f"training ({(time.time() - t_launch) / 60:.0f} min)", public=True)
        time.sleep(holder.get("poll_s", 5.0))
    holder["t_child"] = (t_launch, time.time())
    try:  # leftovers of the child's process group (dataloader workers of an os._exit-ed runner)
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001  (no such group: nothing left)
        pass
    status = {}
    try:
        status = json.loads(Path(spec["status"]).read_text())
    except Exception:  # noqa: BLE001
        pass
    log(f"training process exit {rc}; status {json.dumps(status)[:300]}")
    if spec.get("edit_mask_dir"):
        edit_mask_line(status.get("mask") or {})
    if spec.get("ts_strata"):
        ts_strata_line(status.get("strata") or {})
    if spec.get("tstrat"):
        tstrat_line(status.get("tstrat"))
    if spec.get("sdpa_backend"):
        sdpa_line(status.get("sdpa"))
    if spec.get("freeze_blocks"):
        freeze_line(status.get("freeze"), status.get("step_s"))
    if spec.get("profile"):         # G56NF_PROFILE (a measuring tool): what the runner's profiler read
        profile_line(status.get("profile"))
    if spec.get("lora_noop") or spec.get("bnb_nosync"):
        speed_lines(spec, status)
    if spec.get("gc_step_fence"):
        gc_step_line(status.get("gc_step"))
    if spec.get("tf32"):
        t32 = status.get("tf32")
        log("tf32: no reading (the runner wrote none)" if not isinstance(t32, dict) or "matmul" not in t32 else
            f"tf32: matmul {'on' if t32.get('matmul') else 'off'}, cudnn {'on' if t32.get('cudnn') else 'off'}")
    if spec.get("cfg_watch"):       # a 'configuration only' knob is on: what the run really trained on, from its last status
        config_effect_line(status.get("cfg") or {})
    if mem is not None:
        holder["members_done"] = len(mem["sds"])
        if spec.get("plan5"):   # PLAN5: the members whose concat mean is shipped (plan5_final)
            holder["p5_mem"] = mem
    if tails is not None:   # G56NF_TAILMEAN: a single run's window saves (tail_ship_aitk); the members' groups (tail_final)
        holder["tails"] = tails
        if mem is not None:
            holder["tail_mem"] = mem
    return rc, status, shipped, t_first


def relaunch_aitk(fam, mode, header, spec, work, env, alog, out_dir, holder, t_train_end, rc, status, shipped, t_first):
    """L2: the one relaunch of an AI Toolkit child that ended before its own stop (see RES_RUNGS)."""
    ooms = count_ooms(alog, r"OOM during training step")
    need = (t_first if t_first is not None else RELAUNCH_LOAD_S) + RELAUNCH_MARGIN_S
    left = t_train_end - POST_S - time.time()
    why = f"exit {rc}, phase {status.get('phase')}, OOM lines {ooms}"
    if left < need:
        log(f"relaunch skipped ({why}): {left:.0f} s left < {need:.0f} s; the shipped file stays")
        return
    s2 = json.loads(json.dumps(spec))
    plain = s2.pop("config_plain", None)
    if plain is not None and not ooms and shipped is None:   # ... which are the suspects
        s2["config"] = plain
        s2.pop("cfg_watch", None)
        log("relaunch: no snapshot was shipped with the configuration knobs on and no OOM line: the job without them")
    proc = s2["config"]["config"]["process"][0]
    proc["train"]["gradient_checkpointing"] = True
    lower = lower_rung(proc["datasets"][0].get("resolution") or RESOLUTION) if ooms else []
    if lower:
        for d in proc["datasets"]:   # G56NF_TWO_ENTRY: both entries
            d["resolution"] = lower
    if ooms and proc["train"].get("batch_size", 1) != 1:   # G56NF_BATCH: one item per forward after an OOM
        proc["train"]["batch_size"] = 1
        log("relaunch: batch size 1 after the OOM")
    proc["training_folder"] = str(work / "run2")
    s2["save_root"] = str(work / "run2" / "g56")
    s2["status"] = str(work / "status2.json")
    for k in ("test_ballast_gb", "test_crash_at_step", "keep_steps", "keep_dir"):   # keep: a relaunch is another trajectory
        s2.pop(k, None)
    if s2.get("edit_mask_dir") and not ooms and shipped is None:
        s2.pop("edit_mask_dir")
        log("relaunch: no snapshot was shipped with the edit mask on and no OOM line: the unweighted loss")
    s2.pop("profile", None)   # G56NF_PROFILE: the first child's steps only
    if s2.pop("gc_step_fence", None):
        log("gc step fence: off in the relaunch (gradient checkpointing on for the whole child)")
    if s2.pop("tail_fracs", None):
        holder.pop("tail_aitk", None)
        log("tail mean: off - the AI Toolkit child ended early (L2): the relaunch is another trajectory")
    if s2.get("plan5") and not ooms and shipped is None:   # PLAN5: a suspect too - the relaunch plans with member_plan, the same deadline
        s2.pop("plan5")
        s2["members"] = members_max(fam, mode)
        log("relaunch: no snapshot was shipped with PLAN5 on and no OOM line: member_plan's members (the deadline stays the plan's)")
    elif s2.get("plan5"):   # PLAN5 stays: one member of the plan's file again (the tail mean went: one save per member)
        s2["plan5"]["saves"] = 1
    if s2.get("ts_strata") and not ooms and shipped is None:
        s2.pop("ts_strata")
        log("relaunch: no snapshot was shipped with the timestep strata on and no OOM line: AI Toolkit's own draw")
    if s2.get("tstrat") and not ooms and shipped is None:
        s2.pop("tstrat")
        log("relaunch: no snapshot was shipped with the low-discrepancy timesteps on and no OOM line: AI Toolkit's own draw")
    if s2.get("sdpa_backend") and not ooms and shipped is None:
        s2.pop("sdpa_backend")
        log("relaunch: no snapshot was shipped with the cuDNN attention first and no OOM line: torch's own order")
    if s2.get("freeze_blocks") and not ooms and shipped is None:
        s2.pop("freeze_blocks")
        log("relaunch: no snapshot was shipped with the frozen blocks and no OOM line: every block trained")
    for k, what in (("lora_noop", "the LoRA no-op path"), ("bnb_nosync", "the bitsandbytes synchronizations skipped"), ("tf32", "TF32")):
        if s2.get(k) and not ooms and shipped is None:
            s2.pop(k)
            log(f"relaunch: no snapshot was shipped with {what} and no OOM line: AI Toolkit's own code")
    s2["attempt"] = 2
    log(f"relaunch 1/1 ({why}): gradient checkpointing on, resolution {proc['datasets'][0]['resolution']}, shipped step {shipped}")
    aitk_child(fam, mode, header, s2, work / "spec2.json", env, alog, out_dir, holder, t_train_end, shipped)


#
#
#
#
#
#
#
#
#
#
MUSUBI_RANK = 32
MUSUBI_RANK_RECIPE = 32    # the official recipe's --network_dim / --network_alpha
MUSUBI_RANK_MAX = 256


def musubi_rank():
    """MUSUBI_RANK or G56NF_MUSUBI_RANK=r -> the rank (= the alpha) of the Musubi network's linear modules, a whole number in
    1..MUSUBI_RANK_MAX. A value that is not one is ignored (WARN): the table's value."""
    raw = os.environ.get("G56NF_MUSUBI_RANK")
    if not raw:
        return MUSUBI_RANK
    try:
        r = int(raw)
        if not 1 <= r <= MUSUBI_RANK_MAX:
            raise ValueError(raw)
        return r
    except Exception:  # noqa: BLE001
        log(f"WARN G56NF_MUSUBI_RANK {raw[:20]!r} ignored (a whole number in 1..{MUSUBI_RANK_MAX}): {MUSUBI_RANK}")
        return MUSUBI_RANK


def musubi_band():
    """MUSUBI_TBAND or G56NF_MUSUBI_TBAND=lo,hi -> None (the whole range, also when given as 0,1000: nothing is added to the
    training command) | (lo, hi), whole timesteps with 0 <= lo < hi <= 1000. A value that is not such a pair is ignored."""
    try:
        v = knob("MUSUBI_TBAND", MUSUBI_TBAND, lambda s: tuple(int(x) for x in s.split(",")))
        if v is None:
            return None
        lo, hi = (int(x) for x in v)
        if not 0 <= lo < hi <= 1000:
            raise ValueError(v)
        return None if (lo, hi) == (0, 1000) else (lo, hi)
    except Exception as e:  # noqa: BLE001
        log(f"WARN G56NF_MUSUBI_TBAND ignored ({type(e).__name__}): the whole range")
        return None


def musubi_notext():
    """MUSUBI_NOTEXT or G56NF_MUSUBI_NOTEXT=f -> None (no no-text block: the official dataset config) | (text repeats, no-text
    repeats): the whole num_repeats, at most 8 in all, whose no-text share of the steps is nearest to f (0.5 -> 1 : 1,
    0.25 -> 3 : 1, 1.0 -> 0 : 1 = the no-text block alone). f <= 0 or not a number -> None; a share too small for one repeat
    in eight is ignored."""
    try:
        f = float(knob("MUSUBI_NOTEXT", MUSUBI_NOTEXT))
        if not 0.0 < f < float("inf"):
            return None
        share = Fraction(min(1.0, f)).limit_denominator(8)
        if share.numerator == 0:
            raise ValueError(f)
        return share.denominator - share.numerator, share.numerator
    except Exception as e:  # noqa: BLE001
        log(f"WARN G56NF_MUSUBI_NOTEXT ignored ({type(e).__name__}): no no-text block")
        return None


def musubi_speed():
    """MUSUBI_CPU_THREADS / MUSUBI_GC / MUSUBI_TIMING or their G56NF_ knobs -> (threads, gradient checkpointing, timing): threads a
    whole number 1..64, checkpointing a bool (1 / 0, true / false, on / off, yes / no), timing '' | '1' | 'profile' (0 / off = '').
    Any other value is ignored (one WARN line): the table's value, and a table value that is none of these: the official 1 / on / ''."""
    threads = MUSUBI_CPU_THREADS if type(MUSUBI_CPU_THREADS) is int and 1 <= MUSUBI_CPU_THREADS <= 64 else 1
    gc = MUSUBI_GC if type(MUSUBI_GC) is bool else True
    timing = MUSUBI_TIMING if MUSUBI_TIMING in ("", "1", "profile") else ""
    v = (os.environ.get("G56NF_MUSUBI_CPU_THREADS") or "").strip()
    if v:
        if re.fullmatch(r"[0-9]{1,2}", v) and 1 <= int(v) <= 64:
            threads = int(v)
        else:
            log(f"WARN G56NF_MUSUBI_CPU_THREADS {v[:20]!r} ignored (a whole number 1..64): {threads}")
    v = (os.environ.get("G56NF_MUSUBI_GC") or "").strip().lower()
    if v:
        if v in ("1", "true", "on", "yes", "0", "false", "off", "no"):
            gc = v in ("1", "true", "on", "yes")
        else:
            log(f"WARN G56NF_MUSUBI_GC {v[:20]!r} ignored (1 or 0): gradient checkpointing {'on' if gc else 'off'}")
    v = (os.environ.get("G56NF_MUSUBI_TIMING") or "").strip().lower()
    if v:
        if v in ("1", "profile", "0", "off"):
            timing = "" if v in ("0", "off") else v
        else:
            log(f"WARN G56NF_MUSUBI_TIMING {v[:20]!r} ignored (1 or profile): timing {timing or 'off'}")
    return threads, gc, timing


def prepare_notext(data_dir, out):
    """The no-text copy of a prepared edit set, as the evaluator writes its own (diffusion_objective._write_dataset, blank=True):
    every target image under its name - Musubi pairs a target with its source by that name, so the block shares the control
    folder - and an EMPTY caption file. -> (the folder, images)."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(Path(data_dir).iterdir()):
        if f.suffix.lower() in IMAGE_EXT:
            shutil.copyfile(f, out / f.name)
            (out / f"{f.stem}.txt").write_text("", encoding="utf-8")
            n += 1
    if n == 0:
        raise ValueError("no target images")
    return out, n


def musubi_toml(data_dir, ctrl_dir, cache_dir, path, res=1024, blank=None):
    """The official miner entrypoint's Musubi dataset config (prepare_hidream_edit_config), with our folders (res: the L2 relaunch
    after a CUDA OOM lowers it one rung; the first run is the official 1024). blank = (the no-text folder, its cache folder, text
    repeats, no-text repeats) adds the no-text block (MUSUBI_NOTEXT) - Musubi takes a block's items from the cache files of its
    own cache_directory and refuses two blocks on one folder; None = the official config, byte for byte."""
    def block(images, cache, repeats):
        return ["[[datasets]]", f'image_directory = "{images}"', f'control_directory = "{ctrl_dir}"',
                f'cache_directory = "{cache}"', f"num_repeats = {repeats}", ""]
    r_text = blank[2] if blank else 1
    blocks = (block(data_dir, cache_dir, r_text) if r_text else []) + (block(blank[0], blank[1], blank[3]) if blank else [])
    path.write_text("\n".join(["[general]", f"resolution = [{res}, {res}]", 'caption_extension = ".txt"', "batch_size = 1",
                               "enable_bucket = true", "bucket_no_upscale = false", "", *blocks]), encoding="utf-8")
    return path


def _child(cmd, cwd, env, logf):
    with open(logf, "ab") as lf:
        return subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)


def musubi_cache(toml, dit, env, mlog, holder, t_train_end):
    """Musubi's two cache scripts (pixels, text encoder outputs) as the official entrypoint runs them. -> None | the failure."""
    common = ["--dataset_config", toml, "--batch_size", "1"]
    for what, cmd in (("cache pixels", [MUSUBI_PY, "src/musubi_tuner/hidream_o1_cache_pixel.py", *common]),
                      ("cache text", [MUSUBI_PY, "src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py", *common, "--dit", dit,
                                      "--model_type", "full"])):
        proc = _child(cmd, MUSUBI_ROOT, env, mlog)
        holder["proc"] = proc
        while proc.poll() is None:
            if time.time() > t_train_end:
                _kill_group(proc, f"training deadline during {what}")
            time.sleep(1.0)
        if proc.returncode != 0:
            return f"musubi {what}: exit {proc.returncode}"
    return None


def musubi_train_cmd(dit, toml, out, every, steps, band=None, masked=False, threads=1, gc=True, rank=MUSUBI_RANK_RECIPE):
    """
    
    
    """
    script = MUSUBI_RUNNER if masked else "src/musubi_tuner/hidream_o1_train_network.py"
    cmd = [MUSUBI_ACCEL, "launch", "--num_cpu_threads_per_process", str(threads), "--mixed_precision", "bf16",
           script, "--dit", dit, "--dataset_config", toml, "--model_type", "full",
           "--task", "i2i", "--sdpa", "--mixed_precision", "bf16", "--timestep_sampling", "uniform", "--weighting_scheme", "none",
           "--noise_scale_start", "8.0", "--noise_scale_end", "8.0", "--noise_clip_std", "0.0", "--optimizer_type", "adamw8bit",
           "--learning_rate", str(knob("LR", MUSUBI_LR)), *(["--gradient_checkpointing"] if gc else []), "--network_module", "networks.lora_hidream_o1",
           "--network_dim", str(rank), "--network_alpha", str(rank), "--network_args", "conv_dim=4", "conv_alpha=1",
           "--save_every_n_steps", str(every), "--save_last_n_steps", str(3 * every), "--seed", str(knob("SEED", SEED, int)),
           "--output_dir", out, "--output_name", "g56", "--max_train_steps", str(steps)]
    return cmd + (["--min_timestep", str(band[0]), "--max_timestep", str(band[1])] if band else [])


def musubi_mask_env(env, mask, work, cache_dir, blank, tag=""):
    """The environment of a masked training child (g56_musubi_runner.py; EDIT_MASK): the masks of the training pairs, the cache
    folders of the dataset config's blocks (the runner recognizes a batch's pair by its cached target) and the file its counts
    go to. mask None -> env as it is (Musubi's own script reads none of it)."""
    if not mask:
        return env
    dirs = ([str(cache_dir)] if not blank or blank[2] else []) + ([str(blank[1])] if blank else [])
    return dict(env, G56_EDIT_MASK_DIR=str(mask), G56_EDIT_CACHE_DIRS=os.pathsep.join(dirs),
                G56_EDIT_MASK_STATUS=str(work / f"musubi_mask{tag}.json"))


def musubi_plin_env(env, plin, work, tag=""):
    """G56NF_MUSUBI_PATCHLINEAR: the runner's switch and the file of its report (plin False -> env as it is)"""
    return dict(env, G56_PATCHLINEAR="1", G56_PATCHLINEAR_STATUS=str(work / f"musubi_plin{tag}.json")) if plin else env


def musubi_plin_line(work, tag=""):
    """G56NF_MUSUBI_PATCHLINEAR: what the runner replaced and how often the linear forms ran -> one private line"""
    try:
        m = json.loads((work / f"musubi_plin{tag}.json").read_text())
    except Exception as e:  # noqa: BLE001
        log(f"patchlinear: no report from the runner ({type(e).__name__})")
        return
    log(f"patchlinear: replaced {len(m.get('replaced') or [])} module(s) {m.get('replaced')}, calls {m.get('calls')} over {m.get('steps')} "
        f"steps; the conv's backend before: {m.get('backend')}; cudnn {m.get('cudnn')}; check max relative difference "
        f"{float(m.get('check') or 0.0):.1e}" + (f"; skipped {m['skipped']}" if m.get("skipped") else "")
        + (f"; OFF: {m['off']}" if m.get("off") else ""))


def musubi_members():
    """MUSUBI_MEMBERS or G56NF_MUSUBI_MEMBERS: K_max of the Musubi path (1 = Musubi's own single run); a value that does not parse ->
    one WARN, 1"""
    try:
        return max(1, knob("MUSUBI_MEMBERS", MUSUBI_MEMBERS, int))
    except Exception as e:  # noqa: BLE001
        log(f"WARN G56NF_MUSUBI_MEMBERS ignored ({type(e).__name__}): 1 member")
        return 1


def musubi_mem_cmd(cmd, mem):
    """G56NF_MUSUBI_MEMBERS: the training command run by our runner (its member rules) with --max_train_steps = K_max x the member
    length (mem None -> the command as it is)"""
    if mem is None:
        return cmd
    cmd = [MUSUBI_RUNNER if str(x) == "src/musubi_tuner/hidream_o1_train_network.py" else x for x in cmd]
    cmd[cmd.index("--max_train_steps") + 1] = str(mem["k_max"] * mem["L"])
    if mem.get("plan5"):   # PLAN5: the members may be stretched (the runner ends the training itself)
        cmd[cmd.index("--max_train_steps") + 1] = str(mem["k_max"] * int(mem["plan5"]["hi"] * mem["L"]) + 1)
    return cmd


def musubi_mem_env(env, mem):
    """G56NF_MUSUBI_MEMBERS: the runner's member variables (mem None -> env as it is)"""
    if mem is None:
        return env
    env = dict(env, G56_MEMBERS=str(mem["k_max"]), G56_MEMBER_LEN=str(mem["L"]), G56_MEMBERS_DEADLINE=repr(float(mem["deadline"])),
               G56_MEMBERS_SEED=str(mem["seed"]), G56_MEMBERS_STATUS=str(mem["status"]), G56_MEMBERS_DIR=str(mem["dir"]))
    return dict(env, G56_PLAN5=json.dumps(mem["plan5"])) if mem.get("plan5") else env   # PLAN5: the plan for the runner's member rules


def musubi_ship_mean(fam, mode, header, sds, out_dir, cost=None):
    """G56NF_MUSUBI_MEMBERS: the upload file = the exact mean of the finished members' deltas (one: that member; more:
    g56_lorafmt.concat_mean - every factor as trained, the 1 / K from the rank, the alphas the members') times export_scale(K); shipped
    only when concat_mean reports no problem, the loader mirror passes with full coverage and the file is finite. -> True when shipped;
    whatever fails, the shipped file stays (logged)."""
    k, t = len(sds), time.time()
    s = export_scale(fam, mode, k)
    try:
        if cost is not None and k > 1 and t + PLAN5_POST_X * cost["merge_s"] * k > cost["t_stop"]:   # PLAN5: never a write into the watchdog's stop
            raise TimeoutError(f"{cost['t_stop'] - t:.0f} s left before the watchdog's stop < {PLAN5_POST_X * cost['merge_s'] * k:.0f} s")
        sd, rep = (dict(sds[0]), dict(problems=[], ranks={})) if k == 1 else LF.concat_mean(sds)
        if s != 1.0:
            sd = LF.scale_delta(sd, s)
        sd = group_scaled(fam, mode, header, sd, k, f"the mean of {k} member(s), Musubi")[0]   # EXPORT_GROUP_SCALE: after the export scale
        chk = LF.check(fam, mode, header, sd, expect_full=True)
        bad = rep["problems"][:1] or ([] if chk["ok"] else ["the loader check fails"]) or ([] if finite(sd) else ["non-finite values"])
        if bad:
            raise ValueError(bad[0])
    except Exception as e:  # noqa: BLE001
        log(f"WARN members: the mean of {k} members is not shipped ({type(e).__name__}: {str(e)[:160]}); the shipped file stays")
        return False
    LF.save_atomic(sd, out_dir / "last.safetensors")
    plan5_note(cost, merge_s=(time.time() - t) / k)
    log(f"merged {k} members (Musubi): {time.time() - t:.1f} s, ranks {rep.get('ranks')}, scale {s:g}, form concat; shipped members "
        f"1-{k}: {len(sd)} tensors, coverage {chk['coverage']:.3f}")
    return True


def musubi_members_new(fam, mode, header, out_dir, mem):
    """G56NF_MUSUBI_MEMBERS, one poll of musubi_child: every member file of the runner not handled yet (member_<k>.safetensors, in
    order) -> its shipped form (Musubi's keys as saved; one that does not pass is left out of the mean) and the upload file rewritten
    with the exact mean of the members so far (musubi_ship_mean); mem['keep'] (G56NF_KEEP_MEMBERS) -> each member also goes there,
    in that form, as m<k>.safetensors (the AI Toolkit path's name), before the runner's file is removed; the runner's new member lines
    -> the private log. Never raises."""
    try:
        files = sorted(f for f in mem["dir"].glob("member_*.safetensors") if f.name not in mem["seen"]) if mem["dir"].is_dir() else []
    except Exception:  # noqa: BLE001
        files = []
    added = 0
    for f in files:
        mem["seen"].add(f.name)
        try:
            t = time.time()
            got = shipped_form(fam, mode, header, f, normalize=_as_saved, what="averaged")
            if got is not None:
                plan5_note(mem.get("cost"), form_s=time.time() - t)
                mem["sds"].append(got[0])
                added += 1
                if mem.get("keep") is not None:
                    LF.save_atomic(got[0], Path(mem["keep"]) / f"m{int(f.stem.split('_')[1])}.safetensors")
            f.unlink()
        except Exception as e:  # noqa: BLE001
            log(f"WARN member {f.name}: {type(e).__name__}: {str(e)[:160]}")
    if added:
        try:
            musubi_ship_mean(fam, mode, header, mem["sds"], out_dir, cost=mem.get("cost"))
        except Exception as e:  # noqa: BLE001
            log(f"WARN members: {type(e).__name__}: {str(e)[:160]}; the shipped file stays")
    try:
        lines = json.loads(Path(mem["status"]).read_text()).get("member_log") or []
    except Exception:  # noqa: BLE001
        lines = []
    for line in lines[mem["logged"]:]:
        log(str(line)[:300])
    mem["logged"] = max(mem["logged"], len(lines))


def musubi_temb_cmd(cmd, temb):
    """G56NF_MUSUBI_TEMB: the training command with our network module in place of Musubi's (temb False -> the command as it is)"""
    if not temb:
        return cmd
    cmd = list(cmd)
    cmd[cmd.index("--network_module") + 1] = TEMB_MODULE
    return cmd


def musubi_temb_env(env, temb):
    """G56NF_MUSUBI_TEMB: the child's environment with MUSUBI_NET_DIR first on its PYTHONPATH (temb False -> env as it is)"""
    if not temb:
        return env
    return dict(env, PYTHONPATH=os.pathsep.join([str(MUSUBI_NET_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])))


def musubi_temb_line(out_dir):
    """G56NF_MUSUBI_TEMB: the TimestepEmbedder modules of the shipped file (their keys, ranks) -> one private line"""
    try:
        keys = LF.read_header(Path(out_dir) / "last.safetensors")
        mods = sorted({k.split(".")[0] for k in keys if "t_embedder1" in k})
        ranks = sorted({int(keys[m + ".lora_down.weight"][1][0]) for m in mods if m + ".lora_down.weight" in keys})
        log(f"temb: the shipped file holds {len(mods)} TimestepEmbedder module(s) {mods}" + (f", rank {ranks}" if ranks else ""))
    except Exception as e:  # noqa: BLE001
        log(f"temb: the shipped file not read ({type(e).__name__}: {str(e)[:120]})")


def musubi_rank_line(out_dir, rank):
    """G56NF_MUSUBI_RANK: the (rank, alpha) pairs the shipped file really holds - its linear modules, the conv LoRA apart - -> one
    private line (the knob's effect, read from the file that leaves the container, not from the command)."""
    try:
        sd = LF.load_file_tensors(Path(out_dir) / "last.safetensors")
        lin, cv = {}, {}
        for k, v in sd.items():
            if k.endswith(".lora_down.weight"):
                a = sd.get(k[:-len("lora_down.weight")] + "alpha")
                key = (int(v.shape[0]), None if a is None else float(a))
                conv = v.ndim > 2 and any(int(x) != 1 for x in v.shape[2:])
                (cv if conv else lin)[key] = (cv if conv else lin).get(key, 0) + 1
        log(f"musubi rank: asked {rank} / {rank}; the shipped file holds {sum(lin.values())} linear modules at (rank, alpha) {sorted(lin)} "
            f"and {sum(cv.values())} conv at {sorted(cv)}")
    except Exception as e:  # noqa: BLE001
        log(f"musubi rank: the shipped file not read ({type(e).__name__}: {str(e)[:120]})")


def musubi_mask_line(work, tag=""):
    try:
        edit_mask_line(json.loads((work / f"musubi_mask{tag}.json").read_text()))
    except Exception as e:  # noqa: BLE001
        edit_mask_line(dict(off=f"no count from the runner ({type(e).__name__})"))


def musubi_child(fam, mode, header, cmd, out, env, mlog, out_dir, holder, t_train_end, floor, steps):
    """ONE Musubi training child: each newest save shipped once its size holds still for one poll (Musubi writes in place) or the
    child ended; stopped at the training deadline. With `floor` a save replaces the shipped file only when its step is above it.
    -> (exit code, stopped by our deadline, the step of the last save shipped (else `floor`), seconds to its first save)."""
    out.mkdir(parents=True, exist_ok=True)
    t_launch = time.time()
    proc = _child(cmd, MUSUBI_ROOT, env, mlog)
    holder["proc"] = proc
    seen, tried, last_beat, shipped, t_first, killed, tseen = {}, {}, time.time(), floor, None, False, {}
    keep = lambda f, sd: {k: v for k, v in sd.items()}   # noqa: E731  (Musubi saves are already in the evaluator's layout)
    mem = holder.get("musubi_mem")
    while True:
        rc = proc.poll()
        if mem is not None:   # first: once a member finished, no step save of this poll may replace the mean
            musubi_members_new(fam, mode, header, out_dir, mem)
        files = sorted(out.glob("g56*.safetensors"), key=lambda p: p.stat().st_mtime) if out.is_dir() else []
        for f in files[-1:]:   # the newest save, once its size held still for one poll (Musubi writes in place) or the child ended
            size = f.stat().st_size
            if tried.get(f.name) != (size, f.stat().st_mtime) and (seen.get(f.name) == size or rc is not None):
                if t_first is None:
                    t_first = time.time() - t_launch
                step = musubi_step(f.name, steps)
                if mem is not None and mem["sds"]:   # members: a step save after a finished member is never shipped
                    log(f"members: {f.name} at step {step} ignored (a finished member is shipped)")
                elif floor is not None and (step is None or step <= floor):
                    log(f"relaunch: {f.name} at step {step} not above the shipped step {floor}: the shipped file stays")
                elif ship(fam, mode, header, f, out_dir, normalize=keep,
                          k_done=int(step is not None and step >= steps)) is not None:   # the save at max_train_steps = a finished run
                    shipped = step if step is not None else 10 ** 9
                    holder["musubi_last"] = (shipped, f)   # G56NF_TAILMEAN: the last member of the tail mean
                tried[f.name] = (size, f.stat().st_mtime)
            seen[f.name] = size
        if holder.get("tail"):
            tail_poll(fam, mode, header, files, steps, holder["tail"], tseen, rc is not None)
        if rc is not None:
            break
        if time.time() > holder.get("t_child_kill", t_train_end + KILL_GRACE_S):   # (PLAN5: the plan's end, not a grace after it)
            _kill_group(proc, "training deadline")
            killed = True
            continue
        if time.time() - last_beat > 300:
            last_beat = time.time()
            log(f"training ({(time.time() - t_launch) / 60:.0f} min)", public=True)
        time.sleep(holder.get("poll_s", 5.0))
    holder["t_child"] = (t_launch, time.time())
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    log(f"musubi training exit {rc}; saves {sorted(tried)[-3:]}")
    if mem is not None:   # G56NF_MUSUBI_MEMBERS: the last member files and lines, and how many finished
        musubi_members_new(fam, mode, header, out_dir, mem)
        holder["members_done"] = len(mem["sds"])
    return rc, killed, shipped, t_first


def relaunch_musubi(fam, mode, header, work, data_dir, ctrl, dit, env, mlog, out_dir, holder, t_train_end, toml, rc, shipped, t_first,
                    cache_s, every, steps, band=None, blank=None, mask=None, threads=1, gc=True, plin=False, temb=False,
                    rank=MUSUBI_RANK_RECIPE):
    """
    
    
    
    
    
    
    
    
    
    """
    ooms = count_ooms(mlog, r"CUDA out of memory|OutOfMemoryError")
    speed = threads != 1 or not gc
    lower = lower_rung([1024]) if ooms and gc else []
    plain = bool(band or blank or mask or speed or plin) and not ooms and shipped is None
    plain = plain or bool(temb and not ooms and shipped is None)
    plain = plain or bool(rank != MUSUBI_RANK_RECIPE and not ooms and shipped is None)
    need = (cache_s if lower or plain else 0.0) + (t_first if t_first is not None else RELAUNCH_LOAD_S) + RELAUNCH_MARGIN_S
    left = t_train_end - POST_S - time.time()
    why = f"exit {rc}, OOM lines {ooms}"
    if left < need:
        log(f"relaunch skipped ({why}): {left:.0f} s left < {need:.0f} s; the shipped file stays")
        return
    if plain:
        log("relaunch: no save was shipped with the Musubi research knobs on and no OOM line: the official recipe")
        band = blank = mask = None
        threads, plin = 1, False
        temb = False
        rank = MUSUBI_RANK_RECIPE
    toml2, cache2, blank2 = toml, work / "musubi_cache", blank
    if lower or plain:
        cache2 = work / "musubi_cache2"
        blank2 = (blank[0], work / "musubi_cache2_nt", *blank[2:]) if blank else None
        toml2 = musubi_toml(data_dir, ctrl[0], cache2, work / "musubi_dataset2.toml", res=lower[0] if lower else 1024, blank=blank2)
    log(f"relaunch 1/1 ({why}): resolution {lower[0] if lower else 1024}, shipped step {shipped}"
        + ("" if gc else "; gradient checkpointing back on (the first child ran without it)"))
    failed = musubi_cache(toml2, dit, env, mlog, holder, t_train_end) if lower or plain else None
    if failed:
        log(f"relaunch: {failed}; the shipped file stays")
        return
    musubi_child(fam, mode, header, musubi_temb_cmd(
                     musubi_train_cmd(dit, toml2, work / "musubi_out2", every, steps, band, bool(mask) or plin, threads, rank=rank), temb),
                 work / "musubi_out2", musubi_temb_env(musubi_plin_env(musubi_mask_env(env, mask, work, cache2, blank2, "2"), plin, work, "2"),
                                                       temb), mlog,
                 out_dir, holder, t_train_end, shipped, steps)
    if temb:
        musubi_temb_line(out_dir)
    if mask:
        musubi_mask_line(work, "2")
    if plin:
        musubi_plin_line(work, "2")


def run_musubi(a, fam, mode, header, work, out_dir, holder, t_train_end):
    """
    
    
    
    
    
    
    """
    def stop(msg):
        log(msg)
        holder["done"] = True
        log("done", public=True)
        raise SystemExit(0)
    if not MUSUBI_PY.exists() or not (MUSUBI_ROOT / "src" / "musubi_tuner").is_dir():
        stop(f"WARN no Musubi environment at {MUSUBI_PY} / {MUSUBI_ROOT}: the zero LoRA stays")
    dit = CACHE / LF.HIDREAM_EDIT_CKPT
    if not dit.is_file():
        stop(f"WARN no HiDream-O1 checkpoint at {dit}: the zero LoRA stays")
    data_dir, ctrl, n = prepare_edit(dataset_zip(a), work / "data", work / "extract")
    log(f"data: {n} edit rows")
    band, repeats, blank = musubi_band(), musubi_notext(), None
    threads, gc, timing = musubi_speed()
    plin = knob("MUSUBI_PATCHLINEAR", MUSUBI_PATCHLINEAR, bool)
    temb = knob("MUSUBI_TEMB", MUSUBI_TEMB, bool)
    rank = musubi_rank()
    km = musubi_members()
    if repeats:
        try:
            blank = (prepare_notext(data_dir, work / "data" / "no_text")[0], work / "musubi_cache_nt", *repeats)
        except Exception as e:
            log(f"WARN no-text block not prepared ({type(e).__name__}): the official dataset config")
    mask = edit_mask_plan(fam, mode, data_dir, ctrl).get("edit_mask_dir")
    toml = musubi_toml(data_dir, ctrl[0], work / "musubi_cache", work / "musubi_dataset.toml", blank=blank)
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_HOME=str(work / "hf"), TOKENIZERS_PARALLELISM="false")
    for k in [k for k in env if k.startswith("G56CT_")]:
        env.pop(k)
    staged = staged_hub_snapshot("HiDream-ai/HiDream-O1-Image", ("config.json",))
    if staged:
        env["HF_HUB_CACHE"] = env["HUGGINGFACE_HUB_CACHE"] = str(STAGED_HUB)
    log(f"runtime assets: {'validator-staged' if staged else 'none staged (Musubi needs the HiDream-O1 processor / config)'}")
    mlog = work / "musubi.log"
    t_cache = time.time()
    failed = musubi_cache(toml, dit, env, mlog, holder, t_train_end)
    if failed and blank:
        log(f"WARN {failed} with the no-text block: back to the official dataset config")
        blank = None
        toml = musubi_toml(data_dir, ctrl[0], work / "musubi_cache", work / "musubi_dataset.toml")
        failed = musubi_cache(toml, dit, env, mlog, holder, t_train_end)
    if failed:
        stop(f"WARN {failed}: the zero LoRA stays")
    cache_s = time.time() - t_cache   # both blocks when the no-text block is on: what a re-cache of the L2 relaunch must fit
    speed = threads != 1 or not gc or bool(timing)
    if band or blank or mask or speed or plin:
        log(f"musubi knobs: timestep band {'%d-%d' % band if band else 'whole range'}; no-text block "
            f"{'text x %d / empty x %d' % blank[2:] if blank else 'none'}; cache {cache_s:.0f} s; edit mask {'on' if mask else 'off'}"
            + (f"; cpu threads {threads}; gradient checkpointing {'on' if gc else 'off'}; timing {timing or 'off'}" if speed else "")
            + ("; patch embed as linear on" if plin else ""))
    if temb:
        log(f"musubi knobs: TimestepEmbedder LoRA on (--network_module {TEMB_MODULE} from {MUSUBI_NET_DIR})")
    if rank != MUSUBI_RANK_RECIPE:
        log(f"musubi knobs: network rank {rank} / alpha {rank} (the official recipe: {MUSUBI_RANK_RECIPE} / {MUSUBI_RANK_RECIPE}; the conv LoRA stays 4 / 1)")
    tenv = dict(env, HIDREAM_TIMING="1", **({"HIDREAM_PROFILE": "1"} if timing == "profile" else {})) if timing else env
    every = max(1, knob("SAVE_EVERY", MUSUBI_SAVE_EVERY, int))
    steps = knob("STEPS", MUSUBI_STEPS, int) or 10 ** 6
    if os.environ.get("G56NF_KEEP_STEPS"):
        log("WARN G56NF_KEEP_STEPS: not available on the Musubi path, ignored")
    if os.environ.get("G56NF_MEMBERS"):
        log("WARN G56NF_MEMBERS: the Musubi path trains one member, ignored")
    if os.environ.get("G56NF_FILL_RHO"):
        log("WARN G56NF_FILL_RHO: the Musubi path trains one member, ignored")
    if os.environ.get("G56NF_PROFILE"):
        log("WARN G56NF_PROFILE: the AI Toolkit runner's tool; on the Musubi path G56NF_MUSUBI_TIMING=profile is Musubi's own, ignored")
    for k in CFG_KNOBS + ("TS_STRATA",):
        if os.environ.get("G56NF_" + k):
            log(f"WARN G56NF_{k}: a knob of the AI Toolkit job, not available on the Musubi path, ignored")
    if os.environ.get("G56NF_TSTRAT"):
        log("WARN G56NF_TSTRAT: a knob of the AI Toolkit job, not available on the Musubi path, ignored")
    if os.environ.get("G56NF_SDPA_BACKEND"):
        log("WARN G56NF_SDPA_BACKEND: a knob of the AI Toolkit runner, ignored")
    for k in ("LORA_NOOP", "BNB_NOSYNC"):
        if os.environ.get("G56NF_" + k):
            log(f"WARN G56NF_{k}: a knob of the AI Toolkit runner (Musubi computes its own LoRA and optimizer step), ignored")
    if os.environ.get("G56NF_GC_STEP_FENCE"):
        log("WARN G56NF_GC_STEP_FENCE: a knob of the AI Toolkit runner (MOSS t2i), ignored")
    if os.environ.get("G56NF_TF32"):
        log("WARN G56NF_TF32: a knob of the AI Toolkit runner, not available on the Musubi path, ignored")
    fr = tailmean_fracs(fam, mode)
    if fr:
        kept, gap = tail_plan(fr, steps, every)
        wtext = ",".join(f"{x:g}" for x in fr)
        if kept is None:
            log(f"WARN G56NF_TAILMEAN {wtext}: {gap} - off")
        else:
            holder["tail"] = dict(fracs=wtext, cap=steps, steps=set(kept), gap=gap, got={}, failed=set(), t_cap=0.0, dir=work / "tail")
            log(f"tail mean: on - window {wtext} of the step cap {steps}: the saves at steps {kept} are kept, the mean of them and the "
                f"last save replaces the upload file after training (window spacing {gap})")
    p5 = holder.get("p5") if steps < 10 ** 6 else None   # PLAN5 (None = off): the plan's K_max, window and deadline for the member rules
    spec5 = plan5_spec(p5, fam, mode, data_dir, km, 1, rank, p5["bytes_rank1"], holder) if p5 is not None else None
    if spec5 is not None:
        km = spec5["khi"]
    if km > 1 and steps >= 10 ** 6:
        log("WARN G56NF_MUSUBI_MEMBERS: no step cap (a member has no length), ignored")
    elif km > 1 or spec5 is not None:
        if holder.pop("tail", None):
            log("WARN G56NF_TAILMEAN with G56NF_MUSUBI_MEMBERS: the tail mean is off (the members' mean is shipped)")
        kd = keep_root(work, out_dir) if knob("KEEP_MEMBERS", False, bool) else None
        if kd is not None and inside(kd, out_dir):
            log(f"WARN G56NF_KEEP_MEMBERS ignored: the keep folder {kd} is inside the upload folder")
            kd = None
        holder["musubi_mem"] = dict(k_max=km, L=steps, deadline=t_train_end - POST_S, seed=knob("SEED", SEED, int),
                                    status=work / "musubi_members.json", dir=work / "musubi_members", sds=[], seen=set(), logged=0,
                                    keep=kd)
        if spec5 is not None:   # PLAN5: the runner's plan and the costs it reads
            holder["musubi_mem"].update(plan5=spec5, cost=holder.get("p5_cost"))
        log(f"members: on - K_max {km} members of {steps} steps in one runner process; the exact mean of the finished ones ships")
        if kd is not None:
            log(f"members: every finished member also kept, as shipped and unscaled, as m<k>.safetensors in {kd}")
    mem = holder.get("musubi_mem")
    log("training", public=True)
    rc, killed, shipped, t_first = musubi_child(
        fam, mode, header, musubi_mem_cmd(musubi_temb_cmd(musubi_train_cmd(dit, toml, work / "musubi_out", every, steps, band,
                                                                           bool(mask) or plin, threads, gc, rank), temb), mem),
        work / "musubi_out", musubi_mem_env(musubi_temb_env(musubi_plin_env(musubi_mask_env(tenv, mask, work, work / "musubi_cache", blank),
                                                                            plin, work), temb), mem), mlog, out_dir,
        holder, t_train_end, None, steps)
    if temb:
        musubi_temb_line(out_dir)
    if mask:
        musubi_mask_line(work)
    if plin:
        musubi_plin_line(work)
    if holder.get("members_done"):   # G56NF_MUSUBI_MEMBERS: the mean of the finished members stays; no relaunch joins it
        if rc != 0 and not killed:
            log(f"no relaunch (exit {rc}): the mean of {holder['members_done']} finished members is shipped")
        if rank != MUSUBI_RANK_RECIPE:
            musubi_rank_line(out_dir, rank)
        if spec5 is not None:   # PLAN5: the task's use of its window
            try:
                st5 = json.loads(Path(mem["status"]).read_text())
            except Exception:  # noqa: BLE001
                st5 = {}
            utilization_line(holder, st5, out_dir, *holder.get("t_child", (time.time(), time.time())))
        holder["done"] = True
        log("done", public=True)
        raise SystemExit(0)
    holder.pop("musubi_mem", None)   # (a relaunch trains one member: Musubi's own run)
    if rc != 0 and not killed:   # L2: the training child ended early by itself
        if holder.pop("tail", None):   # G56NF_TAILMEAN: a relaunch is another trajectory - its last save is shipped as it is
            log("tail mean: off - the training child ended early (L2): its saves and the relaunch's are not one trajectory")
        relaunch_musubi(fam, mode, header, work, data_dir, ctrl, dit, env, mlog, out_dir, holder, t_train_end, toml, rc, shipped,
                        t_first, cache_s, every, steps, band, blank, mask, threads, gc, plin, temb, rank)
    elif holder.get("tail"):
        tail_ship(fam, mode, header, holder, out_dir, steps)
    if rank != MUSUBI_RANK_RECIPE:
        musubi_rank_line(out_dir, rank)
    holder["done"] = True
    log("done", public=True)
    raise SystemExit(0)


def run(trainer_args=None, argv=None):
    """The whole task. Never returns: SystemExit(0) after a normal run (the cli() exit-code policy of the trainer keeps a
    usable adapter on any other exit)."""
    main_mod = sys.modules.get("__main__")
    _T0[0] = float(getattr(main_mod, "T0", time.time()))
    a = parse(sys.argv[1:] if argv is None else argv)
    fam, mode = a.model_type, a.training_mode
    loader = LF.loader_of(fam, mode)
    out_dir = Path(a.out_dir) if a.out_dir else CKPT_ROOT / a.task_id / a.expected_repo_name
    if main_mod is not None and hasattr(main_mod, "_OUT_DIR"):
        main_mod._OUT_DIR = out_dir   # the trainer's cli(): a crash after a usable adapter still exits 0
    log("trainer start", public=True)
    try:
        if out_dir.exists():
            for f in out_dir.iterdir():
                if f.is_file():
                    f.unlink()
    except Exception as e:  # noqa: BLE001
        log(f"stale-output clean skipped: {e}")
    hours = float(a.hours_to_complete)
    t_kill = _T0[0] + max(1, int(hours * 3600))
    t_train_end = _T0[0] + hours * 3600 * WALL_FRAC - RESERVE_S
    holder = {"t_kill": t_kill}   # (G56NF_TAILMEAN reads the time left before the watchdog's stop)
    _watchdog(t_kill, out_dir, holder)
    work = WORK / a.task_id
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    p5 = plan5_cell(fam, mode)   # PLAN5: None = off - the deadlines above, member_plan / fill_plan and every file as before
    if p5 is not None:
        t_train_end = plan5_times(p5, fam, t_kill, work, holder, musubi=loader == LF.MUSUBI)
    snap = model_dir(a.model) if loader == LF.AITK else None
    # 1) the evaluator's module set, then the zero LoRA (scores as the base model; never a forfeit)
    try:
        header = LF.read_headers(LF.base_files(fam, mode, snap, CACHE))
        if not header:
            raise FileNotFoundError("empty base header")
        src_header = "task"
    except Exception as e:  # noqa: BLE001  (no readable base: the public header of the same repo, shipped in comfytrain/assets -
        # never under comfytrain/tools, which a release tree does not carry)
        log(f"WARN base header: {type(e).__name__}: {str(e)[:120]}; using the public fixture")
        name = "hidream-o1-edit" if (fam, mode) == ("hidream-o1", LF.EDIT) else fam
        d = json.loads((HERE / "assets" / f"{name}_base_header.json").read_text())
        header = {k: (v[0], tuple(v[1])) for k, v in d["tensors"].items()}
        src_header = "fixture"
    zero = LF.materialize(LF.zero_lora(fam, mode, header))
    LF.save_atomic(zero, out_dir / "last.safetensors")
    rep0 = LF.check(fam, mode, header, zero, expect_full=True, allow_zero=True)
    log(f"zero LoRA written ({len(zero)} tensors, header {src_header}, structure check {'ok' if rep0['ok'] else 'FAILED'}; "
        f"not a scorable file until a trained snapshot replaces it)")
    if p5 is not None:   # PLAN5: the float32 bytes of the file per unit of rank (its plan never passes the file size of maxgb)
        p5["bytes_rank1"] = 4 * sum(int(v.numel()) for v in zero.values()) // (LF.EVAL_RANK if loader == LF.MUSUBI else 1)   # (the Musubi zero LoRA: rank 32)
    del zero
    if mode == LF.EDIT and lpt_radius(fam, mode, hours) > 0:
        log("WARN G56NF_LPT_RADIUS: a knob of the text-to-image training images, not used on an edit task (the pairs stay raw), ignored")
    if loader == LF.MUSUBI:
        run_musubi(a, fam, mode, header, work, out_dir, holder, t_train_end)   # never returns
    if os.environ.get("G56NF_MUSUBI_TEMB"):
        log("WARN G56NF_MUSUBI_TEMB: the Musubi path's knob (hidream-o1 edit), ignored")
    if os.environ.get("G56NF_MUSUBI_RANK"):
        log("WARN G56NF_MUSUBI_RANK: the Musubi path's knob (hidream-o1 edit; the AI Toolkit path has G56NF_RANK), ignored")
    if not AITK_PY.exists() or not (AITK_ROOT / "toolkit").is_dir():
        log(f"WARN no AI Toolkit environment at {AITK_PY} / {AITK_ROOT}: zero LoRA shipped")
        holder["done"] = True
        raise SystemExit(0)
    # 2) data
    zp = dataset_zip(a)
    lpt = 0.0   # the text-to-image training images' Gaussian radius (LPT_RADIUS / G56NF_LPT_RADIUS at the task's hours; 0 = the raw bytes)
    if mode == LF.EDIT:
        data_dir, ctrl, n = prepare_edit(zp, work / "data", work / "extract")
    else:
        lpt = lpt_radius(fam, mode, hours)
        data_dir, ctrl, n = prepare_t2i(zp, work / "data", radius=lpt)
    log(f"data: {n} {'edit rows' if mode == LF.EDIT else 'images'}")
    # 3) offline patches + the job
    st = staged_runtime(fam, mode)
    log(f"runtime assets: {'validator-staged' if st else 'snapshot patches'}")   # private log only
    spec = dict(family=fam, mode=mode, aitk_root=str(AITK_ROOT), hf_home=str(work / "hf"),
                models_path=(st or {}).get("models_path") or str(work / "models"),
                status=str(work / "status.json"), seed=knob("SEED", SEED, int),
                max_steps=step_cap(fam, mode, n),
                snapshot_every_s=knob("SNAPSHOT_EVERY_S", SNAPSHOT_EVERY_S),
                first_snapshot_steps=knob("FIRST_SNAPSHOT_STEPS", FIRST_SNAPSHOT_STEPS, int),
                deadline=t_train_end - POST_S, save_root=str(work / "run" / "g56"))
    ct = captype_cell(fam, mode) if p5 is None and mode == LF.T2I else None   # CAPTYPE_C (a task WITHOUT PLAN5): None = as before
    if ct is not None:   # the caption type's steps per image: the member length (member_plan plans with it), one private line
        ct["kind"], ct_k, ct_n, ct_why = caption_type(data_dir)
        spec["max_steps"], ct_cap = plan5_cap(ct, fam, mode, n, spec["max_steps"])
        log(f"caption type: {ct_k}/{ct_n} captions with text cues (f={ct_k / ct_n if ct_n else 0.0:.3f}) -> {ct['kind']}"
            + (f" ({ct_why})" if ct_why else "") + f"; N {n}: {ct_cap}")
    if lpt > 0:   # the pre-blurred data's radius on record (the data changed, the job did not); no key at 0
        spec["lpt_radius"] = lpt
    if knob("CONSTRUCT_ONLY", False, bool):   # tools/newfam_check.py case F (CPU, real AI Toolkit): build the job, never train
        spec["construct_only"] = True
    spec.update(keep_plan(work, out_dir))
    spec.update(edit_mask_plan(fam, mode, data_dir, ctrl))
    spec.update(ts_strata_plan())
    spec.update(tstrat_plan(fam, mode))
    spec.update(sdpa_plan(fam, mode))
    spec.update(freeze_plan(fam, mode, hours))
    if spec.get("tstrat") and spec.pop("ts_strata", None):
        log("WARN G56NF_TS_STRATA is not used with G56NF_TSTRAT (one sequence of indices hands AI Toolkit the timestep)")
    spec.update(profile_plan())                              # G56NF_PROFILE (a measuring tool): no key unless set
    spec.update(gc_step_plan(fam, mode))
    for key, name, table in (("lora_noop", "LORA_NOOP", LORA_NOOP), ("bnb_nosync", "BNB_NOSYNC", BNB_NOSYNC)):
        if knob(name, bool(table.get((fam, mode), False)), bool):
            spec[key] = True
    if spec.get("lora_noop") or spec.get("bnb_nosync"):
        log(f"speed knobs on: {', '.join(k for k in ('lora_noop', 'bnb_nosync') if spec.get(k))}")
    spec.update(tf32_plan(fam, mode))   # G56NF_TF32 / the release table TF32: no key unless on
    if members_max(fam, mode) > 1:          # K members: no key for a single run
        spec["members"] = members_max(fam, mode)
        if knob("KEEP_MEMBERS", False, bool) and not inside(keep_root(work, out_dir), out_dir):
            spec["members_keep_dir"] = str(keep_root(work, out_dir))
        fr = fill_rho(fam, mode)
        if fr:
            spec["fill_rho"] = {str(k): v for k, v in sorted(fr.items())}
            log(f"fill the window: on - rho_min {', '.join(f'K_fit {k}: {v:g}' for k, v in sorted(fr.items()))}")
    elif os.environ.get("G56NF_FILL_RHO"):
        log("WARN G56NF_FILL_RHO: a single-member run (K_max 1), ignored")
    fr = tailmean_fracs(fam, mode)
    if fr:
        kept, gap = tail_plan(fr, spec["max_steps"], 1)
        wtext = ",".join(f"{x:g}" for x in fr)
        if kept is None:
            log(f"WARN G56NF_TAILMEAN {wtext}: {gap} - off")
        else:
            spec["tail_fracs"] = list(fr)
            holder["tail_aitk"] = dict(fracs=list(fr), t_kill=holder["t_kill"], t_cap=0.0, rate=0.0, seen=set())
            log(f"tail mean: on - window {wtext} of each member's own length (step cap {spec['max_steps']}: the saves at steps {kept} "
                f"and the member's last); the upload file = the exact mean of the members' tail means")
    if st and st.get("hf_hub_cache"):
        spec["hf_hub_cache"] = st["hf_hub_cache"]
    if fam == "flux2-klein-4b" and not st:
        spec["klein_te_dir"] = str(klein_te_dir(snap, work))
        spec["klein_vae"] = [str(snap / "vae" / "diffusion_pytorch_model.safetensors"), str(work / "klein_vae" / "ae.safetensors")]
    if (fam, mode) == ("moss-image-base", LF.EDIT) and not st:
        pdir, how = moss_processor_dir(work)
        log(f"moss edit image processor: {how}")
        if pdir is None:
            log("moss edit: no image processor offline: zero LoRA shipped")
            holder["done"] = True
            raise SystemExit(0)
        spec["moss_processor_dir"] = str(pdir)
    over = {k: v for k, v in os.environ.items() if k.startswith("G56NF_") and v != "" and not k.startswith(("G56NF_AITK", "G56NF_RUNNER", "G56NF_WORK"))}
    if over:
        log(f"research knobs set: {json.dumps(over)}")   # private log only
    ck = config_knobs(fam, mode)
    if spec.get("ts_strata") and ck.pop("ts_bias", None):
        log("WARN G56NF_TS_BIAS is not used with G56NF_TS_STRATA (the strata cover the band evenly)")
    if spec.get("tstrat") and ck.pop("ts_bias", None):
        log("WARN G56NF_TS_BIAS is not used with G56NF_TSTRAT (its sequence keeps AI Toolkit's uniform marginal)")
    bound = seq_bound(fam, mode, data_dir, recipe(fam, mode)["resolution"])
    spec["config"] = build_config(fam, mode, snap, data_dir, ctrl, work / "run", staged=st, bound=bound, knobs=ck)
    if p5 is not None:   # PLAN5: its K_max, window, deadline and costs for the runner; the member rules run whatever K_max is
        if spec.pop("fill_rho", None):
            log("WARN G56NF_FILL_RHO is not used with PLAN5 (its window fills the time)")
        if plan5_kind(p5, mode, data_dir):   # the caption type (when the entry names one): its window, its c - one private line
            spec["max_steps"], why_cap = plan5_cap(p5, fam, mode, n, spec["max_steps"])   # cS / cL: the type's own full length
            log(f"{p5['kind_text']}; N {n}: {why_cap}")
        saves = 1 + (len(tail_steps(spec["tail_fracs"], spec["max_steps"])) if spec.get("tail_fracs") else 0)
        spec["plan5"] = plan5_spec(p5, fam, mode, data_dir, members_max(fam, mode), saves, ck.get("rank", LF.EVAL_RANK), p5["bytes_rank1"], holder)
        if holder.get("tail_aitk") is not None:   # the tail mean's time rule at the plan's own factor (the plan kept that time: saves x K)
            holder["tail_aitk"]["cost_x"] = PLAN5_POST_X
        spec["members"] = spec["plan5"]["khi"]
        if knob("KEEP_MEMBERS", False, bool) and not inside(keep_root(work, out_dir), out_dir):
            spec["members_keep_dir"] = str(keep_root(work, out_dir))
    if ck:   # no key and no line unless one is on; the job without them stays at hand for the L2 relaunch
        gc = spec["config"]["config"]["process"][0]["train"]["gradient_checkpointing"]
        spec["config_plain"] = build_config(fam, mode, snap, data_dir, ctrl, work / "run", staged=st, bound=bound, knobs={}, gc=gc)
        spec["cfg_watch"] = sorted(ck)   # the runner counts what the run trains on and reads what AI Toolkit holds (config_effect_line)
        log(config_knobs_line(ck, n, spec["max_steps"]))
    for k, v in (("test_ballast_gb", knob("TEST_BALLAST_GB", 0.0)), ("test_crash_at_step", knob("TEST_CRASH_AT_STEP", 0, int))):
        if v:
            spec[k] = v
    spec_path = work / "spec.json"
    # 4) train in the child; ship snapshots as they land
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", DISABLE_TELEMETRY="YES", HF_HOME=str(work / "hf"),
               MODELS_PATH=str(work / "models"), TOKENIZERS_PARALLELISM="false")
    for k in [k for k in env if k.startswith("G56CT_")]:
        env.pop(k)
    alog = work / "aitk.log"
    log("training", public=True)
    rc, status, shipped, t_first = aitk_child(fam, mode, header, spec, spec_path, env, alog, out_dir, holder, t_train_end, None)
    if spec.get("keep_dir"):
        have = sorted(int(f.stem[1:]) for f in Path(spec["keep_dir"]).glob("s*.safetensors") if f.stem[1:].isdigit())
        log(f"keep: steps kept {have} of {spec['keep_steps']}")
    if status.get("phase") not in ("done", "finished", "constructed"):
        if holder.get("members_done"):   # K members: the mean of the finished ones stays; no member of another recipe joins it
            log(f"no relaunch (exit {rc}, phase {status.get('phase')}): the mean of {holder['members_done']} finished members is shipped")
        else:
            relaunch_aitk(fam, mode, header, spec, work, env, alog, out_dir, holder, t_train_end, rc, status, shipped, t_first)
    elif holder.get("tail_aitk") and int(spec.get("members") or 1) <= 1 and not spec.get("plan5"):
        tail_ship_aitk(fam, mode, header, holder, out_dir, spec)
    if holder.get("tail_mem") is not None and holder.get("tail_aitk") is not None:
        tail_final(fam, mode, header, holder, out_dir)
    elif p5 is not None:   # PLAN5: the members' concat mean once more in soup_form() when it fits before the plan's end
        plan5_final(fam, mode, header, holder, out_dir)
    if p5 is not None:
        utilization_line(holder, status, out_dir, *holder.get("t_child", (time.time(), time.time())))
    holder["done"] = True
    log("done", public=True)
    raise SystemExit(0)
