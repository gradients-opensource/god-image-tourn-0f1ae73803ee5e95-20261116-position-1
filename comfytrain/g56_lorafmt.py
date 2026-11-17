#!/usr/bin/env python3
"""

















  
  
  
  
  
  
  
  
  

  
  



  

  



  


"""
import json
import os
import re
import struct
from pathlib import Path

T2I, EDIT = "text-to-image", "edit"
AITK, MUSUBI = "aitk", "musubi"
EVAL_RANK = 32
#
#
#
#
MUSUBI_CONV_RANK = 4      # diffusion_objective.py:281-283  conv_dim=4 conv_alpha=1
MUSUBI_CONV_ALPHA = 1.0

#
FAMILIES = {
    "flux2-klein-4b": dict(modes=(T2I, EDIT), repo="black-forest-labs/FLUX.2-klein-base-4B"),
    "qwen-image-edit-2511": dict(modes=(EDIT,), repo="Qwen/Qwen-Image-Edit-2511"),
    "hidream-o1": dict(modes=(T2I, EDIT), repo="HiDream-ai/HiDream-O1-Image"),
    "moss-image-base": dict(modes=(T2I, EDIT), repo="OpenMOSS-Team/MOSS-Image-Base"),
}
HIDREAM_EDIT_CKPT = "hf_cache/checkpoints/hidream_o1_image_bf16.safetensors"  # downloader.py:592-602, under /cache
KLEIN_ROOT_FILE = "flux-2-klein-base-4b.safetensors"                          # AT flux2_klein_model.py:54


def loader_of(family, mode):
    if family not in FAMILIES:
        raise ValueError(f"not a new-family model type: {family}")
    if mode not in FAMILIES[family]["modes"]:
        raise ValueError(f"{family} has no {mode} mode (validator: {FAMILIES[family]['modes']})")
    return MUSUBI if (family, mode) == ("hidream-o1", EDIT) else AITK


# ----------------------------------------------------------------------------------------------------------------------
# safetensors headers (names, dtypes, shapes; no tensor data)
# ----------------------------------------------------------------------------------------------------------------------
def read_header(path):
    """{name: (dtype, shape)} of one safetensors file (the 8-byte length + JSON header only)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 100_000_000:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in h.items()}


def read_headers(paths):
    out = {}
    for p in paths:
        out.update(read_header(p))
    return out


def base_files(family, mode, snapshot, cache_root="/cache"):
    """The base-model files whose tensor names the evaluator's model uses (AT loads these same files from the task snapshot;
    the evaluator downloads the same repo / the same Comfy file)."""
    snap = Path(snapshot) if snapshot is not None else None
    if (family, mode) == ("hidream-o1", EDIT):
        return [Path(cache_root) / HIDREAM_EDIT_CKPT]
    if family == "flux2-klein-4b":
        return [snap / KLEIN_ROOT_FILE]
    if family in ("qwen-image-edit-2511", "moss-image-base"):
        return sorted((snap / "transformer").glob("*.safetensors"))
    if family == "hidream-o1":
        return sorted(snap.glob("model-*.safetensors"))
    raise ValueError(family)


def musubi_normalize_name(key):
    """MU hidream_o1_utils.py:58-72 normalize_single_checkpoint_state_dict: None = dropped from the model."""
    if key.startswith(("vae.", "text_encoders.")):
        return None
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model."):]
    if "visual.deepstack_merger_list" in key:
        return None
    if key.startswith("model.") or key.startswith("lm_head."):
        return key
    return "model." + key


# ----------------------------------------------------------------------------------------------------------------------
# the evaluator's network module set
# ----------------------------------------------------------------------------------------------------------------------
# AT transformer_only block names (substring of the dotted module name, lora_special.py:524-535)
_AT_BLOCKS = {
    "flux2-klein-4b": ("double_blocks", "single_blocks"),
    "qwen-image-edit-2511": ("transformer_blocks",),
    "moss-image-base": ("layers",),
    "hidream-o1": ("model.language_model.layers",),
}
# MU HiDream-O1 i2i target classes (networks/lora_hidream_o1.py:12-22) as module-path prefixes of the Comfy checkpoint:
# Qwen3VLTextDecoderLayer = model.language_model.layers.N, BottleneckPatchEmbed = model.x_embedder, FinalLayer =
# model.final_layer2, Qwen3VLVisionPatchEmbed = model.visual.patch_embed, Qwen3VLVisionBlock = model.visual.blocks.N,
# Qwen3VLVisionPatchMerger = model.visual.merger (the deepstack mergers are removed for Comfy files, hidream_o1_utils.py:91-93).
_MU_TARGETS = re.compile(r"^model\.(language_model\.layers\.\d+|x_embedder|final_layer2|visual\.patch_embed|"
                         r"visual\.blocks\.\d+|visual\.merger)\.")
_MU_EXCLUDE = re.compile(r".*(embed_tokens|lm_head).*")  # lora_hidream_o1.py:47
# MU HiDream-O1 TimestepEmbedder (networks/lora_hidream_o1.py:14) = model.t_embedder1 (mlp.0 / mlp.2): in the list of the evaluator's
# loader (create_arch_network_from_weights, HIDREAM_O1_TARGET_REPLACE_MODULES :24, 64-74), not in the training lists (:20-21, 37-39).
# The loader builds a module only for a lora name the FILE holds (networks/lora.py:983-1011, 617-621): loaded when present, never
# required (G56NF_MUSUBI_TEMB trains them; optional_modules).
_MU_OPTIONAL = re.compile(r"^model\.t_embedder1\.")


def optional_modules(family, mode, header):
    """{module: spec} - the evaluator modules a file MAY hold and is never required to (eval_modules' form): on the Musubi path the
    TimestepEmbedder's linears (_MU_OPTIONAL); {} on the AI Toolkit path."""
    if loader_of(family, mode) != MUSUBI:
        return {}
    mods = {}
    for k, (dt, shp) in header.items():
        name = musubi_normalize_name(k)
        if name is None or not name.endswith(".weight") or len(shp) != 2:
            continue
        path = name[:-len(".weight")]
        if _MU_OPTIONAL.match(path + ".") and not _MU_EXCLUDE.fullmatch(path):
            mods[path] = dict(kind="linear", in_f=shp[1], out_f=shp[0])
    return dict(sorted(mods.items()))


def eval_modules(family, mode, header):
    """{module: spec} - the evaluator's LoRA modules. AT: module = the network's dotted name ('transformer.' + path inside
    the model); spec = dict(kind='linear', in_f, out_f). MU: module = the dotted path inside the model; spec also carries
    kind 'conv' with the kernel shape (lora_down = Conv(in, 4, kernel), lora_up = Conv(4, out, 1...))."""
    loader = loader_of(family, mode)
    mods = {}
    if loader == MUSUBI:
        for k, (dt, shp) in header.items():
            name = musubi_normalize_name(k)
            if name is None or not name.endswith(".weight"):
                continue
            path = name[:-len(".weight")]
            if not _MU_TARGETS.match(path + ".") or _MU_EXCLUDE.fullmatch(path):
                continue
            if len(shp) == 2:
                mods[path] = dict(kind="linear", in_f=shp[1], out_f=shp[0])
            elif len(shp) in (4, 5):  # Conv2d / Conv3d (Linear-like when the kernel is 1)
                one = all(s == 1 for s in shp[2:])
                mods[path] = dict(kind="linear1x1" if one else "conv", in_f=shp[1], out_f=shp[0], kernel=tuple(shp[2:]))
        return dict(sorted(mods.items()))
    blocks = _AT_BLOCKS[family]
    for k, (dt, shp) in header.items():
        if not k.endswith(".weight") or len(shp) != 2:
            continue
        path = k[:-len(".weight")]
        if family == "moss-image-base":
            m = re.match(r"^(layers\.\d+\.proj)\.(q|k|v)$", path)
            if m:  # the model fuses the vendor's q/k/v into proj.qkv (AT moss transformer.py:213-217, 496-511)
                path = m.group(1) + ".qkv"
                prev = mods.get("transformer." + path)
                if prev is not None:
                    if prev["in_f"] != shp[1]:
                        raise ValueError(f"MOSS {path}: q/k/v in_features differ")
                    prev["out_f"] += shp[0]
                    continue
        name = "transformer." + path
        if any(b in name for b in blocks):
            mods[name] = dict(kind="linear", in_f=shp[1], out_f=shp[0])
    return dict(sorted(mods.items()))


# ----------------------------------------------------------------------------------------------------------------------
# key names of the shipped file
# ----------------------------------------------------------------------------------------------------------------------
def shipped_prefix(family, module):
    """The network module name -> the module part of the shipped key (what the model's convert_lora_weights_before_save
    writes: AT base_model.py:1743-1750 'transformer.' -> 'diffusion_model.'; HiDream hidream_o1_model.py:534-540 also drops
    '.model.')."""
    if not module.startswith("transformer."):
        raise ValueError(module)
    if family == "hidream-o1":
        return "diffusion_model." + module[len("transformer."):].replace("model.", "", 1)
    return "diffusion_model." + module[len("transformer."):]


def musubi_name(path):
    return ("lora_unet." + path).replace(".", "_")  # networks/lora.py create_modules: f"{pfx}.{original_name}".replace(".", "_")


def zero_lora(family, mode, header, rank=1, dtype="bf16"):
    """The fallback file as {key: (shape, fill)}: every evaluator module, A = 0, B = 0 (AT: rank `rank`, zero-padded to 32
    on load; MU: exactly 32 / conv 4 with the alpha buffers 32 / 1 - MU raises on any other shape). Scores as the base."""
    mods = eval_modules(family, mode, header)
    out = {}
    if loader_of(family, mode) == MUSUBI:
        for path, s in mods.items():
            n = musubi_name(path)
            if s["kind"] == "conv":
                r, a = MUSUBI_CONV_RANK, MUSUBI_CONV_ALPHA
                out[n + ".lora_down.weight"] = ((r, s["in_f"]) + s["kernel"], 0.0)
                out[n + ".lora_up.weight"] = ((s["out_f"], r) + (1,) * len(s["kernel"]), 0.0)
            elif s["kind"] == "linear1x1":
                r, a = EVAL_RANK, float(EVAL_RANK)
                out[n + ".lora_down.weight"] = ((r, s["in_f"]) + s["kernel"], 0.0)
                out[n + ".lora_up.weight"] = ((s["out_f"], r) + (1,) * len(s["kernel"]), 0.0)
            else:
                r, a = EVAL_RANK, float(EVAL_RANK)
                out[n + ".lora_down.weight"] = ((r, s["in_f"]), 0.0)
                out[n + ".lora_up.weight"] = ((s["out_f"], r), 0.0)
            out[n + ".alpha"] = ((), a)
        return out
    for name, s in mods.items():
        p = shipped_prefix(family, name)
        out[p + ".lora_A.weight"] = ((rank, s["in_f"]), 0.0)
        out[p + ".lora_B.weight"] = ((s["out_f"], rank), 0.0)
    return out


def materialize(spec, dtype="bf16"):
    """{key: (shape, fill)} -> {key: torch tensor}."""
    import torch
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    return {k: torch.full(shape, fill, dtype=torch.float32 if k.endswith(".alpha") else dt) for k, (shape, fill) in spec.items()}


def save_atomic(sd, path, metadata=None):
    """Atomic write (tmp + os.replace, 0644). No metadata unless given (only the LoRA tensors leave the container)."""
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp), metadata or None)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass
    return path


# ----------------------------------------------------------------------------------------------------------------------
# an AI Toolkit save -> the shipped file
# ----------------------------------------------------------------------------------------------------------------------
def normalize_aitk(family, sd):
    """AI Toolkit's own save (keys diffusion_model.*.lora_A/B.weight, MOSS q/k/v split with cloned lora_A) -> the shipped
    state: float32, lora_A / lora_B names, no alpha, MOSS fused into proj.qkv exactly (only when the three lora_A are
    bit-equal - otherwise ValueError: the evaluator would build the rank-3r block form and cut it to 32)."""
    import torch
    out = {}
    for k, v in sd.items():
        if k.endswith(".alpha"):
            continue  # AT peft saves carry none; any alpha would be skipped by the evaluator anyway
        k2 = k.replace("lora_down", "lora_A").replace("lora_up", "lora_B")
        out[k2] = v.detach().to(torch.float32).contiguous()
    if family == "moss-image-base":
        qs = [k for k in out if k.endswith(".proj.q.lora_A.weight")]
        for qa in qs:
            base = qa[: -len(".q.lora_A.weight")]
            parts = [f"{base}.{p}" for p in ("q", "k", "v")]
            if not all(p + ".lora_A.weight" in out and p + ".lora_B.weight" in out for p in parts):
                raise ValueError(f"MOSS {base}: incomplete q/k/v")
            downs = [out[p + ".lora_A.weight"] for p in parts]
            if not all(torch.equal(downs[0], d) for d in downs[1:]):
                raise ValueError(f"MOSS {base}: q/k/v lora_A differ (would load as a rank-3r block LoRA, cut to 32)")
            ups = [out.pop(p + ".lora_B.weight") for p in parts]
            for p in parts:
                out.pop(p + ".lora_A.weight")
            out[base + ".qkv.lora_A.weight"] = downs[0]
            out[base + ".qkv.lora_B.weight"] = torch.cat(ups, 0).contiguous()
    return dict(sorted(out.items()))


UP_KEYS = (".lora_B.weight", ".lora_up.weight")


def scale_delta(sd, s):
    """`sd` with every module's delta times s, exactly (delta' = s x delta): each up factor (lora_B / lora_up) times s, computed in
    float64 and stored float32; the down factors and any alpha are left as they are (AI Toolkit files: the evaluator's module scale
    is fixed at 1 and alpha is ignored; Musubi files: the file's alpha keeps its meaning). One side only - scaling both factors,
    or a factor and its alpha, would give s x s. s == 1 -> `sd` itself (no copy: the shipped file stays byte for byte)."""
    import torch
    s = float(s)
    if s == 1.0:
        return sd
    out = {}
    for k, v in sd.items():
        if k.endswith(UP_KEYS):
            out[k] = (v.double() * s).to(torch.float32)
        else:
            out[k] = v
    return out


DOWN_KEYS = (".lora_A.weight", ".lora_down.weight")
MEAN_TOL = 1e-5           # exactness gate of mean_delta: relative Frobenius error per module against the float64 reference
SOUP_FORMS = ("svd_full", "svd32", "concat")


def _ab_modules(sd):
    """[module] of a file in the shipped AI Toolkit layout (<module>.lora_A.weight r x in, <module>.lora_B.weight out x r, no
    alpha = scale 1); any other key, or an unpaired / non 2-D factor, raises ValueError."""
    other = [k for k in sd if not k.endswith((".lora_A.weight", ".lora_B.weight"))]
    if other:
        raise ValueError(f"keys outside the lora_A / lora_B layout: {other[:3]}")
    mods = sorted(k[: -len(".lora_A.weight")] for k in sd if k.endswith(".lora_A.weight"))
    if mods != sorted(k[: -len(".lora_B.weight")] for k in sd if k.endswith(".lora_B.weight")):
        raise ValueError("lora_A / lora_B modules do not pair up")
    for m in mods:
        a, b = sd[m + ".lora_A.weight"], sd[m + ".lora_B.weight"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"{m}: factors {tuple(b.shape)} x {tuple(a.shape)}")
    return mods


def _gram_err(bp, ap, b, a):
    """||bp ap - b a||_F / ||b a||_F through r x r Gram matrices (float64; no out x in product)."""
    import torch
    n_ref = float(torch.trace((b.T @ b) @ (a @ a.T)))
    n_out = float(torch.trace((bp.T @ bp) @ (ap @ ap.T)))
    n_x = float(torch.trace((b.T @ bp) @ (ap @ a.T)))
    return (max(n_ref + n_out - 2.0 * n_x, 0.0) / n_ref) ** 0.5 if n_ref > 0 else (0.0 if n_out == 0 else float("inf"))


def _svd_factors(b, a, r_out):
    """b (out x R) @ a (R x in), float64 -> (B' out x r_out, A' r_out x in, singular values): QR of both factors, SVD of the small
    core, components ordered by singular value, factors balanced (sqrt of the singular value on each side); zero-padded when the
    module cannot hold r_out components."""
    import torch
    qb, rb = torch.linalg.qr(b)
    qa, ra = torch.linalg.qr(a.T)
    u, sv, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    k = min(r_out, sv.shape[0])
    sq = sv[:k].sqrt()
    bp, ap = (qb @ u[:, :k]) * sq, sq[:, None] * (vh[:k] @ qa.T)
    if k < r_out:
        bp = torch.cat([bp, bp.new_zeros(bp.shape[0], r_out - k)], 1)
        ap = torch.cat([ap, ap.new_zeros(r_out - k, ap.shape[1])], 0)
    return bp, ap, sv


def mean_delta(members, scale=1.0, form="svd_full", tol=MEAN_TOL, weights=None, gate=True):
    """
    
    
      
      
                  
      
    
    
    
    
    
    
    
    
    
    """
    import torch
    if form not in SOUP_FORMS:
        raise ValueError(f"form {form!r}")
    members = list(members)
    if not members:
        raise ValueError("no member")
    mods = _ab_modules(members[0])
    for sd in members[1:]:
        if _ab_modules(sd) != mods:
            raise ValueError("the members do not have the same modules")
    rank = {sum(int(sd[m + ".lora_A.weight"].shape[0]) for sd in members) for m in mods}
    report = dict(k=len(members), form=form, n=len(mods), rank=sorted(rank), max_rel_err=0.0, fallback=[], trunc=0.0)
    if weights is not None and len(weights) != len(members):
        raise ValueError(f"{len(weights)} weights for {len(members)} members")
    if len(members) == 1:
        return scale_delta(members[0], scale if weights is None else float(weights[0])), report
    w = float(scale) / len(members)
    out = {}
    for m in mods:
        ka, kb = m + ".lora_A.weight", m + ".lora_B.weight"
        b = torch.cat([w * sd[kb].double() for sd in members], 1)
        if weights is not None:   # G56NF_TAILMEAN on the AI Toolkit path: one weight per member in place of scale / K
            b = torch.cat([float(wi) * sd[kb].double() for wi, sd in zip(weights, members)], 1)
        a = torch.cat([sd[ka].double() for sd in members], 0)
        bp, ap = b.to(torch.float32), a.to(torch.float32)
        if form != "concat":
            try:
                r_out = b.shape[1] if form == "svd_full" else EVAL_RANK
                bs, as_, sv = _svd_factors(b, a, r_out)
                bs, as_ = bs.to(torch.float32), as_.to(torch.float32)
                good = bool(torch.isfinite(bs).all()) and bool(torch.isfinite(as_).all())
                if form == "svd32":
                    tot = float((sv ** 2).sum())
                    report["trunc"] = max(report["trunc"], (float((sv[r_out:] ** 2).sum()) / tot) ** 0.5 if tot > 0 else 0.0)
                else:
                    good = good and _gram_err(bs.double(), as_.double(), b, a) <= tol
            except Exception:  # noqa: BLE001  (a failed factorization never costs the file: the concat form is exact)
                good = False
            if good:
                bp, ap = bs, as_
            else:
                report["fallback"].append(m)
        if (form != "svd32" or m in report["fallback"]) and (gate or form != "concat"):
            report["max_rel_err"] = max(report["max_rel_err"], _gram_err(bp.double(), ap.double(), b, a))
        out[ka], out[kb] = ap.contiguous(), bp.contiguous()
    return out, report


def concat_mean(members):
    """
    
    
    
    
    
    
    
    
    
    """
    import torch
    K = len(members)
    if K < 2:
        raise ValueError(f"{K} member(s): a mean needs two")
    dn, up, al = ".lora_down.weight", ".lora_up.weight", ".alpha"

    def mods(sd):
        names = sorted(k[: -len(dn)] for k in sd if k.endswith(dn))
        if not names or sorted(sd) != sorted(m + s for m in names for s in (dn, up, al)):
            raise ValueError("keys outside the lora_down / lora_up / alpha layout, or a module without all three")
        return names
    names = mods(members[0])
    for i, sd in enumerate(members[1:], 2):
        if mods(sd) != names:
            raise ValueError(f"member {i}: other modules than member 1")
        for m in names:
            if sd[m + dn].shape != members[0][m + dn].shape or sd[m + up].shape != members[0][m + up].shape:
                raise ValueError(f"member {i}: {m} has another shape than member 1")
            if not torch.equal(sd[m + al], members[0][m + al]):
                raise ValueError(f"member {i}: {m} has another alpha than member 1")
    out = {}
    for m in names:
        out[m + dn] = torch.cat([sd[m + dn] for sd in members], 0).contiguous()
        out[m + up] = torch.cat([sd[m + up] for sd in members], 1).contiguous()
        out[m + al] = members[0][m + al].clone()
    problems, ranks, live = [], {"linear": set(), "conv": set()}, False
    for m in names:
        r = members[0][m + dn].shape[0]
        ranks["conv" if out[m + dn].ndim > 2 else "linear"].add(int(out[m + dn].shape[0]))
        if out[m + dn].shape[0] != K * r or out[m + up].shape[1] != K * r:
            problems.append(f"{m}: rank {out[m + dn].shape[0]} is not {K} x {r}")
            continue
        if not torch.equal(out[m + al], members[0][m + al]):
            problems.append(f"{m}: alpha changed")
        if not all(torch.equal(out[m + dn][i * r:(i + 1) * r], sd[m + dn]) and torch.equal(out[m + up][:, i * r:(i + 1) * r], sd[m + up])
                   for i, sd in enumerate(members)):
            problems.append(f"{m}: a block is not its member's factor")
        live = live or float(out[m + up].float().abs().max()) > 0
    if not all(bool(torch.isfinite(v).all()) for v in out.values()):
        problems.append("non-finite values")
    if not live:
        problems.append("every up factor is zero")
    return out, dict(members=K, ranks={k: sorted(v) for k, v in ranks.items()}, problems=problems)


def uniform_rank(sd):
    """
    
    """
    ranks = {}
    for k, v in sd.items():
        if k.endswith(DOWN_KEYS) and getattr(v, "ndim", 0) == 2:
            ranks.setdefault(int(v.shape[0]), []).append(k)
    problems = []
    if len(ranks) != 1:
        problems.append(f"linear ranks {sorted(ranks)} ({ {r: len(v) for r, v in sorted(ranks.items())} } modules)")
    alphas = [k for k in sd if k.endswith(".alpha")]
    if alphas:
        problems.append(f"{len(alphas)} alpha keys ({alphas[0]})")
    return dict(ok=not problems, rank=max(ranks) if ranks else None, problems=problems)


# ----------------------------------------------------------------------------------------------------------------------
# mirror of the evaluator's loaders
# ----------------------------------------------------------------------------------------------------------------------
def _aitk_convert_before_load(family, sd):
    """The model's convert_lora_weights_before_load (keys -> network names; MOSS also fuses q/k/v with tensors)."""
    import torch
    if family == "hidream-o1":  # hidream_o1_model.py:542-549
        out = {}
        for k, v in sd.items():
            nk = k.replace("diffusion_model.", "transformer.model.")
            nk = nk.replace("transformer.model.model.", "transformer.model.")
            out[nk] = v
        return out
    # lora_keys_use_comfy_prefix (flux2_model.py:458, qwen_image.py:361, moss_image.py:179) -> base_model.py:1752-1759
    sd = {k.replace("diffusion_model.", "transformer."): v for k, v in sd.items()}
    if family != "moss-image-base":
        return sd
    out = dict(sd)  # moss_image.py:680-710
    down_keys = [k for k in sd if k.endswith((".proj.q.lora_A.weight", ".proj.q.lora_down.weight"))]
    for q_down in down_keys:
        base, _, down_suffix = q_down.partition(".proj.q.")
        up_suffix = "lora_B.weight" if down_suffix == "lora_A.weight" else "lora_up.weight"
        names = [f"{base}.proj.{p}." for p in ("q", "k", "v")]
        if not all(n + down_suffix in sd and n + up_suffix in sd for n in names):
            continue
        downs = [out.pop(n + down_suffix) for n in names]
        ups = [out.pop(n + up_suffix) for n in names]
        alphas = [out.pop(n + "alpha", None) for n in names]
        rank = downs[0].shape[0]
        if all(torch.equal(downs[0], d) for d in downs[1:]):
            down, up = downs[0], torch.cat(ups, dim=0)
        else:
            down = torch.cat(downs, dim=0)
            up = torch.block_diag(*ups)
        out[f"{base}.proj.qkv.{down_suffix}"] = down
        out[f"{base}.proj.qkv.{up_suffix}"] = up
        if alphas[0] is not None:
            out[f"{base}.proj.qkv.alpha"] = alphas[0] * (down.shape[0] / rank)
    return out


def _aitk_peft_key(key):
    """."""
    if key.endswith(".alpha"):
        return None
    k = key.replace("lora_A", "lora_down").replace("lora_B", "lora_up")
    k = k.replace(".", "$$").replace("$$lora_down$$", ".lora_down.").replace("$$lora_up$$", ".lora_up.")
    if k.endswith("$$diff"):
        k = k[:-len("$$diff")] + ".diff"
    elif k.endswith("$$diff_b"):
        k = k[:-len("$$diff_b")] + ".diff_b"
    return k


def _intended(sd, mod_keys, rank_dim=0):
    """The file's own meaning of a module (kohya / Comfy convention): delta = (alpha / r) * up @ down, alpha = r when absent."""
    down, up, alpha = mod_keys
    a = sd[down].float().reshape(sd[down].shape[0], -1)
    b = sd[up].float().reshape(sd[up].shape[0], -1)
    r = a.shape[0]
    s = (float(sd[alpha].float()) / r) if (alpha is not None and alpha in sd) else 1.0
    return b, a, s


def _rel_err(b1, a1, s1, b2, a2, s2):
    """||s1 b1 a1 - s2 b2 a2||_F / max(||s2 b2 a2||_F, tiny) via r x r Gram matrices (no out x in product)."""
    import torch
    def tr(x, y, z, w):  # <x w, y z>_F = tr(w^T x^T y z) = tr((x^T y)(z w^T)) for x: o x r1, y: o x r2, z: r2 x i, w: r1 x i
        return float(torch.trace((x.T.double() @ y.double()) @ (z.double() @ w.T.double())))
    n11 = s1 * s1 * tr(b1, b1, a1, a1)
    n22 = s2 * s2 * tr(b2, b2, a2, a2)
    n12 = s1 * s2 * tr(b1, b2, a2, a1)
    num = max(n11 + n22 - 2 * n12, 0.0) ** 0.5
    den = max(n22, n11) ** 0.5
    return (num / den) if den > 1e-30 else 0.0, den


def check(family, mode, header, sd, tol=1e-5, expect_full=False, keep_loaded=False, allow_zero=False):
    """
    
    
    
    """
    import torch
    mods = eval_modules(family, mode, header)
    rep = dict(family=family, mode=mode, loader=loader_of(family, mode), n_eval_modules=len(mods), n_file_keys=len(sd),
               dropped=[], shape_errors=[], cut=[], scale_errors=[], uncovered=[], max_rel_err=0.0, notes=[], zero_up=True)
    if rep["loader"] == MUSUBI:
        # create_arch_network_from_weights: one module per lora name of the FILE, with the file's own rank (and conv rank) and
        # alpha (missing = rank) -> the file's own scale; file dims must fit the target layer (in / out / kernel)
        net = {}
        optional = optional_modules(family, mode, header)   # the TimestepEmbedder: loaded when the file holds it, never required
        for path, s in list(mods.items()) + list(optional.items()):
            n = musubi_name(path)
            key_d = n + ".lora_down.weight"
            r = int(sd[key_d].shape[0]) if key_d in sd and getattr(sd[key_d], "ndim", 0) >= 2 else EVAL_RANK
            if s["kind"] in ("conv", "linear1x1"):
                k = s["kernel"]
                net[n] = dict(down=(r, s["in_f"]) + k, up=(s["out_f"], r) + (1,) * len(k))
            else:
                net[n] = dict(down=(r, s["in_f"]), up=(s["out_f"], r))
            a = sd.get(n + ".alpha")
            net[n]["scale"] = (float(a.float()) if a is not None else float(r)) / r
        seen = set()
        for k, v in sd.items():
            m, _, suf = k.partition(".")
            if m not in net or suf not in ("lora_down.weight", "lora_up.weight", "alpha"):
                rep["dropped"].append(k)  # strict=False: ignored
                continue
            seen.add(m)
            want = () if suf == "alpha" else net[m]["down" if suf == "lora_down.weight" else "up"]
            if tuple(v.shape) != tuple(want):
                rep["shape_errors"].append(f"{k}: {tuple(v.shape)} != {tuple(want)} (load_state_dict raises)")
        skip = {musubi_name(p) for p in optional}
        for m, spec in net.items():
            if m not in seen:
                if m not in skip:
                    rep["uncovered"].append(m)
                continue
            keys = (m + ".lora_down.weight", m + ".lora_up.weight", m + ".alpha")
            if keys[0] not in sd or keys[1] not in sd or rep["shape_errors"]:
                continue
            b1, a1, s_int = _intended(sd, keys)
            if spec["scale"] == s_int:
                continue   # the same two tensors at the same scale: the same product, error 0 (no Gram matrices)
            err, _ = _rel_err(b1, a1, spec["scale"], b1, a1, s_int)
            rep["max_rel_err"] = max(rep["max_rel_err"], err)
            if err > tol:
                rep["scale_errors"].append(f"{m}: evaluator scale {spec['scale']:g} vs the file's {s_int:g}")
        rep["zero_up"] = not any(float(sd[m + ".lora_up.weight"].float().abs().max()) > 0 for m in seen
                                 if m + ".lora_up.weight" in sd and sd[m + ".lora_up.weight"].numel())
    else:
        # the evaluator network's rank = the file's largest linear rank (lora_down / lora_A, 2-D) before any conversion
        lin = [int(v.shape[0]) for k, v in sd.items() if k.endswith((".lora_down.weight", ".lora_A.weight")) and v.ndim == 2]
        if not lin:
            rep["notes"].append("no linear lora_down / lora_A weight: the evaluator raises")
            rep["no_linear"] = True
        R = max(lin) if lin else EVAL_RANK
        rep["eval_rank"] = R
        conv = _aitk_convert_before_load(family, dict(sd))
        net = {}
        for name, s in mods.items():
            ln = name.replace(".", "$$")
            net[ln] = s
        loaded, intended = {}, {}
        for k, v in conv.items():
            if k.endswith(".alpha"):  # skipped by the loader; kept as the file's own meaning of its module
                m_a = k[: -len(".alpha")].replace(".", "$$")
                intended.setdefault(m_a, {})["alpha"] = v
                continue
            lk = _aitk_peft_key(k)
            m, sep, suf = lk.partition(".")
            if not sep or m not in net or suf not in ("lora_down.weight", "lora_up.weight"):
                rep["dropped"].append(k)
                continue
            intended.setdefault(m, {})[suf] = v
            s = net[m]
            if v.ndim != 2:
                rep["shape_errors"].append(f"{k}: {v.ndim}-D")
                continue
            tgt = (R, s["in_f"]) if suf == "lora_down.weight" else (s["out_f"], R)
            src = tuple(v.shape)
            if src == tgt:
                val = v
            elif suf == "lora_down.weight" and src[0] < tgt[0] and src[1] == tgt[1]:
                val = torch.zeros(tgt, dtype=v.dtype); val[:src[0], :src[1]] = v
            elif suf == "lora_up.weight" and src[1] < tgt[1] and src[0] == tgt[0]:
                val = torch.zeros(tgt, dtype=v.dtype); val[:src[0], :src[1]] = v
            elif suf == "lora_down.weight" and src[0] > tgt[0]:
                rep["cut"].append(f"{k}: rank {src[0]} -> {tgt[0]}")
                val = v[:tgt[0], :tgt[1]]
            elif suf == "lora_up.weight" and src[1] > tgt[1]:
                rep["cut"].append(f"{k}: rank {src[1]} -> {tgt[1]}")
                val = v[:tgt[0], :tgt[1]]
            else:
                rep["shape_errors"].append(f"{k}: {src} vs {tgt} (ValueError: unhandled LoRA shape change)")
                continue
            if tuple(val.shape) != tgt:  # e.g. the in/out side differs: load_state_dict would raise
                rep["shape_errors"].append(f"{k}: {src} does not fit {tgt}")
                continue
            loaded.setdefault(m, {})[suf] = val
        for m, s in net.items():
            got = loaded.get(m)
            if not got:
                rep["uncovered"].append(m.replace("$$", "."))
                continue
            if len(got) != 2:
                rep["notes"].append(f"{m.replace('$$', '.')}: only {sorted(got)} in the file (the other factor stays at init)")
                continue
        # the effective delta of every module (loaded factors, scale 1) vs the file's own meaning of the same module
        worst = 0.0
        for m, got in loaded.items():
            it = intended.get(m, {})
            if len(got) != 2 or "lora_down.weight" not in it or "lora_up.weight" not in it:
                continue
            a_i = it["lora_down.weight"].float().reshape(it["lora_down.weight"].shape[0], -1)
            b_i = it["lora_up.weight"].float().reshape(it["lora_up.weight"].shape[0], -1)
            s_i = float(it["alpha"].float()) / a_i.shape[0] if "alpha" in it else 1.0
            if "alpha" not in it and got["lora_up.weight"] is it["lora_up.weight"] and got["lora_down.weight"] is it["lora_down.weight"]:
                continue   # the loader keeps the file's own two tensors at scale 1: the same product, error 0 (no Gram matrices)
            err, _ = _rel_err(got["lora_up.weight"].float(), got["lora_down.weight"].float(), 1.0, b_i, a_i, s_i)
            worst = max(worst, err)
            if err > tol:
                rep["scale_errors"].append(f"{m.replace('$$', '.')}: evaluator delta differs from the file's own by "
                                           f"{err:.3g} (relative Frobenius)")
        rep["max_rel_err"] = worst
        rep["zero_up"] = not any(float(got["lora_up.weight"].float().abs().max()) > 0 for got in loaded.values()
                                 if "lora_up.weight" in got and got["lora_up.weight"].numel())
        if keep_loaded:  # the factors the loader leaves in the network (tools/lora_format_check.py --impl exact)
            rep["_loaded"] = {m.replace("$$", "."): got for m, got in loaded.items()}
    rep["coverage"] = 1.0 - len(rep["uncovered"]) / max(1, len(mods))
    bad = (rep["dropped"] or rep["shape_errors"] or rep["cut"] or rep["scale_errors"] or (expect_full and rep["uncovered"])
           or rep.get("no_linear") or (rep["zero_up"] and not allow_zero))
    rep["ok"] = not bad and rep["coverage"] > 0
    return rep


def load_file_tensors(path):
    from safetensors.torch import load_file
    return load_file(str(path))
