"""G56CT_DENSE (T112, 14 Oct 2026; ideogram4 only; OFF by default): after training, rewrite the shipped LoRA file as a dense
on-grid file the evaluator applies without re-quantization noise.

Why: the evaluator (ComfyUI 694815f) merges every patched fp8 matrix in fp16 and re-quantizes it with scale = amax/448 and
stochastic rounding (comfy/ops.py set_weight, comfy/quant_ops.py _TensorCoreFP8LayoutBase.quantize). When the merged matrix
W' has amax(|W'|) == 448 * 2^k (integer k, -15 <= k <= 0) and every W'/2^k is an fp8 e4m3 value, that scale is exactly 2^k, the
scaled values are exact and the rounding keeps them: the evaluator's weight is W' bit for bit.

File: per LoRA module m of the shipped file, 'm.lokr_w1' = F32 [[2^k]] and 'm.lokr_w2' = F8_E4M3 (W' - W0) / 2^k (comfy/
weight_adapter/lokr.py: w1 / w2 given -> alpha 1.0; kron(w1, w2) in fp32 is exact; cast to fp16; weight += ...). One byte per
weight: 8.98 GB for ideogram4's 204 block matrices, whatever the LoRA rank (the rank-576..768 packages are 7.6..10.2 GB).

W' = the shipped LoRA's merge, W0 + dW (dW = up @ down * alpha / rank exactly as the evaluator's LoRA adapter computes it),
rounded onto the 2^k grid (mode rtn: nearest; hyb: nearest except where that would give W0 back although dW != 0, there an
unbiased stochastic choice; sr: unbiased stochastic everywhere; the draws are a counter hash of (seed, module, element), no
RNG state). Rounding keeps (W' - W0)/2^k an e4m3 value (the nearest feasible value when the plain rounding is not).
W0 = what the trainer can build: RTNE(fp32(q_c * s_c)) of the task folder's per-channel fp8 transformer (= ideogram4_state_dict
evalq). The evaluator's own base (Comfy-Org scale-1 file) differs in ~1.07% of the elements (measured 14 Oct, all 211 matrices
of the cache vs the Comfy-Org file); an element is 'certain' when no scale-1 rounding boundary lies within the cache's own
rounding interval (0 mismatches among certain elements on layers 0 and 33). So: k = smallest with 448 * 2^k >= amax; the pin
(the one element set to sign * 448 * 2^k) is a certain element (the largest |W'| certain element; else a certain zero); an
uncertain element keeps |W'| + (the largest possible W0 difference) <= 448 * 2^k. Then the evaluator's scale is 2^k whatever
the uncertain elements' W0 are; an uncertain element whose W0 differs ends at W' +- 1 ulp of W0 (as it would through the LoRA
path), on the grid or, rarely, stochastically rounded alone.

Safety: the LoRA file stays the shipped file until a complete dense file has been written NEXT TO the upload folder (never in
it: a killed write must not be uploaded), re-read and verified (every module: amax == 448 * 2^k, every element on the grid,
d and W' exact in fp16, no NaN code; every key consumed by comfy.lora.load_lora against the live model's key map); only then
os.replace. Any exception, too little time (DENSE_EST_S before the hard stop = the validator's kill - DENSE_MARGIN_S) or disk
(the file + DENSE_FREE_GB): the LoRA file is kept and the run ends as without the knob.
"""
import json
import math
import os
import struct
import time
from pathlib import Path

PREFIX = "diffusion_model."
KMIN, KMAX = -15, 0      # 1/scale must stay <= the fp16 max in quant_ops.py (2^15 fits, 2^16 is clamped); weights < 448
NCAND = 4096             # largest-|u| certain candidates kept for the pin
M32 = 0xFFFFFFFF
CHUNK = 1 << 23          # elements per row chunk on the device


# ------------------------------------------------------------------ e4m3 grid (torch, float32 values; exact)
def e4m3_round(x):
    """RTNE onto the e4m3 grid, saturating at +-448 (torch's fp8 cast is RTNE)."""
    import torch
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)


def is_e4m3(y):
    return (y.abs() <= 448.0) & (e4m3_round(y) == y)


def ulp_e4m3(ax):
    import torch
    _, e = torch.frexp(ax)
    ue = torch.where(((e - 1) >= -6) & (ax > 0), e - 4, torch.full_like(e, -9))
    return torch.ldexp(torch.ones_like(ax), ue)


def floor_ceil(x):
    """grid neighbours lo <= x <= hi (|x| <= 448)."""
    import torch
    ax = x.abs()
    _, e = torch.frexp(ax)
    ue = torch.where(((e - 1) >= -6) & (ax > 0), e - 4, torch.full_like(e, -9))
    sc = torch.ldexp(ax, -ue)
    lo_a, hi_a = torch.ldexp(sc.floor(), ue), torch.ldexp(sc.ceil(), ue).clamp(max=448.0)
    neg = x < 0
    return torch.where(neg, -hi_a, lo_a), torch.where(neg, -lo_a, hi_a)


# ------------------------------------------------------------------ counter-hash uniforms (identical to the converter)
def _mul32(x, c):
    return ((x * (c & 0xFFFF)) + (((x * (c >> 16)) & 0xFFFF) << 16)) & M32


def hash32(x):
    x = x & M32
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def uniforms(seed, mod_id, idx):
    """u in [0, 1) (24 bits) from the global row-major element index idx (int64 tensor) of module mod_id."""
    import torch
    h = hash32(torch.tensor(mod_id, dtype=torch.int64) ^ hash32(torch.tensor(seed & M32, dtype=torch.int64))).to(idx.device)
    return (hash32(idx ^ h) >> 8).to(torch.float64) / float(1 << 24)


# ------------------------------------------------------------------ the trainer's W0 + certainty
def cache_w0(q8, s, cert_out=True):
    """q8: per-channel fp8 rows (float8 tensor, device), s: their scales (float32 [rows] or scalar). Returns W0 (float32,
    RTNE of fp32(q * s)), cert (bool), dmax (float32: the largest possible |W0_eval - W0| of an uncertain element)."""
    import torch
    qc = q8.to(torch.float32)
    s2 = s.reshape(-1, 1).to(torch.float32) if s.numel() > 1 else s.reshape(1, 1).to(torch.float32)
    wc = qc * s2
    wcl = wc.clamp(-448.0, 448.0)
    w0 = e4m3_round(wcl)
    if not cert_out:
        return w0, None, None
    eps = s2.to(torch.float64) * ulp_e4m3(qc.abs()).to(torch.float64) / 2.0
    wcd = wcl.to(torch.float64)
    lo_c, hi_c = floor_ceil(wcl)
    lo2, _ = floor_ceil(torch.nextafter(lo_c, torch.full_like(lo_c, -math.inf)).clamp(-448.0, 448.0))
    _, hi2 = floor_ceil(torch.nextafter(hi_c, torch.full_like(hi_c, math.inf)).clamp(-448.0, 448.0))
    lo_c, hi_c, lo2, hi2 = (t.to(torch.float64) for t in (lo_c, hi_c, lo2, hi2))
    mid = (lo_c + hi_c) / 2.0
    on = lo_c == hi_c
    b_lo = torch.where(on | (wcd < mid), (lo2 + lo_c) / 2.0, mid)
    b_hi = torch.where(on | (wcd >= mid), (hi_c + hi2) / 2.0, mid)
    cert = (wcd - eps > b_lo) & (wcd + eps < b_hi)
    aw = w0.abs()
    dmax = ulp_e4m3(aw + ulp_e4m3(aw))
    return w0, cert, dmax


def w0_exact(q8, s):
    """T118 (15 Oct, IO_FIX v2): the evaluator's own scale-1 W0 bit for bit. The Comfy-Org file is RTNE_e4m3(bf16(fp32(q * s)))
    of the task folder's per-channel fp8 transformer (dequantized to bf16, then cast to fp8 at scale 1): 0 mismatches over every
    
    
    (all of them 'uncertain' there). q8 per-channel fp8 rows, s their scales (float32 [rows] or scalar); float32 out."""
    import torch
    qc = q8.to(torch.float32)
    s2 = s.reshape(-1, 1).to(torch.float32) if s.numel() > 1 else s.reshape(1, 1).to(torch.float32)
    return e4m3_round((qc * s2).clamp(-448.0, 448.0).to(torch.bfloat16).to(torch.float32))


def choose_k(amax):
    if amax <= 0:
        return KMIN
    k = math.ceil(math.log2(amax / 448.0))
    while 448.0 * 2.0 ** (k - 1) >= amax:
        k -= 1
    while 448.0 * 2.0 ** k < amax:
        k += 1
    return max(KMIN, min(KMAX, k))


def round_feasible(x, w0s, mode, ufun, w0d=None):
    """x = u / 2^k, w0s = W0 / 2^k (float32, same shape). q on the e4m3 grid, |q| <= 448, (q - w0s) e4m3.
    w0d (v2b, 16 Oct): the base of the hyb dead zone ('rounding gives the base back although dW != 0') = the TRAINING base u was
    built on (cache_w0); None = w0s (as before). Feasibility stays against w0s (the evaluator's W0)."""
    import torch
    if w0d is None:
        w0d = w0s
    if mode in ("rtn", "hyb"):
        q = e4m3_round(x)
        if mode == "hyb":
            dz = (q == w0d) & (x != w0d)
            if bool(dz.any()):
                xd = x[dz].clamp(-448.0, 448.0)
                lo, hi = floor_ceil(xd)
                span = hi - lo
                p = torch.where(span > 0, (xd - lo).to(torch.float64) / torch.where(span > 0, span, torch.ones_like(span)).to(torch.float64),
                                torch.zeros_like(xd, dtype=torch.float64))
                q[dz] = torch.where(ufun(dz) < p, hi, lo)
    elif mode == "sr":
        xc = x.clamp(-448.0, 448.0)
        lo, hi = floor_ceil(xc)
        span = hi - lo
        p = torch.where(span > 0, (x - lo).to(torch.float64) / torch.where(span > 0, span, torch.ones_like(span)).to(torch.float64),
                        torch.zeros_like(x, dtype=torch.float64))
        q = torch.where(ufun(torch.ones_like(x, dtype=torch.bool)).reshape(x.shape) < p, hi, lo)
    else:
        raise ValueError(f"dense mode {mode!r}")
    ok = is_e4m3(q - w0s)
    nbad = int((~ok).sum())
    if nbad == 0:
        return q, 0, 0
    idx = ~ok
    xb, wb, qb = x[idx], w0s[idx], q[idx]
    d = (xb - wb).clamp(-448.0, 448.0)
    dlo, dhi = floor_ceil(d)
    ul = ulp_e4m3(qb.abs())
    cands = [wb, wb + e4m3_round(d), wb + dlo, wb + dhi, qb - ul, qb + ul, qb - 2 * ul, qb + 2 * ul]
    f0 = is_e4m3(cands[0]) & is_e4m3(cands[0] - wb)
    best, besterr = cands[0], torch.where(f0, (cands[0] - xb).abs(), torch.full_like(xb, math.inf))
    for c in cands[1:]:
        feas = is_e4m3(c) & is_e4m3(c - wb)
        err = torch.where(feas, (c - xb).abs(), torch.full_like(xb, math.inf))
        better = err < besterr
        best = torch.where(better, c, best)
        besterr = torch.where(better, err, besterr)
    q[idx] = best
    return q, nbad, int((best != qb).sum())


# ------------------------------------------------------------------ streaming safetensors writer
def _header(entries):
    h, off = {}, 0
    for k, dt, sh, nb in entries:
        h[k] = {"dtype": dt, "shape": list(sh), "data_offsets": [off, off + nb]}
        off += nb
    h["__metadata__"] = {"format": "pt"}
    js = json.dumps(h, separators=(",", ":")).encode()
    js += b" " * ((8 - (8 + len(js)) % 8) % 8)
    base = 8 + len(js)
    return struct.pack("<Q", len(js)) + js, {k: base + h[k]["data_offsets"][0] for k, *_ in entries}, base + off


class DenseSkip(Exception):
    """the dense file is not attempted / abandoned; the LoRA file stays."""


def _mkey(m):
    """."""
    parts = m.split(".", 2)
    if len(parts) == 3 and parts[0] == "layers" and parts[1].isdigit():
        return (int(parts[1]), parts[2])
    return (10 ** 6, m)


def lora_modules(lora_keys):
    mods = sorted({k[len(PREFIX):-len(".lora_up.weight")] for k in lora_keys if k.startswith(PREFIX) and k.endswith(".lora_up.weight")},
                  key=_mkey)
    other = [k for k in lora_keys if not (k.startswith(PREFIX) and k.endswith((".lora_up.weight", ".lora_down.weight", ".alpha")))]
    if other:
        raise DenseSkip(f"the shipped file has non-LoRA keys ({other[:2]})")
    return mods


def convert(lora_path, cache_files, out_path, mode, seed, device, hard_stop, log, chunk=CHUNK, extra=None):
    """The dense file of lora_path (the shipped LoRA) at out_path. cache_files: {tensor name: path} of the task folder's
    transformer. Raises DenseSkip / anything on failure (the caller keeps the LoRA). Returns stats.
    extra (G56CT_IO_FIX, T118): {module: (k, d uint8 codes [o, i])} of io_fix_build, written as LoKr pairs after the LoRA
    modules (None: the file is byte for byte as before).
    v2b (16 Oct, T118 sec. 21): W' = the training base cache_w0 (v1, = ideogram4_state_dict evalq16) + dW rounded onto the grid as
    before, but made feasible against and coded relative to the EVALUATOR's W0 = w0_exact (bit for bit for the tournament folder),
    so the evaluator holds W' itself instead of W' +- 1 scale-1 ulp at the ~1.1% elements where the single rounding differs
    
    sits on a certain element (there every W0 agrees) and |W'| + dmax <= 448 * 2^k keeps the evaluator's scale whatever an
    unknown base's rounding is (its worst case = v1). stats["w0fix"] = elements where w0_exact != cache_w0."""
    import torch
    from safetensors import safe_open
    t0 = time.time()
    with safe_open(str(lora_path), framework="pt", device="cpu") as fl:
        lkeys = list(fl.keys())
    mods = lora_modules(lkeys)
    handles = {}

    def cache_tensor(name):
        p = cache_files[name]
        if p not in handles:
            handles[p] = safe_open(str(p), framework="pt", device="cpu")
        return handles[p].get_tensor(name)

    shapes = {}
    for m in mods:
        if m + ".weight" not in cache_files or m + ".weight_scale" not in cache_files:
            raise DenseSkip(f"{m}: not in the task folder's transformer")
    entries = [(PREFIX + m + ".lokr_w1", "F32", [1, 1], 4) for m in mods]
    fl = safe_open(str(lora_path), framework="pt", device="cpu")
    for m in mods:
        up = fl.get_slice(PREFIX + m + ".lora_up.weight").get_shape()
        dn = fl.get_slice(PREFIX + m + ".lora_down.weight").get_shape()
        shapes[m] = (up[0], dn[1])
        entries.append((PREFIX + m + ".lokr_w2", "F8_E4M3", [up[0], dn[1]], up[0] * dn[1]))
    xmods = sorted(extra, key=_mkey) if extra else []
    if set(xmods) & set(mods):
        raise DenseSkip(f"extra modules overlap the LoRA modules: {sorted(set(xmods) & set(mods))[:2]}")
    for m in xmods:  # G56CT_IO_FIX (T118): the io LoKr pairs after the LoRA modules
        o_, i_ = (int(v) for v in extra[m][1].shape)
        entries.append((PREFIX + m + ".lokr_w1", "F32", [1, 1], 4))
        entries.append((PREFIX + m + ".lokr_w2", "F8_E4M3", [o_, i_], o_ * i_))
    head, offs, end = _header(entries)
    stats = {"modules": len(mods), "k": {}, "clamped": 0, "uncertain": 0, "infeasible": 0, "moved": 0, "pin_zero": 0, "w0fix": 0}
    xfac = 1.0 + (sum(int(extra[m][1].numel()) for m in xmods) / max(1, sum(o_ * i_ for o_, i_ in shapes.values()))
                  if xmods else 0.0)  # G56CT_IO_FIX: the io pairs' write + verify in the projected end below
    with open(out_path, "wb") as f:
        f.write(head)
        f.truncate(end)
        for n_m, m in enumerate(mods):
            if time.time() > hard_stop:
                raise DenseSkip(f"hard stop reached at module {n_m}/{len(mods)}")
            o, i = shapes[m]
            q8 = cache_tensor(m + ".weight").to(device)
            sc = cache_tensor(m + ".weight_scale").to(device)
            up = fl.get_tensor(PREFIX + m + ".lora_up.weight").to(device, torch.float32)
            dn = fl.get_tensor(PREFIX + m + ".lora_down.weight").to(device, torch.float32)
            ak = PREFIX + m + ".alpha"
            a = float(fl.get_tensor(ak).reshape(-1)[0]) / dn.shape[0] if ak in lkeys else 1.0
            ch = max(1, chunk // i)
            # pass 1: amax, certain candidates, the certain zero with the largest |u|
            amax, cand, certz = 0.0, [], None
            for r0 in range(0, o, ch):
                r1 = min(o, r0 + ch)
                w0, cert, _ = cache_w0(q8[r0:r1], sc[r0:r1] if sc.numel() > 1 else sc)
                dw = up[r0:r1] @ dn
                if a != 1.0:
                    dw = dw * a
                u = w0 + dw
                au = u.abs()
                amax = max(amax, float(au.max()))
                flat, cf = au.reshape(-1), cert.reshape(-1)
                fc = torch.where(cf, flat, torch.full_like(flat, -1.0))
                kk = min(NCAND, fc.numel())
                tv, ti = torch.topk(fc, kk)
                keep = tv >= 0
                for v_, t_ in zip(tv[keep].tolist(), ti[keep].tolist()):
                    cand.append((v_, r0 + t_ // i, t_ % i))
                zc = torch.where(cf & (w0.reshape(-1) == 0), flat, torch.full_like(flat, -1.0))
                zv, zi = torch.max(zc, 0)
                if float(zv) >= 0:  # torch.max: the first maximal index; across chunks the earliest one wins ties
                    zi = int(torch.nonzero(zc == zv)[0][0])
                    if certz is None or float(zv) > certz[0]:
                        certz = (float(zv), r0 + zi // i, zi % i)
                del w0, cert, dw, u, au
            cand.sort(key=lambda z: (-z[0], z[1], z[2]))   # ties: the first in row-major order (= the converter)
            cand = cand[:NCAND]
            k = choose_k(amax)
            s_ = 2.0 ** k

            def row(r):
                w0r, _, _ = cache_w0(q8[r:r + 1], sc[r:r + 1] if sc.numel() > 1 else sc, cert_out=False)
                dwr = up[r:r + 1] @ dn
                if a != 1.0:
                    dwr = dwr * a
                return w0r[0], (w0r + dwr)[0]
            pin = None
            for _, r, c in cand[:64]:
                w0r, ur = row(r)
                w0v, uv = float(w0r[c]), float(ur[c])
                sg = 1.0 if uv >= 0 else -1.0
                if bool(is_e4m3(torch.tensor([sg * 448.0 - w0v / s_], dtype=torch.float32))[0]):
                    pin = (r, c, sg)
                    break
            if pin is None and certz is not None:
                _, ur = row(certz[1])
                pin = (certz[1], certz[2], 1.0 if float(ur[certz[2]]) >= 0 else -1.0)
                stats["pin_zero"] += 1
            if pin is None:
                raise DenseSkip(f"{m}: no certain pin location")
            # pass 2: round, clamp uncertain elements, pin, write
            inv = 1.0 / s_
            for r0 in range(0, o, ch):
                r1 = min(o, r0 + ch)
                w0, cert, dmax = cache_w0(q8[r0:r1], sc[r0:r1] if sc.numel() > 1 else sc)
                w0x = w0_exact(q8[r0:r1], sc[r0:r1] if sc.numel() > 1 else sc)  # v2b: the evaluator's W0
                stats["w0fix"] += int((w0x != w0).sum())
                dw = up[r0:r1] @ dn
                if a != 1.0:
                    dw = dw * a
                u = w0 + dw   # the target: the training base (v1) + dW
                x, w0s = u * inv, w0x * inv

                def ufun(mask, _r0=r0, _i=i, _m=n_m):
                    nz = torch.nonzero(mask, as_tuple=True)
                    rr, cc = nz[0].to(torch.int64), nz[1].to(torch.int64)
                    return uniforms(seed, _m, (rr + _r0) * _i + cc)
                q, nbad, nfar = round_feasible(x, w0s, mode, ufun, w0d=w0 * inv)  # v2b review: dead zone vs the training base
                stats["infeasible"] += nbad
                stats["moved"] += nfar
                lim = 448.0 - dmax * inv
                over = (~cert) & (q.abs() > lim)
                stats["uncertain"] += int((~cert).sum())
                if bool(over.any()):
                    stats["clamped"] += int(over.sum())
                    qo = q[over]
                    cap = floor_ceil(lim[over].clamp(min=0.0))[0]
                    nq = torch.copysign(torch.minimum(qo.abs(), cap), qo)
                    w0o = w0s[over]
                    for _ in range(16):
                        bad = ~is_e4m3(nq - w0o)
                        if not bool(bad.any()):
                            break
                        aa = nq[bad].abs()
                        nq[bad] = torch.copysign(floor_ceil(torch.nextafter(aa, torch.zeros_like(aa)))[0], nq[bad])
                    if bool((~is_e4m3(nq - w0o)).any()):
                        raise DenseSkip(f"{m}: an uncertain element cannot be clamped feasibly")
                    q[over] = nq
                if r0 <= pin[0] < r1:
                    q[pin[0] - r0, pin[1]] = pin[2] * 448.0
                if float(q.abs().max()) > 448.0:
                    raise DenseSkip(f"{m}: |q| > 448")
                dq = q - w0s
                if not bool(is_e4m3(dq).all()):  # v2b review: never write an infeasible code (verify would refuse it later)
                    raise DenseSkip(f"{m}: infeasible d left after rounding")
                b = dq.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy().tobytes()
                f.seek(offs[PREFIX + m + ".lokr_w2"] + r0 * i)
                f.write(b)
                del w0, w0x, cert, dmax, dw, u, x, w0s, q, dq
            f.seek(offs[PREFIX + m + ".lokr_w1"])
            f.write(struct.pack("<f", s_))
            stats["k"][m] = k
            el = time.time() - t0
            if t0 + el / (n_m + 1) * len(mods) * 1.5 * xfac > hard_stop and n_m + 1 < len(mods):  # write + verify ~ 1.5 x the write
                raise DenseSkip(f"projected end after the hard stop ({el:.0f}s for {n_m + 1}/{len(mods)} modules)")
            del q8, sc, up, dn
        for m in xmods:  # G56CT_IO_FIX (T118): precomputed codes, written verbatim
            k_, d_ = extra[m]
            f.seek(offs[PREFIX + m + ".lokr_w1"])
            f.write(struct.pack("<f", 2.0 ** int(k_)))
            f.seek(offs[PREFIX + m + ".lokr_w2"])
            f.write(d_.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        if xmods:
            stats["extra"] = len(xmods)
        f.flush()
        os.fsync(f.fileno())
    stats["seconds"] = round(time.time() - t0, 1)
    try:
        log(f"dense: block d against the evaluator's exact W0 (v2b): w0fix {stats['w0fix']} of the {len(mods)} block modules' elements")
    except Exception:  # noqa: BLE001  (a logging problem never fails the pass)
        pass
    return stats


def verify(path, lora_path, cache_files, device, key_map=None, chunk=CHUNK, hard_stop=None, extra=None):
    """Re-read the dense file: one lokr pair per module of the LoRA file and nothing else; every module amax == 448 * 2^k
    (k in [KMIN, KMAX]) on the trainer's W0, every element on the 2^k grid, d and W' exact in fp16, no NaN code; with
    key_map (the live model's comfy.lora.model_lora_keys_unet): every key consumed by comfy.lora.load_lora. The evaluator-side
    invariants of the trainer form, re-derived from the cache: at least one element at 448 * 2^k has a certain W0 (the pin),
    and every uncertain element keeps |W'| + dmax <= 448 * 2^k (its W0 may differ by dmax in the evaluator).
    hard_stop: time.time() limit, checked per module (DenseSkip).
    extra (G56CT_IO_FIX, T118): the io modules convert(extra=...) added - their lokr pairs are expected too and pass the very
    same per-module checks (None: as before)."""
    import torch
    from safetensors import safe_open
    with safe_open(str(lora_path), framework="pt", device="cpu") as fl:
        mods = lora_modules(list(fl.keys()))
    xmods = sorted(extra, key=_mkey) if extra else []
    with safe_open(str(path), framework="pt", device="cpu") as fd:
        keys = set(fd.keys())
        want = {PREFIX + m + s for m in list(mods) + xmods for s in (".lokr_w1", ".lokr_w2")}
        if keys != want:
            raise DenseSkip(f"key set differs: missing {sorted(want - keys)[:2]} extra {sorted(keys - want)[:2]}")
        if key_map is not None:
            _key_check(keys, key_map, len(mods) + len(xmods))
        handles = {}
        for m in list(mods) + xmods:
            _check_lokr(fd, m, cache_files, handles, device, chunk, hard_stop, exact=True, io=m in xmods)  # v2b: blocks too
        for m in xmods:
            _same_io(fd, m, extra[m])
    return len(mods) + len(xmods)


def _same_io(fd, m, ref):
    """G56CT_IO_FIX: the written io pair == the trainer's in-memory (k, d codes) byte for byte (DenseSkip otherwise) - the
    grid / pin checks alone accept any e4m3 code where W0 == 0, most of these matrices."""
    import torch
    k_, d_ = ref
    w1 = fd.get_tensor(PREFIX + m + ".lokr_w1")
    w2 = fd.get_tensor(PREFIX + m + ".lokr_w2")
    if float(w1.reshape(-1)[0]) != 2.0 ** int(k_) or not torch.equal(w2.contiguous().view(torch.uint8), d_.contiguous().view(torch.uint8)):
        raise DenseSkip(f"{m}: the written io pair differs from the trainer's codes")


def _key_check(keys, key_map, n_mods):
    """every key consumed by comfy.lora.load_lora against the live key map, one patch per module (DenseSkip otherwise)."""
    import logging
    import torch
    import comfy.lora
    missing = []

    class H(logging.Handler):
        def emit(self, r):
            msg = r.getMessage()
            if msg.startswith(("lora key not loaded:", "NOT LOADED ")):
                missing.append(msg)
    h = H()
    logging.getLogger().addHandler(h)
    try:
        ph = {k: torch.zeros(1) for k in keys}  # placeholders: the key check reads names only
        patches = comfy.lora.load_lora(ph, key_map)
    finally:
        logging.getLogger().removeHandler(h)
    if missing or len(patches) != n_mods:
        raise DenseSkip(f"key check: {len(missing)} unmatched, {len(patches)} patches for {n_mods} modules")


def _check_lokr(fd, m, cache_files, handles, device, chunk, hard_stop, exact=False, io=False):
    """the per-module invariants of verify() for module m of the open file fd (DenseSkip on any violation). exact (G56CT_IO_FIX
    v2, T118): the evaluator's W0 is w0_exact (the io modules, whose d io_fix_build computes against it); certainty and dmax
    stay cache_w0's (the pin / clamp invariants io_fix_build / convert keep) unless io and the matrix is io_known (v3: all certain,
    dmax 0). v2b (16 Oct): verify() checks the block modules with exact=True too (convert codes them against w0_exact)."""
    import torch
    from safetensors import safe_open
    if hard_stop is not None and time.time() > hard_stop:
        raise DenseSkip(f"hard stop reached while verifying ({m})")
    w1 = fd.get_tensor(PREFIX + m + ".lokr_w1")
    if w1.dtype != torch.float32 or list(w1.shape) != [1, 1]:
        raise DenseSkip(f"{m}: lokr_w1 not F32 [1,1]")
    s_ = float(w1[0, 0])
    k = math.log2(s_) if s_ > 0 else None
    if k is None or k != int(k) or not (KMIN <= k <= KMAX):
        raise DenseSkip(f"{m}: lokr_w1 {s_!r} is not 2^k, k in [{KMIN}, {KMAX}]")
    w2 = fd.get_tensor(PREFIX + m + ".lokr_w2")
    if w2.dtype != torch.float8_e4m3fn:
        raise DenseSkip(f"{m}: lokr_w2 not F8_E4M3")
    p = cache_files[m + ".weight"]
    if p not in handles:
        handles[p] = safe_open(str(p), framework="pt", device="cpu")
    q8 = handles[p].get_tensor(m + ".weight")
    sc = handles[p].get_tensor(m + ".weight_scale")
    o, i = q8.shape
    if list(w2.shape) != [o, i]:
        raise DenseSkip(f"{m}: lokr_w2 shape {list(w2.shape)} != {[o, i]}")
    ch = max(1, chunk // i)
    known = bool(exact) and bool(io) and io_known(q8, sc)   # io modules only (no 9 GB hash of the blocks)
    amax, pin_certain, smax = 0.0, False, 448.0 * s_
    for r0 in range(0, o, ch):
        r1 = min(o, r0 + ch)
        wb = w2[r0:r1].view(torch.uint8).to(device)
        if bool(((wb & 0x7F) == 0x7F).any()):
            raise DenseSkip(f"{m}: NaN code in lokr_w2")
        d = w2[r0:r1].to(device).to(torch.float32) * s_
        w0, cert, dmax = cache_w0(q8[r0:r1].to(device), (sc[r0:r1] if sc.numel() > 1 else sc).to(device))
        if exact:
            w0 = w0_exact(q8[r0:r1].to(device), (sc[r0:r1] if sc.numel() > 1 else sc).to(device))
            if known:  # v3 (io_known): the evaluator's W0 is known bit for bit -> every element certain, no dmax margin
                cert, dmax = torch.ones_like(cert), torch.zeros_like(dmax)
        t = w0 + d
        ta = t.abs()
        pin_certain = pin_certain or bool(((ta == smax) & cert).any())
        if bool(((~cert) & (ta + dmax > smax)).any()):
            raise DenseSkip(f"{m}: an uncertain element may exceed 448 * 2^k in the evaluator")
        for arr in (d, t):
            if not bool((arr.to(torch.float16).to(torch.float32) == arr).all()):
                raise DenseSkip(f"{m}: not exact in fp16")
        if not bool(is_e4m3(t / s_).all()):
            raise DenseSkip(f"{m}: elements off the 2^k e4m3 grid")
        amax = max(amax, float(t.abs().max()))
    if amax != 448.0 * s_:
        raise DenseSkip(f"{m}: amax {amax!r} != 448 * 2^{int(k)}")
    if not pin_certain:
        raise DenseSkip(f"{m}: no element at 448 * 2^{int(k)} has a certain W0 (the pin)")


# ------------------------------------------------------------------ G56CT_IO_FIX (T118): the 7 non-block fp8 matrices
IO_FIX_MODULES = ("input_proj", "final_layer.linear", "final_layer.adaln_modulation", "adaln_proj", "llm_cond_proj",
                  "t_embedding.mlp_in", "t_embedding.mlp_out")


# G56CT_IO_FIX v3 (T118, 16 Oct): sha256(weight bytes + weight_scale bytes, as stored) of the 7 io matrices of the tournament folder
# gradients-io-tournaments/ideogram-4-fp8. For these, w0_exact equals the evaluator's base (Comfy-Org scale-1 file) at every element
# (verified offline over all 211 fp8 matrices of that folder, 0 mismatches), so they are built in the research form (v3: the best
# pin, grid auto, every element certain). Any other matrix keeps the v2 form (a cache_w0-certain pin, HI grid, the uncertain clamp):
# a folder we have not verified cannot move the evaluator's scale through a pin whose W0 it does not determine.
IO_FIX_KNOWN = frozenset((
    "3087f84204af93aab6dd761fcaeef4a570df83d4c26b2eb5a3bbf2fc21adb486",  # input_proj
    "ba70e90792c4758b47d88b71b891a7ccc665f99bb0b5d385ebdaf067b4ba3228",  # final_layer.linear
    "dd79d30c9c34bce597517004f8eb1fcb3caecec92fee8dfa6df7eb0f3c6e3cc0",  # final_layer.adaln_modulation
    "68404b72dab437e5865bdd614f03561a63a133e5d7aaea422d51ef2043fd338d",  # adaln_proj
    "b54cb82900d14e8337c96a29268a3b5495fac6670fcae4bb0d4e5e77f84dfdf5",  # llm_cond_proj
    "8afafc451f3a1e8830d39d38b99bed1f941da83541c9d030166e655ca61e6fb8",  # t_embedding.mlp_in
    "629176ab5e8c54a8613b404b8df40a1ce9a0d2a2d8f1bea21626f4f0ca978ee3",  # t_embedding.mlp_out
))


def io_known(q8, sc):
    """True when (q8, sc) is one of the IO_FIX_KNOWN matrices: sha256 over the raw bytes of the per-row fp8 weight, then of its
    scales (as stored; the trainer and verify() read them from the task folder unchanged)."""
    import hashlib
    import torch
    h = hashlib.sha256()
    for t in (q8, sc):
        b = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        h.update(memoryview(b.numpy()))
    return h.hexdigest() in IO_FIX_KNOWN


def io_fix_build(q8, sc, device, chunk=CHUNK):
    """G56CT_IO_FIX (T118): one non-block fp8 matrix of the task folder (q8 per-row fp8, sc its scales) as a LoKr pair the
    evaluator re-quantizes onto a 2^k grid. A KNOWN matrix (io_known) is built in the research form v3 (_io_fix_build_v3: best pin,
    grid auto, no uncertain element), any other in the v2 form (_io_fix_build_v2: certain pin, HI grid, uncertain clamp); both
    compute d against w0_exact. Returns (L, d codes, k, stats) with stats["mode"] = "v3" | "v2". Raises DenseSkip on any problem."""
    if io_known(q8, sc):
        L, d8, k, st = _io_fix_build_v3(q8, sc, device, chunk)
        st["mode"] = "v3"
    else:
        L, d8, k, st = _io_fix_build_v2(q8, sc, device, chunk)
        st["mode"] = "v2"
    return L, d8, k, st


def _io_fix_build_v2(q8, sc, device, chunk=CHUNK):
    """G56CT_IO_FIX (T118): one non-block fp8 matrix of the task folder (q8 per-row fp8, sc its scales) on the evaluator's
    side as a LoKr pair the evaluator re-quantizes onto a 2^k grid, exactly like a dense block module but with NO training
    delta: the target u = fp32(q8 * s) (the task folder's own per-row weight; the evaluator's scale-1 file re-quantized it:
    39% zeros in adaln_proj, 75% in final_layer.linear). Trainer form = the offline converter's
    
    certainty / dmax = cache_w0's, W0 = w0_exact (v2, 15 Oct: the evaluator's scale-1 value bit for bit; v1 used cache_w0's
    
    
    e4m3), uncertain elements kept at |q| <= 448 - dmax / 2^k, the pin = the first of the 64 largest-|u| CERTAIN elements whose
    sign*448 - W0/2^k is e4m3 (else the certain W0 == 0 element with the largest |u|) set to sign*448.
    Returns (L float32 [o, i] on the CPU = q * 2^k: what the evaluator holds, d uint8 codes [o, i] on the CPU =
    e4m3(q - W0/2^k), k, stats; stats["w0fix"] = elements where w0_exact != cache_w0). Raises DenseSkip on any problem."""
    import torch
    o, i = (int(v) for v in q8.shape)
    s2 = sc.to(device=device, dtype=torch.float32)
    srow = (lambda r0, r1: s2.reshape(-1)[r0:r1].reshape(-1, 1)) if s2.numel() > 1 else (lambda r0, r1: s2.reshape(1, 1))
    q8 = q8.to(device)
    ch = max(1, chunk // i)
    amax, cand, certz = 0.0, [], None
    for r0 in range(0, o, ch):
        r1 = min(o, r0 + ch)
        w0, cert, _ = cache_w0(q8[r0:r1], s2.reshape(-1)[r0:r1] if s2.numel() > 1 else s2)
        u = q8[r0:r1].to(torch.float32) * srow(r0, r1)
        au = u.abs()
        amax = max(amax, float(au.max()))
        flat, cf = au.reshape(-1), cert.reshape(-1)
        fc = torch.where(cf, flat, torch.full_like(flat, -1.0))
        kk = min(NCAND, fc.numel())
        tv, ti = torch.topk(fc, kk)
        keep = tv >= 0
        for v_, t_ in zip(tv[keep].tolist(), ti[keep].tolist()):
            cand.append((v_, r0 + t_ // i, t_ % i))
        zc = torch.where(cf & (w0.reshape(-1) == 0), flat, torch.full_like(flat, -1.0))
        zv, _ = torch.max(zc, 0)
        if float(zv) >= 0:
            zi = int(torch.nonzero(zc == zv)[0][0])
            if certz is None or float(zv) > certz[0]:
                certz = (float(zv), r0 + zi // i, zi % i)
        del w0, cert, u, au
    cand.sort(key=lambda z: (-z[0], z[1], z[2]))
    cand = cand[:NCAND]
    k = choose_k(amax)
    s_ = 2.0 ** k
    pin = None
    for _, r, c in cand[:64]:
        w0r = w0_exact(q8[r:r + 1], s2.reshape(-1)[r:r + 1] if s2.numel() > 1 else s2)
        uv = float(q8[r, c].to(torch.float32) * srow(r, r + 1)[0, 0])
        sg = 1.0 if uv >= 0 else -1.0
        if bool(is_e4m3(torch.tensor([sg * 448.0 - float(w0r[0, c]) / s_], dtype=torch.float32))[0]):
            pin = (r, c, sg)
            break
    stats = {"k": k, "amax": amax, "uncertain": 0, "clamped": 0, "infeasible": 0, "moved": 0, "pin_zero": 0, "w0fix": 0}
    if pin is None and certz is not None:
        r = certz[1]
        uv = float(q8[r, certz[2]].to(torch.float32) * srow(r, r + 1)[0, 0])
        pin = (r, certz[2], 1.0 if uv >= 0 else -1.0)
        stats["pin_zero"] = 1
    if pin is None:
        raise DenseSkip("io_fix: no certain pin location")
    inv = 1.0 / s_
    L = torch.empty((o, i), dtype=torch.float32)
    d8 = torch.empty((o, i), dtype=torch.uint8)
    for r0 in range(0, o, ch):
        r1 = min(o, r0 + ch)
        w0, cert, dmax = cache_w0(q8[r0:r1], s2.reshape(-1)[r0:r1] if s2.numel() > 1 else s2)
        w0x = w0_exact(q8[r0:r1], s2.reshape(-1)[r0:r1] if s2.numel() > 1 else s2)
        stats["w0fix"] += int((w0x != w0).sum())
        u = q8[r0:r1].to(torch.float32) * srow(r0, r1)
        x, w0s = u * inv, w0x * inv
        q, nbad, nfar = round_feasible(x, w0s, "rtn", None)
        stats["infeasible"] += nbad
        stats["moved"] += nfar
        lim = 448.0 - dmax * inv
        over = (~cert) & (q.abs() > lim)
        stats["uncertain"] += int((~cert).sum())
        if bool(over.any()):
            stats["clamped"] += int(over.sum())
            qo = q[over]
            cap = floor_ceil(lim[over].clamp(min=0.0))[0]
            nq = torch.copysign(torch.minimum(qo.abs(), cap), qo)
            w0o = w0s[over]
            for _ in range(16):
                bad = ~is_e4m3(nq - w0o)
                if not bool(bad.any()):
                    break
                aa = nq[bad].abs()
                nq[bad] = torch.copysign(floor_ceil(torch.nextafter(aa, torch.zeros_like(aa)))[0], nq[bad])
            if bool((~is_e4m3(nq - w0o)).any()):
                raise DenseSkip("io_fix: an uncertain element cannot be clamped feasibly")
            q[over] = nq
        if r0 <= pin[0] < r1:
            q[pin[0] - r0, pin[1]] = pin[2] * 448.0
        if float(q.abs().max()) > 448.0:
            raise DenseSkip("io_fix: |q| > 448")
        dq = q - w0s
        if not bool(is_e4m3(dq).all()):
            raise DenseSkip("io_fix: infeasible d left after rounding")
        L[r0:r1] = (q * s_).cpu()
        d8[r0:r1] = dq.to(torch.float8_e4m3fn).view(torch.uint8).cpu()
        del w0, w0x, cert, dmax, u, x, w0s, q, dq
    if not bool((L.to(torch.bfloat16).to(torch.float32) == L).all()):
        raise DenseSkip("io_fix: L not exact in bf16")
    return L, d8, k, stats


def _io_fix_build_v3(q8, sc, device, chunk=CHUNK):
    """io_fix_build for a KNOWN matrix (io_known: the tournament folder, w0_exact verified bit for bit there).
    v3 (16 Oct, T118 sec. 20) = the research form T118IOL / T118IOZA exactly (offline `t112_dense.py convert --mode zero
    --lift-cache <task transformer> --w0 comfy --pin best --grid auto`), now that the evaluator's W0 is known bit for bit
    (w0_exact, 0 mismatches over all 211 fp8 matrices of the task folder): no 'uncertain' element any more, so (1) the pin is
    the first of the 64 largest-|u| elements whose sign*448 - W0/2^k is e4m3 (v2: certain elements only - its pins sat 0.06-0.23
    from the per-row weight vs 0.007-0.11 for the research pins), no uncertain clamp; (2) grid auto: HI = smallest k with
    448*2^k >= amax (pin cost (448*2^k - amax)^2) or LO = k-1 with every |u| > 448*2^(k-1) clipped to it (cost: the summed
    squared excess), LO when its cost <= the HI pin cost and fewer than NCAND elements clip (final_layer.linear: LO, 1 clip).
    W0 = w0_exact; q = RTN(u / 2^k) made feasible ((q - W0/2^k) e4m3); pin fallback: the W0 == 0 element with the largest |u|.
    Returns (L float32 [o, i] on the CPU = q * 2^k: what the evaluator holds, d uint8 codes [o, i] on the CPU = e4m3(q - W0/2^k),
    k, stats; stats["w0fix"] = elements where w0_exact != cache_w0). Raises DenseSkip on any problem."""
    import torch
    o, i = (int(v) for v in q8.shape)
    s2 = sc.to(device=device, dtype=torch.float32)
    srow = (lambda r0, r1: s2.reshape(-1)[r0:r1].reshape(-1, 1)) if s2.numel() > 1 else (lambda r0, r1: s2.reshape(1, 1))
    srows = (lambda r0, r1: s2.reshape(-1)[r0:r1]) if s2.numel() > 1 else (lambda r0, r1: s2)
    q8 = q8.to(device)
    ch = max(1, chunk // i)
    amax, cand, zbest = 0.0, [], None
    for r0 in range(0, o, ch):
        r1 = min(o, r0 + ch)
        u = q8[r0:r1].to(torch.float32) * srow(r0, r1)
        au = u.abs()
        amax = max(amax, float(au.max()))
        flat = au.reshape(-1)
        kk = min(NCAND, flat.numel())
        tv, ti = torch.topk(flat, kk)
        for v_, t_ in zip(tv.tolist(), ti.tolist()):
            cand.append((v_, r0 + t_ // i, t_ % i))
        w0x = w0_exact(q8[r0:r1], srows(r0, r1))
        zc = torch.where(w0x.reshape(-1) == 0, flat, torch.full_like(flat, -1.0))
        zv, _ = torch.max(zc, 0)
        if float(zv) >= 0:
            zi = int(torch.nonzero(zc == zv)[0][0])
            if zbest is None or float(zv) > zbest[0]:
                zbest = (float(zv), r0 + zi // i, zi % i)
        del u, au, w0x
    cand.sort(key=lambda z: (-z[0], z[1], z[2]))   # ties: the first in row-major order (= t112_dense v6)
    cand = cand[:NCAND]

    def find_pin(scale, among):
        for _, r, c in among:
            w0v = float(w0_exact(q8[r:r + 1], srows(r, r + 1))[0, c])
            uv = float(q8[r, c].to(torch.float32) * srow(r, r + 1)[0, 0])
            sg = 1.0 if uv >= 0 else -1.0
            if bool(is_e4m3(torch.tensor([sg * 448.0 - w0v / scale], dtype=torch.float32))[0]):
                return (r, c, sg)
        return None
    k_hi = choose_k(amax)
    S_hi = 448.0 * 2.0 ** k_hi
    pin_hi = find_pin(2.0 ** k_hi, cand[:64])
    cost_hi = (S_hi - amax) ** 2
    lo = None
    if k_hi - 1 >= KMIN:
        S_lo = S_hi / 2.0
        clip = [z for z in cand if z[0] > S_lo]
        if 0 < len(clip) < NCAND:
            cost_lo = float(sum((z[0] - S_lo) ** 2 for z in clip))
            pin_lo = find_pin(2.0 ** (k_hi - 1), clip)
            if pin_lo is not None and (cost_lo <= cost_hi or pin_hi is None):
                lo = (pin_lo, len(clip))
    k, pin = (k_hi - 1, lo[0]) if lo is not None else (k_hi, pin_hi)
    s_ = 2.0 ** k
    stats = {"k": k, "grid": "lo" if lo is not None else "hi", "clip": lo[1] if lo is not None else 0, "amax": amax,
             "uncertain": 0, "clamped": 0, "infeasible": 0, "moved": 0, "pin_zero": 0, "w0fix": 0}
    if pin is None and zbest is not None:  # a W0 == 0 element: d = +-448 is always e4m3
        r = zbest[1]
        uv = float(q8[r, zbest[2]].to(torch.float32) * srow(r, r + 1)[0, 0])
        pin = (r, zbest[2], 1.0 if uv >= 0 else -1.0)
        stats["pin_zero"] = 1
    if pin is None:
        raise DenseSkip("io_fix: no pin location")
    inv = 1.0 / s_
    lmax = 0.0
    L = torch.empty((o, i), dtype=torch.float32)
    d8 = torch.empty((o, i), dtype=torch.uint8)
    for r0 in range(0, o, ch):
        r1 = min(o, r0 + ch)
        w0x = w0_exact(q8[r0:r1], srows(r0, r1))
        w0v1, _, _ = cache_w0(q8[r0:r1], srows(r0, r1), cert_out=False)
        stats["w0fix"] += int((w0x != w0v1).sum())
        u = q8[r0:r1].to(torch.float32) * srow(r0, r1)
        x, w0s = u * inv, w0x * inv
        q, nbad, nfar = round_feasible(x, w0s, "rtn", None)
        stats["infeasible"] += nbad
        stats["moved"] += nfar
        if r0 <= pin[0] < r1:
            q[pin[0] - r0, pin[1]] = pin[2] * 448.0
        if float(q.abs().max()) > 448.0:
            raise DenseSkip("io_fix: |q| > 448")
        dq = q - w0s
        if not bool(is_e4m3(dq).all()):
            raise DenseSkip("io_fix: infeasible d left after rounding")
        Lc = q * s_
        if not bool((Lc.to(torch.bfloat16).to(torch.float32) == Lc).all()):  # per chunk: no full-size temporaries
            raise DenseSkip("io_fix: L not exact in bf16")
        lmax = max(lmax, float(Lc.abs().max()))
        L[r0:r1] = Lc.cpu()
        d8[r0:r1] = dq.to(torch.float8_e4m3fn).view(torch.uint8).cpu()
        del w0x, w0v1, u, x, w0s, q, dq, Lc
    if lmax != 448.0 * s_:
        raise DenseSkip("io_fix: amax(L) != 448 * 2^k")
    return L, d8, k, stats


def io_lokr_tensors(extra):
    """{key: tensor} of the io LoKr pairs (lokr_w1 F32 [[2^k]], lokr_w2 F8_E4M3) for a safetensors writer."""
    import torch
    out = {}
    for m in sorted(extra, key=_mkey):
        k_, d_ = extra[m]
        out[PREFIX + m + ".lokr_w1"] = torch.tensor([[2.0 ** int(k_)]], dtype=torch.float32)
        out[PREFIX + m + ".lokr_w2"] = d_.contiguous().view(torch.float8_e4m3fn)
    return out


def verify_io_lora(path, lora_path, extra, cache_files, device, key_map=None, chunk=CHUNK, hard_stop=None):
    """G56CT_IO_FIX fallback file (the LoRA shipped as written + the io LoKr pairs): its keys are exactly the LoRA file's keys
    plus the io pairs, the LoRA tensors are byte for byte the LoRA file's, every io module passes verify()'s per-module checks,
    and with key_map every key is consumed by comfy.lora.load_lora (one patch per module). Returns the io module count."""
    import torch
    from safetensors import safe_open
    xmods = sorted(extra, key=_mkey)
    with safe_open(str(lora_path), framework="pt", device="cpu") as fl, safe_open(str(path), framework="pt", device="cpu") as fd:
        lkeys = set(fl.keys())
        mods = lora_modules(list(lkeys))
        if set(xmods) & set(mods):
            raise DenseSkip("io modules overlap the LoRA modules")
        keys = set(fd.keys())
        want = lkeys | {PREFIX + m + s for m in xmods for s in (".lokr_w1", ".lokr_w2")}
        if keys != want:
            raise DenseSkip(f"io+LoRA key set differs: missing {sorted(want - keys)[:2]} extra {sorted(keys - want)[:2]}")
        for k_ in lkeys:
            a, b = fl.get_tensor(k_), fd.get_tensor(k_)
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)):
                raise DenseSkip(f"io+LoRA file: {k_} differs from the LoRA file")
        if key_map is not None:
            _key_check(keys, key_map, len(mods) + len(xmods))
        handles = {}
        for m in xmods:
            _check_lokr(fd, m, cache_files, handles, device, chunk, hard_stop, exact=True, io=True)
            _same_io(fd, m, extra[m])
    return len(xmods)


# ------------------------------------------------------------------ G56CT_DENSE=gptq (T144, 18 Oct 2026; ideogram4 only; OFF by default)
# Activation-aware rounding (GPTQ, Frantar et al. 2022) of the SAME trainer-form target as convert(): pass 1 (amax, k, the certain
# pin), the training-base target u = cache_w0 + dW, the uncertain clamp, feasibility against and coding relative to w0_exact (v2b),
# the io pairs and the file layout are convert()'s; only the rounding of x = u / 2^k onto the grid changes. Column by column, in
# act-order (descending diag H), the error-compensated column takes the nearest FEASIBLE value (round_feasible 'rtn'), then the
# uncertain clamp, then the pin; its error is fed back to the later columns through U = the upper Cholesky factor of
# (H + damp * mean(diag H) I)^-1, H = E[x x^T] of the module's inputs (lazy batches of GPTQ_BLOCK columns). This minimises the
# module's OUTPUT error ||(W' - W*) x|| on the activations H was collected on instead of the weight error (bedah T139 on stage-1
#
# in groups: every per-column operation is elementwise and each module's lazy-batch update is its own torch.mm call, so the
# codes do not depend on the group size (gptq_check). verify() checks the written file like any dense file.
GPTQ_DAMP = 0.01   # relative dampening of H (x mean(diag H)); x10 / x100 retried when H + damp is not positive definite
GPTQ_BLOCK = 128   # columns per lazy batch


def gptq_factor(H, device, damp=GPTQ_DAMP):
    """H [i, i] (float32, any device) -> (perm int64 [i]: act-order = descending diag(H), stable; U float32 [i, i] on device: the
    upper Cholesky factor of (H + damp * mean(diag) I)^-1 in the permuted order; n dead inputs; the damping used). Inputs whose
    diag(H) <= 0 never fed the module: their diagonal becomes 1 (as GPTQ). Raises DenseSkip on a non-square / non-finite H or
    when H + damp is not positive definite at 1x, 10x and 100x the damping."""
    import torch
    if H.dim() != 2 or H.shape[0] != H.shape[1]:
        raise DenseSkip(f"gptq: H shape {tuple(H.shape)} is not square")
    Hd = H.to(device=device, dtype=torch.float64, copy=True)
    if not bool(torch.isfinite(Hd).all()):
        raise DenseSkip("gptq: non-finite H")
    dead = torch.nonzero(torch.diagonal(Hd) <= 0).flatten()
    if dead.numel():
        Hd[dead, dead] = 1.0
    mean = float(torch.diagonal(Hd).mean())
    perm = torch.argsort(torch.diagonal(Hd), descending=True, stable=True)
    Hp = Hd[perm][:, perm]
    del Hd
    for f in (1.0, 10.0, 100.0):
        A = Hp.clone()
        A.diagonal().add_(damp * f * mean)
        L, info = torch.linalg.cholesky_ex(A)
        del A
        if int(info) != 0:
            del L
            continue
        Hinv = torch.cholesky_inverse(L)
        del L
        S = (Hinv + Hinv.T) / 2.0
        del Hinv
        U, info2 = torch.linalg.cholesky_ex(S, upper=True)
        del S
        if int(info2) == 0 and bool(torch.isfinite(U).all()):
            return perm, U.to(torch.float32), int(dead.numel()), damp * f
        del U
    raise DenseSkip("gptq: H + damping is not positive definite (1x, 10x, 100x)")


def _gptq_pass1(q8, sc, up, dn, a, o, i, chunk):
    """convert()'s pass 1 for one module, statement for statement: amax, the certain candidates, the certain zero with the
    largest |u|, k and the pin (r, c, sign). Returns (k, pin, pin_zero)."""
    import torch
    ch = max(1, chunk // i)
    amax, cand, certz = 0.0, [], None
    for r0 in range(0, o, ch):
        r1 = min(o, r0 + ch)
        w0, cert, _ = cache_w0(q8[r0:r1], sc[r0:r1] if sc.numel() > 1 else sc)
        dw = up[r0:r1] @ dn
        if a != 1.0:
            dw = dw * a
        u = w0 + dw
        au = u.abs()
        amax = max(amax, float(au.max()))
        flat, cf = au.reshape(-1), cert.reshape(-1)
        fc = torch.where(cf, flat, torch.full_like(flat, -1.0))
        kk = min(NCAND, fc.numel())
        tv, ti = torch.topk(fc, kk)
        keep = tv >= 0
        for v_, t_ in zip(tv[keep].tolist(), ti[keep].tolist()):
            cand.append((v_, r0 + t_ // i, t_ % i))
        zc = torch.where(cf & (w0.reshape(-1) == 0), flat, torch.full_like(flat, -1.0))
        zv, zi = torch.max(zc, 0)
        if float(zv) >= 0:
            zi = int(torch.nonzero(zc == zv)[0][0])
            if certz is None or float(zv) > certz[0]:
                certz = (float(zv), r0 + zi // i, zi % i)
        del w0, cert, dw, u, au
    cand.sort(key=lambda z: (-z[0], z[1], z[2]))
    cand = cand[:NCAND]
    k = choose_k(amax)
    s_ = 2.0 ** k

    def row(r):
        w0r, _, _ = cache_w0(q8[r:r + 1], sc[r:r + 1] if sc.numel() > 1 else sc, cert_out=False)
        dwr = up[r:r + 1] @ dn
        if a != 1.0:
            dwr = dwr * a
        return w0r[0], (w0r + dwr)[0]
    pin = None
    for _, r, c in cand[:64]:
        w0r, ur = row(r)
        w0v, uv = float(w0r[c]), float(ur[c])
        sg = 1.0 if uv >= 0 else -1.0
        if bool(is_e4m3(torch.tensor([sg * 448.0 - w0v / s_], dtype=torch.float32))[0]):
            pin = (r, c, sg)
            break
    pin_zero = 0
    if pin is None and certz is not None:
        _, ur = row(certz[1])
        pin = (certz[1], certz[2], 1.0 if float(ur[certz[2]]) >= 0 else -1.0)
        pin_zero = 1
    if pin is None:
        raise DenseSkip("gptq: no certain pin location")
    return k, pin, pin_zero


def _gptq_targets(q8, sc, up, dn, a, o, i, k, chunk, out_x, out_w, out_l):
    """convert()'s pass-2 quantities of one module, per row chunk as convert() computes them, into the float32 [o, i] buffers:
    out_x = u / 2^k (u = cache_w0 + dW: the training base + the delta), out_w = w0_exact / 2^k (the evaluator's W0), out_l = the
    uncertain clamp limit 448 - dmax / 2^k (+inf at certain elements). Returns (w0fix, uncertain)."""
    import torch
    ch = max(1, chunk // i)
    inv = 1.0 / 2.0 ** k
    w0fix = nunc = 0
    for r0 in range(0, o, ch):
        r1 = min(o, r0 + ch)
        w0, cert, dmax = cache_w0(q8[r0:r1], sc[r0:r1] if sc.numel() > 1 else sc)
        w0x = w0_exact(q8[r0:r1], sc[r0:r1] if sc.numel() > 1 else sc)
        w0fix += int((w0x != w0).sum())
        dw = up[r0:r1] @ dn
        if a != 1.0:
            dw = dw * a
        u = w0 + dw
        out_x[r0:r1] = u * inv
        out_w[r0:r1] = w0x * inv
        lim = 448.0 - dmax * inv
        out_l[r0:r1] = torch.where(cert, torch.full_like(lim, math.inf), lim)
        nunc += int((~cert).sum())
        del w0, w0x, cert, dmax, dw, u, lim
    return w0fix, nunc


def _gptq_clamp(qo, limo, w0o):
    """convert()'s uncertain clamp for the elements qo whose |q| exceeds their limit limo: |q| down to the grid floor of the limit,
    then grid step by grid step (at most 16) until the code q - W0 / 2^k is e4m3. Raises DenseSkip when none is."""
    import torch
    cap = floor_ceil(limo.clamp(min=0.0))[0]
    nq = torch.copysign(torch.minimum(qo.abs(), cap), qo)
    for _ in range(16):
        bad = ~is_e4m3(nq - w0o)
        if not bool(bad.any()):
            break
        aa = nq[bad].abs()
        nq[bad] = torch.copysign(floor_ceil(torch.nextafter(aa, torch.zeros_like(aa)))[0], nq[bad])
    if bool((~is_e4m3(nq - w0o)).any()):
        raise DenseSkip("gptq: an uncertain element cannot be clamped feasibly")
    return nq


def gptq_round(X, W0S, LIMU, U, pins, block=GPTQ_BLOCK, hard_stop=None):
    """The GPTQ column loop for G modules of one shape at once. X, W0S, LIMU: float32 [G, o, i] in each module's permuted column
    order (X is the working copy and is consumed); U: float32 [G, i, i]; pins: G tuples (row, permuted column, sign) or None. Per column:
    q = round_feasible(column, 'rtn'), the uncertain clamp, the pin; err = (column - q) / U_jj fed back to the block's later
    columns, the block's errors to the columns after it (one torch.mm per module). Returns (codes uint8 [G, o, i] = the e4m3 codes
    of q - W0S in the permuted order, counters). Raises DenseSkip on any off-grid / infeasible element or after hard_stop."""
    import torch
    G, o, i = (int(v) for v in X.shape)
    codes = torch.empty((G, o, i), dtype=torch.uint8, device=X.device)
    pin_at = {}
    for g, p_ in enumerate(pins):  # None: no pin in that module (gptq_check's core test only)
        if p_ is not None:
            pin_at.setdefault(int(p_[1]), []).append((g, int(p_[0]), float(p_[2]) * 448.0))
    bad = torch.zeros((), dtype=torch.int64, device=X.device)
    nbad = nfar = nclamp = 0
    for b0 in range(0, i, block):
        if hard_stop is not None and time.time() > hard_stop:
            raise DenseSkip(f"gptq: hard stop reached at column {b0}/{i}")
        b1 = min(i, b0 + block)
        X1 = X[:, :, b0:b1].clone()
        E1 = torch.zeros_like(X1)
        Ub = U[:, b0:b1, b0:b1]
        for j in range(b1 - b0):
            c = b0 + j
            col = X1[:, :, j].contiguous()
            w0c = W0S[:, :, c].contiguous()
            q, nb, nf = round_feasible(col.reshape(-1), w0c.reshape(-1), "rtn", None)
            q = q.reshape(G, o)
            nbad += nb
            nfar += nf
            lim = LIMU[:, :, c]
            over = q.abs() > lim
            if bool(over.any()):
                nclamp += int(over.sum())
                q[over] = _gptq_clamp(q[over], lim[over], w0c[over])
            for g, r, v in pin_at.get(c, ()):
                q[g, r] = v
            dq = q - w0c
            bad += (~is_e4m3(dq)).sum() + (q.abs() > 448.0).sum()
            codes[:, :, c] = dq.to(torch.float8_e4m3fn).view(torch.uint8)
            err = (col - q) / Ub[:, j, j].unsqueeze(1)
            if j + 1 < b1 - b0:
                X1[:, :, j + 1:] -= err.unsqueeze(2) * Ub[:, j, j + 1:].unsqueeze(1)
            E1[:, :, j] = err
        if b1 < i:
            for g in range(G):
                X[g, :, b1:] -= E1[g] @ U[g, b0:b1, b1:]
        del X1, E1
    nb_ = int(bad)
    if nb_:
        raise DenseSkip(f"gptq: {nb_} off-grid / infeasible element(s) after rounding")
    return codes, {"infeasible": nbad, "moved": nfar, "clamped": nclamp}


def gptq_group_size(o, i, n, budget_bytes):
    """Modules of shape (o, i) per gptq_round group within budget_bytes of device memory: per module X, W0S, LIMU float32 + the
    codes uint8 (13 B per weight) + U float32; once: a float32 [o, i] permute temporary, gptq_factor's float64 [i, i] work
    matrices (<= 4 alive) and 1 GiB for the row-chunk temporaries. At least 1, at most n (None / 0 budget: 1)."""
    if not budget_bytes:
        return 1
    per = 13 * o * i + 4 * i * i
    fixed = 4 * o * i + 32 * i * i + (1 << 30)
    return max(1, min(int(n), int((budget_bytes - fixed) // per)))


def convert_gptq(lora_path, cache_files, out_path, hmap, device, hard_stop, log, chunk=CHUNK, extra=None, budget_bytes=None,
                 damp=GPTQ_DAMP, block=GPTQ_BLOCK, mod_log=None):
    """G56CT_DENSE=gptq: the dense file of lora_path at out_path with every block module rounded by GPTQ (module docstring above).
    hmap: {module (no 'diffusion_model.' prefix): H float32 [in, in] tensor, or a callable returning it} for EVERY LoRA module
    (DenseSkip otherwise: the caller ships its 'hyb' file instead). Same file layout as convert() (the LoRA modules, then the io
    pairs of extra); k and the pin are convert()'s pass 1 (identical to the hyb file's). budget_bytes: device memory for one
    group of same-shape modules (gptq_group_size; None = one module per group; a callable: evaluated before every group, e.g. the
    free memory as H tensors are released). mod_log: callable(dict) per module (optional).
    Raises DenseSkip / anything on failure. Returns stats (as convert(), + gptq counters and seconds)."""
    import torch
    from safetensors import safe_open
    t0 = time.time()
    with safe_open(str(lora_path), framework="pt", device="cpu") as fl0:
        lkeys = list(fl0.keys())
    mods = lora_modules(lkeys)
    nomiss = [m for m in mods if m not in hmap]
    if nomiss:
        raise DenseSkip(f"gptq: no H for {len(nomiss)} module(s) ({nomiss[:2]})")
    handles = {}

    def cache_tensor(name):
        p = cache_files[name]
        if p not in handles:
            handles[p] = safe_open(str(p), framework="pt", device="cpu")
        return handles[p].get_tensor(name)

    for m in mods:
        if m + ".weight" not in cache_files or m + ".weight_scale" not in cache_files:
            raise DenseSkip(f"{m}: not in the task folder's transformer")
    entries = [(PREFIX + m + ".lokr_w1", "F32", [1, 1], 4) for m in mods]
    fl = safe_open(str(lora_path), framework="pt", device="cpu")
    shapes = {}
    for m in mods:
        up_ = fl.get_slice(PREFIX + m + ".lora_up.weight").get_shape()
        dn_ = fl.get_slice(PREFIX + m + ".lora_down.weight").get_shape()
        shapes[m] = (up_[0], dn_[1])
        entries.append((PREFIX + m + ".lokr_w2", "F8_E4M3", [up_[0], dn_[1]], up_[0] * dn_[1]))
    xmods = sorted(extra, key=_mkey) if extra else []
    if set(xmods) & set(mods):
        raise DenseSkip(f"extra modules overlap the LoRA modules: {sorted(set(xmods) & set(mods))[:2]}")
    for m in xmods:
        o_, i_ = (int(v) for v in extra[m][1].shape)
        entries.append((PREFIX + m + ".lokr_w1", "F32", [1, 1], 4))
        entries.append((PREFIX + m + ".lokr_w2", "F8_E4M3", [o_, i_], o_ * i_))
    head, offs, end = _header(entries)
    stats = {"modules": len(mods), "k": {}, "clamped": 0, "uncertain": 0, "infeasible": 0, "moved": 0, "pin_zero": 0, "w0fix": 0,
             "gptq": {"groups": 0, "dead": 0, "damp_up": 0, "sec_factor": 0.0, "sec_round": 0.0}}
    groups = {}
    for m in mods:  # same-shape modules, in module order
        groups.setdefault(shapes[m], []).append(m)
    nel_all = sum(o_ * i_ for o_, i_ in shapes.values())
    with open(out_path, "wb") as f:
        f.write(head)
        f.truncate(end)
        nel_done = 0
        for (o, i), ms in groups.items():
            g0 = 0
            while g0 < len(ms):
                if time.time() > hard_stop:
                    raise DenseSkip(f"gptq: hard stop reached ({nel_done / max(1, nel_all):.0%} of the weights)")
                gsz = gptq_group_size(o, i, len(ms) - g0, budget_bytes() if callable(budget_bytes) else budget_bytes)
                part = ms[g0:g0 + gsz]
                g0 += gsz
                Gn = len(part)
                tg = time.time()
                Xg = torch.empty((Gn, o, i), dtype=torch.float32, device=device)
                Wg = torch.empty_like(Xg)
                Lg = torch.empty_like(Xg)
                Ug = torch.empty((Gn, i, i), dtype=torch.float32, device=device)
                tmp = torch.empty((o, i), dtype=torch.float32, device=device)
                pins, invs, ks, info = [], [], [], []
                for gi, m in enumerate(part):
                    q8 = cache_tensor(m + ".weight").to(device)
                    sc = cache_tensor(m + ".weight_scale").to(device)
                    up = fl.get_tensor(PREFIX + m + ".lora_up.weight").to(device, torch.float32)
                    dn = fl.get_tensor(PREFIX + m + ".lora_down.weight").to(device, torch.float32)
                    ak = PREFIX + m + ".alpha"
                    a = float(fl.get_tensor(ak).reshape(-1)[0]) / dn.shape[0] if ak in lkeys else 1.0
                    k, pin, pz = _gptq_pass1(q8, sc, up, dn, a, o, i, chunk)
                    tf = time.time()
                    H = hmap[m]() if callable(hmap[m]) else hmap[m]
                    if list(H.shape) != [i, i]:
                        raise DenseSkip(f"gptq: {m}: H shape {list(H.shape)} != [{i}, {i}]")
                    perm, U, ndead, dmp = gptq_factor(H, device, damp)
                    del H
                    stats["gptq"]["sec_factor"] += time.time() - tf
                    invp = torch.argsort(perm)
                    Ug[gi] = U
                    del U
                    w0fix, nunc = _gptq_targets(q8, sc, up, dn, a, o, i, k, chunk, Xg[gi], Wg[gi], Lg[gi])
                    for buf in (Xg, Wg, Lg):  # each module's own column order (act-order of its H)
                        torch.index_select(buf[gi], 1, perm, out=tmp)
                        buf[gi].copy_(tmp)
                    pins.append((pin[0], int(invp[pin[1]]), pin[2]))
                    invs.append(invp)
                    ks.append(k)
                    stats["pin_zero"] += pz
                    stats["w0fix"] += w0fix
                    stats["uncertain"] += nunc
                    stats["gptq"]["dead"] += ndead
                    stats["gptq"]["damp_up"] += int(dmp != damp)
                    info.append({"module": m, "k": k, "pin_zero": pz, "dead": ndead, "damp": dmp})
                    del q8, sc, up, dn, perm
                del tmp
                tr = time.time()
                codes, cst = gptq_round(Xg, Wg, Lg, Ug, pins, block=block, hard_stop=hard_stop)
                stats["gptq"]["sec_round"] += time.time() - tr
                del Xg, Wg, Lg, Ug
                for key_ in ("clamped", "infeasible", "moved"):
                    stats[key_] += cst[key_]
                for gi, m in enumerate(part):
                    cm = torch.index_select(codes[gi], 1, invs[gi])
                    f.seek(offs[PREFIX + m + ".lokr_w2"])
                    f.write(cm.cpu().numpy().tobytes())
                    f.seek(offs[PREFIX + m + ".lokr_w1"])
                    f.write(struct.pack("<f", 2.0 ** ks[gi]))
                    stats["k"][m] = ks[gi]
                    del cm
                    if mod_log is not None:
                        try:
                            mod_log(dict(info[gi], group=Gn, group_s=round(time.time() - tg, 2), **cst))
                        except Exception:  # noqa: BLE001  (a logging problem never fails the pass)
                            pass
                del codes
                stats["gptq"]["groups"] += 1
                nel_done += Gn * o * i
                el = time.time() - t0
                if t0 + el / nel_done * nel_all * 1.5 > hard_stop and nel_done < nel_all:  # write + verify ~ 1.5 x the write
                    raise DenseSkip(f"gptq: projected end after the hard stop ({el:.0f}s for {nel_done / nel_all:.0%} of the weights)")
        for m in xmods:  # G56CT_IO_FIX (T118): precomputed codes, written verbatim (as convert)
            k_, d_ = extra[m]
            f.seek(offs[PREFIX + m + ".lokr_w1"])
            f.write(struct.pack("<f", 2.0 ** int(k_)))
            f.seek(offs[PREFIX + m + ".lokr_w2"])
            f.write(d_.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        if xmods:
            stats["extra"] = len(xmods)
        f.flush()
        os.fsync(f.fileno())
    stats["seconds"] = round(time.time() - t0, 1)
    stats["gptq"]["sec_factor"] = round(stats["gptq"]["sec_factor"], 1)
    stats["gptq"]["sec_round"] = round(stats["gptq"]["sec_round"], 1)
    return stats
