"""G56CT_QF8_D15 (T167, 23 Oct 2026; qwen-image on the Jib-Mix root only; OFF by default; G56CT_PROFILE=c26s turns it on in
the Jib cell): the dense 15-block base correction "D15" of the T165 bedah, shipped by the trainer.



matrices of the 15 stratified blocks S15 of the shipped LoRA file become LoKr pairs
    'm.lokr_w1' = F32 [[2^k]],   'm.lokr_w2' = F8_E4M3 RTNE(target / 2^k),   target = lam * r + (alpha / rank) * up @ down
with r = m - J, the posterior restoration of the Jib root (float8_e5m2) towards the vanilla prior P (the bf16 transformer/ of the
task's own repo - the source G56CT_QF8_PRIOR already reads), m = E[T | the e5m2 rounding bin of J], T ~ N(P, s^2), s = the T119
fit of that matrix (FITS_* below), r rounded to F32; the module's trained LoRA is folded into the target and its three keys are
dropped (ComfyUI keeps one patch per weight); k = the smallest integer with 448 * 2^k >= max|target|, clamped to [-15, 0]. Every


the arithmetic below is a line-for-line port of those two tools (numpy float64, the same chunks, the same operation order), so
on one machine the shipped tensors are byte-identical to the tools' output for the same input file (tools/qf8d15_check.py; the

and the BLAS product may differ by an ulp in rare elements, as for the bedah tools themselves.

FITS: the 186 scalars (6 io matrices for the estimator gate + the 180 S15 matrices) are the T119 maximum-likelihood fits of the
Jib-Mix V4_E root against the vanilla prior (both files of the task repo itself; model 'abs' in all 186) - statistics of the base,
no weights, no data. They are used only when the task's root is that very file: size ROOT_BYTES and safetensors header sha256
ROOT_HDR_SHA256 (root_ok); any other root -> no D15 (the trainer ships the LoRA file, one WARN line).
Estimator gate (b20q op_r2d's): the same estimator on the six io matrices must reproduce the io restoration the run's own
QF8_PRIOR built (cos >= 0.99, rel <= 0.15), else no D15.

numpy + stdlib only (threads: numpy releases the GIL in its loops; the BLAS product runs under one lock). Streams one module at a
time per worker; never touches the input file; the output is written to the path given (the caller writes it next to the upload
folder and swaps it in after its own checks).
"""
import hashlib
import json
import math
import os
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import numpy as np

PREFIX = "diffusion_model."
UP, DN, AL = ".lora_up.weight", ".lora_down.weight", ".alpha"
W1, W2 = ".lokr_w1", ".lokr_w2"
ES = {"F32": 4, "BF16": 2, "F16": 2, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1}
S15 = (3, 7, 11, 15, 19, 22, 26, 30, 34, 37, 41, 45, 49, 52, 56)   # b21q_ops.S15: one block per stratum of 4 (0-3 ... 56-59)
BLOCK_MODULES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj",
                 "attn.add_v_proj", "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj",
                 "txt_mlp.net.2")
QWEN_IO = ("img_in", "txt_in", "norm_out.linear", "proj_out", "time_text_embed.timestep_embedder.linear_1",
           "time_text_embed.timestep_embedder.linear_2")
MOD_RE = re.compile(r"^diffusion_model\.transformer_blocks\.\d+\.(img_mod|txt_mod)\.1$")
OWN_SUFFIXES = (UP, DN, AL, W1, W2, ".diff", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2", ".dora_scale",
                ".set_weight", ".w_norm")
CHUNK = 1 << 22            # b20q_ops.CHUNK: elements per posterior chunk of the flattened matrix
ROWS = 512                 # rows per encode chunk (elementwise: any split gives the same bytes)
KMIN, KMAX = -15, 0        # b21q_ops: k >= -15 keeps every 2^k * e4m3 value exact in fp16 (the evaluator's patch dtype on A100)
KRES = 8                   # b21q lokr 'kres': the rank of a restoration-only modulation module (layout guard)
LAM = 1.0                  # the pre-registered dose (b21q lokr refuses lam != 1 in restore mode without allow_lam)
GATE_COS, GATE_REL = 0.99, 0.15   # b20q op_r2d's estimator gate on the six io matrices
REL_MAX = 0.05             # b21q lokr's verify: the e4m3 representation's relative error per module
ROOT_BYTES = 20430679696   # Jib_Mix_Qwen-Image_V4_E_fp8_e5m2_00001_.safetensors (gradients-io-tournaments/Qwen-Image-Jib-Mix)
ROOT_HDR_SHA256 = "a7399e5bf76d3c5bd53d43f44d9215f3b7f6c242219b04d67708c5351e8d34a8"   # its 278600-byte header (box 2 = g56-storage)
FITS_SRC_SHA256 = "3a14138fbcb2e6a0724ea32110414b964b409c39e12e2866428ef5c6ac17ad09"   # T119 jib_bayes.jsonl, the entries below

# T119 fits (model 'abs': sd = the value) - the io matrices (estimator gate) and, per S15 block, the 12 BLOCK_MODULES in order
FITS_IO = {
    "img_in": 0.0016624029626787816,
    "txt_in": 0.0015100500891628834,
    "norm_out.linear": 0.000759192816741393,
    "proj_out": 0.00015084726305849563,
    "time_text_embed.timestep_embedder.linear_1": 0.00040283489430525233,
    "time_text_embed.timestep_embedder.linear_2": 0.0003889384634892228,
}
FITS_S15 = {
    3: (0.0029226043548397046, 0.0025060589350540093, 0.003541679062127778, 0.0032422256762251076, 0.002597345911863125, 0.002529923885531652, 0.0027380811864123316, 0.002875786381127249, 0.0026431604086924946, 0.002369221310301118, 0.002296496347710573, 0.001982689765280853),
    7: (0.0035365399224617945, 0.0028446794617777746, 0.0030371758914631906, 0.002746504837820018, 0.0024925668957484844, 0.0025722397691071083, 0.002390876766344148, 0.002411527328910227, 0.00268526087951112, 0.002217614394289966, 0.0027682603002535146, 0.0019317933063756792),
    11: (0.0031285646484658157, 0.003276640402245422, 0.0033586206030989654, 0.002761122030985775, 0.0024058418603595424, 0.002908139853575867, 0.0025823426471402486, 0.0023501121773310983, 0.002927691601194825, 0.0024003084293795374, 0.002744731066990356, 0.002145194288828505),
    15: (0.002821324688310258, 0.00263320769466255, 0.0035339798406345347, 0.002747809275700962, 0.003225544144542171, 0.003120145439413471, 0.0029750381439527965, 0.002706235519898605, 0.0028410694618681838, 0.0025240657292745065, 0.0029109824686678774, 0.0022774807288620125),
    19: (0.0029104606739636337, 0.0026199494193081077, 0.0027829531206525894, 0.0026023506161518117, 0.002380749147874247, 0.002607290111152022, 0.0027026793222084274, 0.002323928424319308, 0.002802596223103226, 0.0026927439240723152, 0.0029238875886940094, 0.002245868593043896),
    22: (0.0027163304175566237, 0.002589844973917791, 0.0026998553870116486, 0.0024543648684579944, 0.0026789132070872447, 0.00256409244931947, 0.0024327742915938645, 0.0023363940755424194, 0.002414721305345921, 0.0024843925745146153, 0.0028118108719279635, 0.00224856689998902),
    26: (0.002533848524721803, 0.0027257829637431156, 0.002894833094384732, 0.002711270892092005, 0.0022973978124694733, 0.0025002187838777707, 0.002349509240864338, 0.00228715263041238, 0.0022335070811911697, 0.0025390205183560286, 0.0030476215095259977, 0.0024947407257743404),
    30: (0.0023203462637923183, 0.002742208908558044, 0.00260186237994398, 0.002642229099017698, 0.002324384341563382, 0.0023119324013289597, 0.0022517460892880216, 0.0028796199112163, 0.0023768442243303446, 0.002669888097241632, 0.0035952269775807004, 0.002258661567998217),
    34: (0.0023028227769470865, 0.0024238634068989985, 0.0026677700703597934, 0.0025145015041133727, 0.0020961611372087067, 0.00235934892728372, 0.0026075696248007733, 0.0023412557602445734, 0.0023341565500112327, 0.0023607997883831684, 0.003324805045093355, 0.002007128674069033),
    37: (0.002453301187574383, 0.00289021747910286, 0.002718170667289606, 0.002711959096075464, 0.0021204357276976443, 0.002141384086745056, 0.002699019731214433, 0.002307997497633538, 0.0033540388873027484, 0.0025530356687897176, 0.0032319562640370177, 0.002194205120454816),
    41: (0.0026880303566160792, 0.0028852199476531236, 0.0026698200444684513, 0.0031166083296784355, 0.0026496371719211495, 0.002493297816960876, 0.0027319021728717014, 0.0023546851367438567, 0.0028167807466556974, 0.0025260860745310886, 0.0032682210658339103, 0.0021233927198417734),
    45: (0.0025535609868292163, 0.0027124417080469135, 0.002619457665432734, 0.002875128778547104, 0.0019919102187857273, 0.0025871300165910154, 0.0024667710725532867, 0.0019452473386607988, 0.0023396376280962945, 0.0023093802296948102, 0.0029520576553411504, 0.0021950286053527327),
    49: (0.0023742680137167137, 0.0023724323137934537, 0.002351592582532935, 0.0027944773056011413, 0.0020715824559473807, 0.0029772571827735186, 0.0021451706991183848, 0.001541663338980991, 0.0021986838538750574, 0.0022474154336571154, 0.002439048984655042, 0.0023293651739337206),
    52: (0.002721065288900958, 0.002317658796545405, 0.0029908880552547987, 0.003008434350168129, 0.0021231495310905664, 0.002984095677472737, 0.0030977093115354256, 0.0018344866637564903, 0.003015510384798161, 0.0024204190492684756, 0.0007681941024533221, 0.00069729239526699),
    56: (0.0028216326576721235, 0.002609381427650779, 0.002853962468130723, 0.003003872174802889, 0.0017770801998057166, 0.0036917622304480373, 0.0030350181594782635, 0.0015653108284498847, 0.0037080589495521576, 0.002748001774728336, 0.0008421852442652953, 0.0007612754769268137),
}


class Skip(Exception):
    """D15 not attempted / abandoned: the caller ships the LoRA file as written."""


def log_default(msg):
    print(f"[g56qd15 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fits_table():
    """{tensor name as in the root / shards (no prefix, '.weight'): ('abs', sd)} for the 6 io + 180 S15 matrices."""
    out = {f"{m}.weight": ("abs", float(v)) for m, v in FITS_IO.items()}
    for b, vals in FITS_S15.items():
        for mod, v in zip(BLOCK_MODULES, vals):
            out[f"transformer_blocks.{b}.{mod}.weight"] = ("abs", float(v))
    return out


def fits_digest(fits=None):
    """sha256 of the table's canonical text (name model repr(sd) per line, sorted) - pinned by tools/qf8d15_check.py"""
    fits = fits_table() if fits is None else fits
    txt = "".join(f"{n} {m} {p!r}\n" for n, (m, p) in sorted(fits.items()))
    return hashlib.sha256(txt.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------- safetensors I/O
class ST:
    """b20q_ops.ST: header, raw bytes, element ranges (one open file; one object per thread)."""

    def __init__(self, path):
        self.path = str(path)
        self.f = open(self.path, "rb")
        head = self.f.read(8)
        if len(head) != 8:
            raise Skip(f"{self.path}: not a safetensors file")
        n = struct.unpack("<Q", head)[0]
        if n > 1 << 28:
            raise Skip(f"{self.path}: header length {n} implausible")
        self.hbytes = self.f.read(n)
        self.h = json.loads(self.hbytes)
        self.meta = self.h.pop("__metadata__", None)
        self.base = 8 + n
        last = max((v["data_offsets"][1] for v in self.h.values()), default=0)
        if self.base + last != os.path.getsize(self.path):
            raise Skip(f"{self.path}: header end != file size")

    def close(self):
        try:
            self.f.close()
        except Exception:  # noqa: BLE001
            pass

    def raw(self, k):
        a, b = self.h[k]["data_offsets"]
        self.f.seek(self.base + a)
        buf = self.f.read(b - a)
        if len(buf) != b - a:
            raise Skip(f"{self.path}: short read {k}")
        return buf

    def raw_elems(self, k, e0, e1):
        """raw bytes of the flattened elements [e0, e1) of tensor k"""
        es = ES[self.dtype(k)]
        a = self.h[k]["data_offsets"][0]
        self.f.seek(self.base + a + e0 * es)
        buf = self.f.read((e1 - e0) * es)
        if len(buf) != (e1 - e0) * es:
            raise Skip(f"{self.path}: short read {k} elements {e0}:{e1}")
        return buf

    def dtype(self, k):
        return self.h[k]["dtype"]

    def shape(self, k):
        return list(self.h[k]["shape"])

    def numel(self, k):
        return math.prod(self.shape(k)) if self.shape(k) else 1

    def arr(self, k, dt=np.float64):
        """b20q ST.arr: float values (float32 decode first, then astype dt)"""
        return to_float(self.raw(k), self.dtype(k)).astype(dt, copy=False).reshape(self.shape(k) or [1])

    def scalar(self, k):
        return float(self.arr(k).reshape(-1)[0])

    def order(self):
        return sorted(self.h, key=lambda k: self.h[k]["data_offsets"][0])


def _fp8_lut(fmt):
    """float64 value of every byte of float8 'e4m3fn' / 'e5m2' (NaN / inf codes -> nan) - b20q_ops._fp8_lut"""
    out = np.empty(256, dtype=np.float64)
    for code in range(256):
        s = -1.0 if code & 0x80 else 1.0
        if fmt == "e4m3fn":
            e, m = (code >> 3) & 0xF, code & 7
            if e == 15 and m == 7:
                v = float("nan")
            elif e == 0:
                v = m / 8.0 * 2.0 ** -6
            else:
                v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        else:
            e, m = (code >> 2) & 0x1F, code & 3
            if e == 31:
                v = float("nan")
            elif e == 0:
                v = m / 4.0 * 2.0 ** -14
            else:
                v = (1.0 + m / 4.0) * 2.0 ** (e - 15)
        out[code] = s * v
    return out


LUT = {"F8_E4M3": _fp8_lut("e4m3fn"), "F8_E5M2": _fp8_lut("e5m2")}


def to_float(b, dt):
    """bytes of one tensor -> float32 (F32 / BF16 / F16 / fp8: exact) or float64 (F64) - b20q_ops.to_float"""
    if dt == "F32":
        return np.frombuffer(b, dtype="<f4")
    if dt == "BF16":
        return (np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if dt == "F16":
        return np.frombuffer(b, dtype="<f2").astype(np.float32)
    if dt == "F64":
        return np.frombuffer(b, dtype="<f8")
    if dt in LUT:
        return LUT[dt][np.frombuffer(b, dtype=np.uint8)].astype(np.float32)
    raise Skip(f"dtype {dt}")


# ----------------------------------------------------------------- e5m2 posterior (b20q_ops port of the trainer's QF8_PRIOR)
_NR = (1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277)


def _log_erfc_pos(z):
    """log(erfc(z)) for z >= 0 (Numerical Recipes erfcc in log form) - b20q_ops._log_erfc_pos"""
    t = 1.0 / (1.0 + 0.5 * z)
    p = _NR[-1]
    for c in _NR[-2::-1]:
        p = c + t * p
    return np.log(t) - z * z - 1.26551223 + t * p


def ndtr_pair(x):
    """(log_ndtr(x), log_ndtr(-x)) from one erfc evaluation per element - b20q_ops.ndtr_pair"""
    x = np.asarray(x, dtype=np.float64)
    e = _log_erfc_pos(np.abs(x) / math.sqrt(2.0))
    tail = math.log(0.5) + e
    body = np.log1p(-0.5 * np.exp(e))
    return np.where(x < 0, tail, body), np.where(x > 0, tail, body)


def qf8b_bins(J):
    """(lo, hi) of the RTNE float8_e5m2 rounding bins of the e5m2 values J (float64) - b20q_ops.qf8b_bins"""
    a = np.abs(J)
    mt, ex = np.frexp(a)
    up = np.exp2((np.maximum(ex - 1, -14) - 2).astype(np.float64))
    dn = np.where((mt == 0.5) & (a >= 2.0 ** -13), up / 2, up)
    zero = a == 0
    lo = np.where(zero, -2.0 ** -17, np.where(J >= 0, J - dn / 2, J - up / 2))
    hi = np.where(zero, 2.0 ** -17, np.where(J >= 0, J + up / 2, J + dn / 2))
    return lo, hi


def qf8b_ldn(a, b):
    """log(Phi(b) - Phi(a)), a < b - b20q_ops.qf8b_ldn"""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        la, qa = ndtr_pair(a)
        lb, qb = ndtr_pair(b)
        left = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-300)))
        right = qa + np.log1p(-np.exp(np.minimum(qb - qa, -1e-300)))
        mid = np.log(np.maximum(1.0 - np.exp(la) - np.exp(qb), 1e-300))
    return np.where(b <= 0, left, np.where(a >= 0, right, mid))


def qf8b_post(P, lo, hi, sd):
    """E[T | lo <= T <= hi], T ~ N(P, sd^2), kept in [lo, hi]; non-finite -> P (then clipped) - b20q_ops.qf8b_post"""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        al, be = (lo - P) / sd, (hi - P) / sd
        lz = qf8b_ldn(al, be)
        c = 0.5 * math.log(2 * math.pi)
        m = P + sd * (np.exp(-0.5 * al * al - c - lz) - np.exp(-0.5 * be * be - c - lz))
    return np.minimum(np.maximum(np.where(np.isfinite(m), m, P), lo), hi)


def _sd_of(p, model, par):
    return par if model == "abs" else par * np.maximum(np.abs(p), 2.0 ** -16)


def jib_restoration(P, J, model, par):
    """m - J (float64) for prior P (float64) and e5m2 values J (float64), chunked - b20q_ops.jib_restoration"""
    P, J = P.reshape(-1), J.reshape(-1)
    out = np.empty(P.size, dtype=np.float64)
    for c0 in range(0, P.size, CHUNK):
        p, j = P[c0:c0 + CHUNK], J[c0:c0 + CHUNK]
        lo, hi = qf8b_bins(j)
        out[c0:c0 + CHUNK] = qf8b_post(p, lo, hi, _sd_of(p, model, par)) - j
    return out


# ----------------------------------------------------------------------------------------------------- e4m3 / LoKr (b21q)
def choose_k(amax):
    """smallest integer k with 448 * 2^k >= amax, clamped to [KMIN, KMAX] - b21q_ops.choose_k"""
    if not (amax >= 0 and math.isfinite(amax)):
        raise Skip(f"amax {amax}")
    if amax == 0:
        return KMIN
    k = math.ceil(math.log2(amax / 448.0))
    while 448.0 * 2.0 ** (k - 1) >= amax:
        k -= 1
    while 448.0 * 2.0 ** k < amax:
        k += 1
    k = max(KMIN, k)
    if k > KMAX:
        raise Skip(f"amax {amax} needs 2^{k} > 2^{KMAX}")
    return k


_POW2_OFF = 64
_POW2 = np.ldexp(1.0, np.arange(-_POW2_OFF, _POW2_OFF + 1)).astype(np.float64)


def rtne_codes_e4m3(x):
    """float8_e4m3fn codes of RTNE(x) by exponent arithmetic (x float64, |x| <= 448, finite) - b21q_ops.rtne_codes_e4m3"""
    v = np.asarray(x, dtype=np.float64)
    a = np.abs(v)
    if a.size and not bool(np.all(np.isfinite(a))):
        raise Skip("non-finite value in the input of the e4m3 encoder")
    if a.size and float(a.max()) > 448.0:
        raise Skip(f"value {float(a.max())} above the F8_E4M3 range")
    _, ex = np.frexp(a)
    y = a * _POW2[np.clip(4 - ex, -_POW2_OFF, _POW2_OFF) + _POW2_OFF]
    np.rint(y, out=y)
    code = y.astype(np.int64) + 8 * ex.astype(np.int64) + 40
    sub = np.flatnonzero(a.reshape(-1) < 2.0 ** -6)
    if sub.size:
        code.reshape(-1)[sub] = np.rint(a.reshape(-1)[sub] * 512.0).astype(np.int64)
    codes = code.astype(np.uint8)
    return codes | np.where((v < 0) & (codes > 0), 0x80, 0).astype(np.uint8)


def e4m3_decode(codes):
    return LUT["F8_E4M3"][np.asarray(codes, dtype=np.uint8)]


# ------------------------------------------------------------------------------------------------------------- layout
def _mod_of(k):
    for s in (UP, DN, AL):
        if k.endswith(s):
            return k[: -len(s)]
    return None


def lora_modules(S):
    return sorted({_mod_of(k) for k in S.h if k.endswith(DN)})


def check_src_layout(S):
    """b21q check_src_layout(base='jib'): trained attn/MLP + restoration-only modulation (rank KRES, alpha KRES) = a vanilla
    layout -> Skip; joined (trained) modulation modules = the Jib-Mix layout of the trainer's QF8_PRIOR files."""
    mods = [m for m in lora_modules(S) if MOD_RE.match(m)]
    joined = [m for m in mods if S.shape(m + DN)[0] != KRES or abs(S.scalar(m + AL) - KRES) > 1e-6]
    if mods and not joined and len(lora_modules(S)) > len(mods):
        raise Skip("the file has trained attn/MLP + restoration-only modulation modules (a vanilla qwen-image layout)")
    return len(mods), len(joined)


def _shard_index(shards):
    ix = os.path.join(str(shards), "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(ix):
        with open(ix) as f:
            return {k: os.path.join(str(shards), v) for k, v in json.load(f)["weight_map"].items()}
    one = os.path.join(str(shards), "diffusion_pytorch_model.safetensors")
    if os.path.isfile(one):
        S = ST(one)
        try:
            return {k: one for k in S.h}
        finally:
            S.close()
    raise Skip(f"no bf16 shards in {shards}")


def _root_names(R):
    out = {}
    for k in R.h:
        n = k
        for p in ("model.diffusion_model.", "diffusion_model."):
            if n.startswith(p):
                n = n[len(p):]
                break
        out[n] = k
    return out


def root_ok(root):
    """the Jib-Mix V4_E root the FITS belong to: size and header sha256 (Skip otherwise)"""
    size = os.path.getsize(str(root))
    with open(str(root), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        sha = hashlib.sha256(f.read(n)).hexdigest() if n <= 1 << 28 else ""
    if size != ROOT_BYTES or sha != ROOT_HDR_SHA256:
        raise Skip(f"root {os.path.basename(str(root))} ({size} B, header {sha[:12]}) is not the Jib-Mix V4_E root of the fits "
                   f"({ROOT_BYTES} B, {ROOT_HDR_SHA256[:12]})")
    return True


def module_names(blocks=S15):
    """the D15 modules in the bedah's order: sorted module names (b21q cache_modules sorts the r2d cache keys)"""
    return sorted(PREFIX + f"transformer_blocks.{int(b)}.{mod}" for b in blocks for mod in BLOCK_MODULES)


def _header(entries, meta):
    """safetensors header bytes for [(key, dtype, shape)] (b20q write_st's JSON: compact, padded with spaces to 8) + total size"""
    hdr, off = {}, 0
    for k, dt, sh in entries:
        nb = ES[dt] * (math.prod(sh) if sh else 1)
        hdr[k] = {"dtype": dt, "shape": list(sh), "data_offsets": [off, off + nb]}
        off += nb
    if meta is not None:
        hdr["__metadata__"] = {str(k): str(v) for k, v in meta.items()}
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    return struct.pack("<Q", len(hb)) + hb, 8 + len(hb) + off


def plan(src, root, shards, blocks=S15, fits=None):
    """Header-only plan of the D15 file of src (the shipped LoRA): checks the layout (Jib), every module's root (F8_E5M2) /
    shard (BF16) pair and LoRA shapes, its fit; -> dict(mods, info, drop, entries, header, out_bytes, elements, ...). Skip on
    anything the bedah's tools would refuse."""
    fits = fits_table() if fits is None else fits
    S, R = ST(src), ST(root)
    opened = {}
    try:
        njm = check_src_layout(S)
        have = set(lora_modules(S))
        rnames = _root_names(R)
        smap = _shard_index(shards)
        mods = module_names(blocks)
        info = {}
        for m in mods:
            if MOD_RE.match(m):
                raise Skip(f"{m}: a modulation module")
            for suf in OWN_SUFFIXES[3:]:   # b21q free_module
                if m + suf in S.h:
                    raise Skip(f"{m}: the file already carries '{m}{suf}'")
            n = m[len(PREFIX):] + ".weight"
            rk, sp = rnames.get(n), smap.get(n)
            if rk is None or sp is None:
                raise Skip(f"{n} not in the root / the shards")
            if sp not in opened:
                opened[sp] = ST(sp)
            sh = opened[sp]
            if R.dtype(rk) != "F8_E5M2" or sh.dtype(n) != "BF16" or R.shape(rk) != sh.shape(n) or len(R.shape(rk)) != 2:
                raise Skip(f"{n}: root {R.dtype(rk)} {R.shape(rk)} vs shard {sh.dtype(n)} {sh.shape(n)} (want F8_E5M2 / BF16, 2-D)")
            fit = fits.get(n)
            if fit is None or fit[0] not in ("abs", "rel") or not (fit[1] > 0 and math.isfinite(fit[1])):
                raise Skip(f"{n}: no usable fit")
            o, i = R.shape(rk)
            lo = m in have
            if lo:
                if not all(m + s in S.h for s in (UP, AL)):
                    raise Skip(f"{m}: a LoRA module without {UP if m + UP not in S.h else AL}")
                if S.shape(m + UP)[0] != o or S.shape(m + DN)[1] != i or len(S.shape(m + UP)) != 2 or len(S.shape(m + DN)) != 2 \
                        or S.shape(m + UP)[1] != S.shape(m + DN)[0]:
                    raise Skip(f"{m}: LoRA {S.shape(m + UP)} x {S.shape(m + DN)} vs root {[o, i]}")
            info[m] = {"shape": [o, i], "rk": rk, "sk": n, "shard": sp, "model": fit[0], "par": float(fit[1]), "lora": lo}
        fits_io = {}
        for mio in QWEN_IO:
            fit = fits.get(mio + ".weight")
            if fit is None or fit[0] not in ("abs", "rel") or not (fit[1] > 0 and math.isfinite(fit[1])):
                raise Skip(f"{mio}: no usable fit (estimator gate)")
            fits_io[mio] = (fit[0], float(fit[1]))
        drop = {m + s for m in mods if info[m]["lora"] for s in (UP, DN, AL)}
        entries = [(k, S.dtype(k), S.shape(k)) for k in S.order() if k not in drop]
        for m in mods:   # b21q op_lokr: w2 before w1 (w1 = [[2^k]] needs the k the w2 pass found)
            entries.append((m + W2, "F8_E4M3", info[m]["shape"]))
            entries.append((m + W1, "F32", [1, 1]))
        head, total = _header(entries, S.meta)
        return {"src": str(src), "root": str(root), "shards": str(shards), "mods": mods, "info": info, "drop": drop,
                "entries": entries, "header": head, "out_bytes": total, "src_bytes": os.path.getsize(str(src)),
                "elements": sum(v["shape"][0] * v["shape"][1] for v in info.values()), "folded": sum(v["lora"] for v in info.values()),
                "modulation": njm, "meta": S.meta, "fits_io": fits_io}
    finally:
        S.close()
        R.close()
        for h in opened.values():
            h.close()


def estimate_s(pl, workers, us_per_el=0.40, fixed_s=15.0, write_mbps=400.0, par_cap=8):
    """projected seconds of build + verify: the posterior + target + encode at us_per_el microseconds per element per worker
    (box 2's bedah r2d + lokr: ~0.37), the output written and the kept tensors re-read at write_mbps, + fixed_s"""
    w = max(1, min(int(workers), int(par_cap)))   # the posterior is memory-bound: the bench saw no gain beyond ~6-8 workers
    return float(fixed_s + pl["elements"] * us_per_el * 1e-6 / w + 2.0 * pl["out_bytes"] / (write_mbps * 1e6))


# ------------------------------------------------------------------------------------------------------------- build
class _Files:
    """one ST per (thread, path): a file object is never shared between threads (seek + read); close_all() at the end"""

    def __init__(self):
        self.files, self.lock = {}, threading.Lock()

    def st(self, path):
        key = (threading.get_ident(), str(path))
        with self.lock:
            f = self.files.get(key)
        if f is None:
            f = ST(path)
            with self.lock:
                self.files[key] = f
        return f

    def close_all(self):
        with self.lock:
            fs, self.files = list(self.files.values()), {}
        for f in fs:
            f.close()


def _restoration_f32(sh, sk, rt, rk, n, model, par, abort):
    """r = m - J as float32 for the n elements of one matrix: b20q op_r2d's _r2d_job (jib_restoration over the flattened matrix in
    CHUNK slices, then astype float32), read slice by slice (elementwise: the same values, less memory)"""
    r32 = np.empty(n, dtype=np.float32)
    for c0 in range(0, n, CHUNK):
        if abort.is_set():
            raise Skip("aborted")
        c1 = min(n, c0 + CHUNK)
        p = to_float(sh.raw_elems(sk, c0, c1), "BF16").astype(np.float64)
        j = to_float(rt.raw_elems(rk, c0, c1), "F8_E5M2").astype(np.float64)
        lo, hi = qf8b_bins(j)
        r32[c0:c1] = (qf8b_post(p, lo, hi, _sd_of(p, model, par)) - j).astype(np.float32)
    return r32


def _module_job(pl, m, lam, loc, lock, abort):
    """one module -> (k, amax, rel, w2 bytes, sha256, seconds): b21q _lokr_build_job on b20q's r2d of the module"""
    t0 = time.time()
    if abort.is_set():
        raise Skip("aborted")
    v = pl["info"][m]
    o, i = v["shape"]
    r32 = _restoration_f32(loc.st(v["shard"]), v["sk"], loc.st(pl["root"]), v["rk"], o * i, v["model"], v["par"], abort)
    acc = r32.astype(np.float64).reshape(o, i)         # the r2d cache value read back (F32 -> float64)
    del r32
    np.multiply(acc, lam, out=acc)                     # acc = lam * r
    if v["lora"]:
        S = loc.st(pl["src"])
        sc = S.scalar(m + AL) / S.shape(m + DN)[0]     # b20q lora_scale
        u = S.arr(m + UP)
        d = S.arr(m + DN).reshape(S.shape(m + DN)[0], -1)
        with lock:                                     # one BLAS product at a time (its own threads parallelize it)
            ud = u.reshape(u.shape[0], -1) @ d
        np.multiply(ud, sc, out=ud)                    # scale * (up @ down)
        np.add(acc, ud, out=acc)                       # lam * r + scale * up @ down
        del ud, u, d
    amax = 0.0
    for r0 in range(0, o, ROWS):
        amax = max(amax, float(np.max(np.abs(acc[r0:r0 + ROWS]))))
    k = choose_k(amax)
    s = 2.0 ** k
    codes = np.empty((o, i), dtype=np.uint8)
    se = st = 0.0
    for r0 in range(0, o, ROWS):
        if abort.is_set():
            raise Skip("aborted")
        t = acc[r0:r0 + ROWS]
        c = rtne_codes_e4m3(t / s)
        codes[r0:r0 + ROWS] = c
        dec = e4m3_decode(c) * s
        se += float(np.sum((dec - t) ** 2))
        st += float(np.sum(t * t))
    rel = math.sqrt(se / st) if st > 0 else (0.0 if se == 0 else float("inf"))
    if not rel <= REL_MAX:
        raise Skip(f"{m}: e4m3 representation rel err {rel:.4f} > {REL_MAX}")
    b = codes.tobytes()
    return {"k": k, "amax": amax, "rel": rel, "w2": b, "sha": hashlib.sha256(b).hexdigest(), "s": time.time() - t0}


def _gate_job(pl, mio, ref, loc, abort):
    """b20q op_r2d's estimator gate on one io matrix: the same estimator (T119 fit) vs the run's own QF8_PRIOR restoration"""
    if abort.is_set():
        raise Skip("aborted")
    n = mio + ".weight"
    rt = loc.st(pl["root"])
    rn = _root_names(rt)
    smap = _shard_index(pl["shards"])
    if n not in rn or n not in smap:
        raise Skip(f"gate: {n} not in the root / the shards")
    sh = loc.st(smap[n])
    if rt.dtype(rn[n]) != "F8_E5M2" or sh.dtype(n) != "BF16" or rt.shape(rn[n]) != sh.shape(n):
        raise Skip(f"gate: {n} root {rt.dtype(rn[n])} {rt.shape(rn[n])} vs shard {sh.dtype(n)} {sh.shape(n)}")
    P, J = sh.arr(n, np.float64), rt.arr(rn[n], np.float64)
    model, par = pl["fits_io"][mio]
    mine = jib_restoration(P, J, model, par)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if ref.size != mine.size:
        raise Skip(f"gate: {mio} reference has {ref.size} elements, the matrix {mine.size}")
    rel = float(np.linalg.norm(mine - ref) / (np.linalg.norm(ref) or 1e-300))
    cos = float(mine @ ref / ((np.linalg.norm(mine) * np.linalg.norm(ref)) or 1e-300))
    if not (cos >= GATE_COS and rel <= GATE_REL):
        raise Skip(f"estimator gate failed on {mio}: rel {rel:.4f} cos {cos:.6f} (need rel <= {GATE_REL}, cos >= {GATE_COS})")
    return {"rel": round(rel, 5), "cos": round(cos, 6)}


def build(pl, out, workers=4, hard_stop=None, log=None, io_ref=None, lam=LAM, tail_s=0.0, copy_buf=1 << 24):
    """Write the D15 file of plan pl at out (never overwritten: Skip when it exists). Threads: `workers` module jobs at a time,
    the file written in order by the calling thread (header, the kept tensors byte for byte, then w2 / w1 per module). io_ref
    {io module: restoration array} runs the estimator gate first (None: no gate - tests). hard_stop (time.time()): after every
    module the projected end (+ tail_s for the caller's verification) must stay before it, else Skip. -> stats; Skip / any
    exception: the partial file is removed."""
    log = log or log_default
    if not math.isfinite(lam) or lam <= 0:
        raise Skip(f"lam {lam}")
    if os.path.exists(out):
        raise Skip(f"{out} exists")
    t0 = time.time()
    mods = pl["mods"]
    loc, lock, abort = _Files(), threading.Lock(), threading.Event()
    stats = {"modules": {}, "gate": None, "workers": int(workers)}
    ex = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="g56qd15")
    ok, futs = False, {}
    try:
        if io_ref is not None:
            gates = {mio: ex.submit(_gate_job, pl, mio, io_ref[mio], loc, abort) for mio in QWEN_IO}
            stats["gate"] = {mio: f.result() for mio, f in gates.items()}
            log(f"qf8-d15: estimator gate OK {json.dumps(stats['gate'])} ({time.time() - t0:.1f}s)")
        window = 2 * max(1, int(workers)) + 4   # results queue ahead of the in-order writer: a slow MLP job does not idle the rest
        nxt = 0

        def fill():
            nonlocal nxt
            while nxt < len(mods) and len(futs) < window:
                futs[mods[nxt]] = ex.submit(_module_job, pl, mods[nxt], lam, loc, lock, abort)
                nxt += 1
        fill()
        S = ST(pl["src"])
        done_el, job_s, tj, nw = 0, 0.0, time.time(), max(1, int(workers))
        try:
            with open(out, "xb") as fo:
                fo.write(pl["header"])
                for k, _, _ in pl["entries"]:
                    if k.endswith((W1, W2)) and k[: -len(W1 if k.endswith(W1) else W2)] in pl["info"]:
                        continue
                    a, b = S.h[k]["data_offsets"]
                    S.f.seek(S.base + a)
                    left = b - a
                    while left:
                        buf = S.f.read(min(copy_buf, left))
                        if not buf:
                            raise Skip(f"short read {k}")
                        fo.write(buf)
                        left -= len(buf)
                for m in mods:
                    wait = None if hard_stop is None else max(1.0, hard_stop - time.time())
                    try:
                        r = futs.pop(m).result(timeout=wait)
                    except FuturesTimeout:
                        raise Skip(f"the hard stop passed while module {len(stats['modules']) + 1} of {len(mods)} was built")
                    fill()
                    fo.write(r["w2"])
                    fo.write(np.array([[2.0 ** r["k"]]], dtype="<f4").tobytes())
                    o, i = pl["info"][m]["shape"]
                    done_el += o * i
                    job_s += r["s"]
                    stats["modules"][m] = {k_: r[k_] for k_ in ("k", "amax", "rel", "sha", "s")}
                    if hard_stop is not None:
                        # the jobs' own seconds (run concurrently: contention included) / workers; once the pipeline has been full
                        # for 2 x workers modules, at least the wall-clock throughput
                        now = time.time()
                        left_el = pl["elements"] - done_el
                        rem = left_el * job_s / done_el / nw
                        if len(stats["modules"]) >= 2 * nw:
                            rem = max(rem, left_el * (now - tj) / done_el)
                        proj = now + rem + tail_s
                        if proj > hard_stop:
                            raise Skip(f"projected end {proj - now:.0f}s from now passes the hard stop ({hard_stop - now:.0f}s left) "
                                       f"after {len(stats['modules'])} of {len(mods)} modules")
                fo.flush()
                os.fsync(fo.fileno())
        finally:
            S.close()
        size = os.path.getsize(out)
        if size != pl["out_bytes"]:
            raise Skip(f"wrote {size} B, the plan says {pl['out_bytes']} B")
        ks = [v["k"] for v in stats["modules"].values()]
        stats.update(bytes=size, seconds=round(time.time() - t0, 1), k_range=[min(ks), max(ks)],
                     worst_rel=round(max(v["rel"] for v in stats["modules"].values()), 5),
                     job_s=round(sum(v["s"] for v in stats["modules"].values()), 1))
        ok = True
        return stats
    finally:
        if not ok:
            abort.set()
        for f in list(futs.values()):
            f.cancel()
        ex.shutdown(wait=True, cancel_futures=True)
        loc.close_all()
        if not ok:
            try:
                os.remove(out)
            except FileNotFoundError:
                pass


def verify(out, pl, stats):
    """Re-read the written file: the planned keys / dtypes / shapes in order, every kept tensor byte-identical to src, every
    module's w1 == [[2^k]] (k in [KMIN, KMAX], the build's k), w2 without a NaN code and == the build's bytes (sha256). Skip on
    any difference; -> the number of D15 modules."""
    O, S = ST(out), ST(pl["src"])
    try:
        got = [(k, O.dtype(k), O.shape(k)) for k in O.order()]
        want = [(k, dt, list(sh)) for k, dt, sh in pl["entries"]]
        if got != want:
            raise Skip(f"verify: key / dtype / shape list differs ({len(got)} vs {len(want)} entries)")
        if O.meta != pl["meta"]:
            raise Skip("verify: metadata differs from the source file's")
        n = 0
        for k, _, _ in want:
            if k.endswith((W1, W2)) and k[: -len(W1 if k.endswith(W1) else W2)] in pl["info"]:
                continue
            if O.raw(k) != S.raw(k):
                raise Skip(f"verify: {k} is not byte-identical to the source file")
            n += 1
        for m in pl["mods"]:
            st = stats["modules"][m]
            w1 = struct.unpack("<f", O.raw(m + W1))[0]
            if not (KMIN <= st["k"] <= KMAX) or w1 != 2.0 ** st["k"]:
                raise Skip(f"verify: {m} lokr_w1 {w1!r} != 2^{st['k']}")
            b = O.raw(m + W2)
            if hashlib.sha256(b).hexdigest() != st["sha"]:
                raise Skip(f"verify: {m} lokr_w2 != the built codes")
            if bool(np.any((np.frombuffer(b, dtype=np.uint8) & 0x7F) == 0x7F)):
                raise Skip(f"verify: {m} lokr_w2 has a NaN code")
        return {"kept_identical": n, "modules": len(pl["mods"])}
    finally:
        O.close()
        S.close()


def _rtne_e4m3_bruteforce(x):
    """INDEPENDENT reference: e4m3fn codes of RTNE(x) by exhaustive distance to every finite non-negative code (ties -> even code)"""
    pos = LUT["F8_E4M3"][:0x7F]
    a = np.abs(np.asarray(x, dtype=np.float64)).reshape(-1)
    dist = np.abs(a[:, None] - pos[None, :])
    best = dist.min(axis=1, keepdims=True)
    cand = dist == best
    codes = np.arange(len(pos))
    pick = np.where(cand.sum(axis=1) > 1, np.argmax(cand & (codes[None, :] % 2 == 0), axis=1), np.argmax(cand, axis=1))
    sign = (np.asarray(x, dtype=np.float64).reshape(-1) < 0) & (pick > 0)
    return (pick | np.where(sign, 0x80, 0)).astype(np.uint8)


def spot_check(out, pl, nmods=4, rows=(0, -1), lam=LAM):
    """Independent re-derivation of sampled rows (b21q verify's idea): for nmods modules spread over the list, rows `rows` (+ the
    middle row): r from the shard / root rows (float64 posterior -> F32), the folded LoRA rows in float64, the target / 2^k
    coded by an exhaustive RTNE search; == the file's codes wherever the value is not within 1e-9 of a rounding tie (the row
    product may differ from the full product by an ulp). Skip on a mismatch; -> rows checked."""
    O, S, R = ST(out), ST(pl["src"]), ST(pl["root"])
    opened = {}
    try:
        mods = pl["mods"]
        pick = sorted({mods[int(round(x * (len(mods) - 1) / max(1, nmods - 1)))] for x in range(nmods)})
        nrow = 0
        for m in pick:
            v = pl["info"][m]
            o, i = v["shape"]
            if v["shard"] not in opened:
                opened[v["shard"]] = ST(v["shard"])
            sh = opened[v["shard"]]
            s = float(struct.unpack("<f", O.raw(m + W1))[0])
            codes = np.frombuffer(O.raw(m + W2), dtype=np.uint8).reshape(o, i)
            for r0 in sorted({x % o for x in rows} | {o // 2}):
                p = to_float(sh.raw_elems(v["sk"], r0 * i, (r0 + 1) * i), "BF16").astype(np.float64)
                j = to_float(R.raw_elems(v["rk"], r0 * i, (r0 + 1) * i), "F8_E5M2").astype(np.float64)
                r = jib_restoration(p, j, v["model"], v["par"]).astype(np.float32).astype(np.float64)
                t = lam * r
                if v["lora"]:
                    sc = S.scalar(m + AL) / S.shape(m + DN)[0]
                    u = to_float(S.raw_elems(m + UP, r0 * S.shape(m + UP)[1], (r0 + 1) * S.shape(m + UP)[1]), S.dtype(m + UP))
                    d = S.arr(m + DN).reshape(S.shape(m + DN)[0], -1)
                    t = t + sc * (u.astype(np.float64)[None, :] @ d).reshape(-1)
                x = t / s
                if float(np.max(np.abs(x))) > 448.0:
                    raise Skip(f"spot: {m} row {r0}: |target| / 2^k above 448")
                ref = _rtne_e4m3_bruteforce(x)
                pos = LUT["F8_E4M3"][:0x7F]
                ax = np.abs(x)
                idx = np.clip(np.searchsorted(pos, ax), 1, len(pos) - 1)
                tie = np.abs(ax - (pos[idx - 1] + pos[idx]) / 2) <= 1e-9 * np.maximum(ax, 2.0 ** -9)
                bad = (ref != codes[r0]) & ~tie
                if bool(bad.any()):
                    raise Skip(f"spot: {m} row {r0}: {int(bad.sum())} codes differ from the independent re-derivation")
                nrow += 1
        return nrow
    finally:
        for h in (O, S, R, *opened.values()):
            h.close()


# ----------------------------------------------------------------------------------------- T249 D15X (31 Oct 2026)
#
#
# 186/186), in the same layout as FITS_S15 (per block, the 12 BLOCK_MODULES in order). Used only by fits_table_x() / D15X_SETS for
# the trainer's T249_D15X switch (S60 / S30 when the run's idle window fits them, else S15); fits_table() and S15 are unchanged.
FITS_X = {
    0: (0.0024490625012952984, 0.002469043734870464, 0.0017589931188142982, 0.0017591261497610588, 0.0038371752128380345, 0.0023740601469288483, 0.0019989976855426706, 0.002674661882303771, 0.0016235520121065687, 0.002013152584542319, 0.003759452179113378, 0.0029204231930289775),
    1: (0.0023219075624578834, 0.0023051408255236393, 0.002207580194549598, 0.002908663203501229, 0.0025939885018742884, 0.002648814429668027, 0.0031540403267823454, 0.003499508763371874, 0.0033275017219655816, 0.002398985930421769, 0.002657843395543596, 0.0022597376668669527),
    2: (0.0027262249108078833, 0.002449734505528932, 0.0035906499317027406, 0.0033057295048716973, 0.002468728329762898, 0.002824185854215032, 0.0024845272216642697, 0.0028308054719363954, 0.003112681911604735, 0.002418064768082828, 0.0021521338059574944, 0.002002249206552754),
    4: (0.0030980596546556763, 0.0027158764534110902, 0.0029398048534646677, 0.002825024027531822, 0.002528715953269486, 0.002413423218009531, 0.0023816853858411014, 0.002758330394670809, 0.0026087229072443096, 0.002146259035733132, 0.0023650789768447943, 0.0018580132782989322),
    5: (0.0030804929880779556, 0.002963024468085837, 0.003568655653646503, 0.002824815382138742, 0.002542758886083032, 0.002838523917219009, 0.0027693882153217776, 0.002628968461404026, 0.002847882309238549, 0.0024073717510022857, 0.0021809370951685458, 0.0018839559840061464),
    6: (0.003185436562998201, 0.0027992072620303897, 0.003712342440076426, 0.003055417893428593, 0.0021932240898497476, 0.0027408873354371447, 0.002480052836740813, 0.0026432926803887044, 0.0029158290572287083, 0.002126853859602883, 0.002507692068605277, 0.0018632873878965004),
    8: (0.002897354775658506, 0.00277717248785398, 0.0036653034867284977, 0.002509689590553935, 0.002964054800066538, 0.0029911792373492467, 0.002860965477861555, 0.0025746565252978294, 0.0026516163291835366, 0.0022477801920150024, 0.0027057021718984915, 0.0019205702774772473),
    9: (0.0036189159053133745, 0.002848983570096148, 0.0033411570052634195, 0.0026966884690737113, 0.002964117104623214, 0.0036803175304793087, 0.003002470024957332, 0.002934261972953039, 0.002874896303989027, 0.0023329007089173837, 0.0027106776983354232, 0.0023288481518479263),
    10: (0.0032173897873861816, 0.0027839735934588666, 0.003299242051447174, 0.0026250301418473754, 0.0027079450373917817, 0.003163420223436298, 0.0037199031688746912, 0.002527742689394205, 0.0026751361611743888, 0.0022605211745313075, 0.002475970552980358, 0.0021014710783559608),
    12: (0.0034751293017493377, 0.0032552645346594104, 0.0030678434163129735, 0.0027181004507644655, 0.0027360023349478945, 0.002649273121650336, 0.0026578284771625153, 0.0025960715755580952, 0.0026545374321959854, 0.002346236976627406, 0.00223556247848254, 0.0018671473122836642),
    13: (0.0028558273898105432, 0.0029579781751859244, 0.0032086848555112468, 0.0026352026928927004, 0.0026233068339172353, 0.003022566216782069, 0.002752973419966303, 0.0027220270729587225, 0.002714228903740353, 0.002194534140258295, 0.0029589227754973075, 0.0018631236903405013),
    14: (0.0031463733243811327, 0.0027507830276079686, 0.0033692672829235075, 0.00254645794963143, 0.0026431767945339717, 0.002846165930218063, 0.002994745574116412, 0.0022327148090335305, 0.0028954826950170084, 0.002588913520673646, 0.002922637444138685, 0.0020921276027434416),
    16: (0.003040640532619661, 0.0034463597796619964, 0.0032511061772555498, 0.0026966055111153136, 0.0029877833974196683, 0.0033249758854483192, 0.0033107201729454623, 0.002527001777272341, 0.002570876678840329, 0.0026928691275050726, 0.002841596048774143, 0.0019831780007763594),
    17: (0.0032830376840059985, 0.0025303692959215756, 0.002812747268653627, 0.002975308502169818, 0.002265113890877388, 0.0030554309832071676, 0.002621099264109461, 0.0023864478040467384, 0.002526892842895872, 0.0026633507813804583, 0.0025242563484860186, 0.0019960273745486374),
    18: (0.0029010141458671304, 0.0031732960762814334, 0.0033940568001708145, 0.002875330491856035, 0.0027477508457161683, 0.002387583281880412, 0.00313095113858757, 0.002548783994778473, 0.002522911778207021, 0.002564480002624249, 0.0027249042567154677, 0.0022614269876578986),
    20: (0.0025207787582636446, 0.0029949615031255356, 0.0030191741112680377, 0.002610665102053875, 0.0024450393696535413, 0.002248694966099264, 0.0024361386424241165, 0.0023690796389167915, 0.0027445861198231894, 0.002624997315307339, 0.002910080011395273, 0.002198600083915121),
    21: (0.002487387593107424, 0.0026668988965499727, 0.0034266418859791284, 0.0025426743104607326, 0.00255345595721913, 0.002615880703731252, 0.00267525434512821, 0.0022810746322070165, 0.0024810483883205654, 0.0025581809124582025, 0.003143021250206295, 0.002408022014622675),
    23: (0.002686543883874928, 0.0026607178015836557, 0.0028943799684087125, 0.002635900439578634, 0.0033207483050041162, 0.002509893584461279, 0.0028609621140705858, 0.00232912160574039, 0.0026576453224883365, 0.002445975893262682, 0.0028275498263779096, 0.0022719805341221908),
    24: (0.0027349455847205497, 0.002787904261639144, 0.0028753637798652775, 0.0026341832524516932, 0.0020964203289810263, 0.0026184462284209813, 0.0024671867106328305, 0.0020505567087325273, 0.002438688140916375, 0.0024836355356519174, 0.0025428793084176886, 0.0022023828052609505),
    25: (0.002752464878560165, 0.0024599832990169386, 0.002571493191952811, 0.002482779307439561, 0.0024385119605607386, 0.0024860613246938186, 0.0025290717103851855, 0.002227906149195042, 0.0022412027720091557, 0.002636801046510983, 0.0027874751648259218, 0.0020187452272057266),
    27: (0.0023880762009734447, 0.002730201206126944, 0.0027578633817497418, 0.0025857560724056603, 0.002174396743369525, 0.002353781642727631, 0.002496058805378533, 0.002026336772566295, 0.002424141774644377, 0.002652950632066504, 0.003046179111208362, 0.002046196012095417),
    28: (0.0022848148500655306, 0.0025865018509066244, 0.0028294114222631537, 0.0024204067550990817, 0.002785395728974624, 0.0025419451334576485, 0.002444204216158816, 0.0021065480510346845, 0.002354285854225546, 0.002621425500768443, 0.0033308992728573327, 0.002242788310256564),
    29: (0.002486886710871947, 0.0025979691054280126, 0.002922801342421303, 0.002737821888257918, 0.00244754046058491, 0.002386604827467957, 0.0023956369691669303, 0.002613710043660648, 0.002379760596389121, 0.002619159071761726, 0.0036628931389110185, 0.0024450509611083455),
    31: (0.0026897358697446057, 0.002353693414906293, 0.002574196707820706, 0.002674162971047439, 0.002266546049946765, 0.002337623322941266, 0.0025446546205449352, 0.002507127835190294, 0.0023115351070712538, 0.0025999837122939813, 0.003606264765331881, 0.0020230077056529053),
    32: (0.002456317448726726, 0.0022924237528889266, 0.0028017496025879566, 0.002490085128208982, 0.0026381756800444595, 0.0022066564545820634, 0.002546394318979571, 0.0022941258506470837, 0.002329873758743755, 0.0025699488094483186, 0.0035457299424139646, 0.0020982488252857248),
    33: (0.0026077596744201004, 0.0027471648262570444, 0.00250788342800545, 0.0026329752347248815, 0.0022721615991813846, 0.00235101967572725, 0.0027185292268108686, 0.0022792047267888565, 0.0022713848490229314, 0.00228160463710399, 0.003162951163075944, 0.0022394946844368703),
    35: (0.0023248017442163388, 0.0024532924573500156, 0.002438741728136183, 0.0026390736041951722, 0.0020603315000729634, 0.0024249799310328197, 0.002613217511615673, 0.002089150913107908, 0.00253023614222286, 0.0026075818990840298, 0.003014946188081176, 0.0019792106426056388),
    36: (0.0024212735822278474, 0.002397929905974546, 0.002457138823136085, 0.0030090487741052303, 0.0024064183953313462, 0.0021168497058667575, 0.00263190281758823, 0.0022980244840413184, 0.0027460496479862775, 0.0023714761757666963, 0.0033132193612778814, 0.001995429518072107),
    38: (0.0026181303602272684, 0.0027115418304388904, 0.002350184416770887, 0.002900978391688131, 0.002021525943241395, 0.002475818420041807, 0.002742480087779699, 0.002358649356159528, 0.002743333891140655, 0.0026637800005654866, 0.003077998750764684, 0.0019607227567903112),
    39: (0.0023642224967690357, 0.0024862503630873043, 0.002601331058276756, 0.0027890436113087304, 0.002316321040754986, 0.002336161516891833, 0.002831266069045372, 0.0021315904566877944, 0.0027119249690166006, 0.0024825262602344387, 0.003335256991498775, 0.0022131259466926525),
    40: (0.0025134501716187934, 0.002447676194153005, 0.0025835387668127147, 0.0026279135732968817, 0.0023208970926923223, 0.002283872365407205, 0.002794401476256224, 0.0021669665525403266, 0.0025923679511800648, 0.0024540323565962786, 0.0029618418107283367, 0.002309618670894553),
    42: (0.00268013238903913, 0.002577591196664667, 0.002514837868672324, 0.0027536700827939992, 0.0021881921871760184, 0.0024313452146287203, 0.0026377520537166125, 0.002181869757455677, 0.002545305509691998, 0.002526871859321707, 0.003024443438343065, 0.0024523788080095393),
    43: (0.002897954574665006, 0.0027807258973285634, 0.0027259666550291843, 0.0029928056984027548, 0.0020277920922278456, 0.0026579599604672063, 0.002619249886365231, 0.002073004708811999, 0.0027934809913662505, 0.002525977590628304, 0.0031134944990218395, 0.00229506039823545),
    44: (0.002683546825894141, 0.0030741364540062327, 0.0027060622756768754, 0.002971109243844684, 0.0022374512745014703, 0.00297723143734281, 0.0029822443649044133, 0.0021270916284014743, 0.00271426664365998, 0.0024819040527926275, 0.0023168698828280434, 0.0022683533488712586),
    46: (0.002545742521228993, 0.002345681020387086, 0.002481313122324703, 0.002691048256650328, 0.0023384565698050938, 0.002513523826446871, 0.0021978996631691803, 0.0018566788446174912, 0.002003057218757425, 0.0020866104563996356, 0.002568466103731488, 0.002017207895800852),
    47: (0.002498017833021127, 0.0022705699324012426, 0.002405377883209892, 0.0026312545142079397, 0.002077165492313791, 0.002646851629472633, 0.0021791185254730725, 0.0020124672545276156, 0.0021300614175598248, 0.00215185912611206, 0.002439026476056213, 0.0019221931051372602),
    48: (0.002538139638799778, 0.002354004638798859, 0.0024038687468260806, 0.0027895588607861595, 0.0020926669901482277, 0.002787394707319404, 0.0023681484380110537, 0.001764962465147142, 0.0021807914373747762, 0.002182680667225665, 0.002564047789038969, 0.0020330121700590036),
    50: (0.0024584483041148113, 0.002361233434910064, 0.0024338422529806966, 0.002883663190628444, 0.0019385173328166899, 0.002637824840333317, 0.002318903529599446, 0.002062304359366026, 0.002407313501382151, 0.0021474393495840524, 0.0019667995909767007, 0.0017810586555737848),
    51: (0.002691552824985619, 0.002713937516040422, 0.0026938967187466736, 0.0029187589752321702, 0.0022070189260488178, 0.003031127458294667, 0.002384749473859289, 0.002122088116079379, 0.0031150456045714267, 0.0024239730350754315, 0.001395550149768043, 0.0013875772525534247),
    53: (0.0024814711793254883, 0.0020701468343412026, 0.0025429903366118944, 0.0024383076356001075, 0.002229192592088531, 0.0028541228782905065, 0.002396129671489138, 0.0016251672827030785, 0.0022930168031468513, 0.002307771271811566, 0.0009562024712233064, 0.0011346194327245484),
    54: (0.0023731700526523863, 0.0019712273148604293, 0.0025659289341866008, 0.0030850485554841326, 0.0021751245648526907, 0.0028963115639006006, 0.0025131799314543945, 0.001564979312257581, 0.0029614293454628335, 0.0023695841529665675, 0.0009399964916864396, 0.0008424681530381236),
    55: (0.0026960189159994275, 0.002251591853812684, 0.0027699715411616797, 0.0030185282207590997, 0.002211619931268166, 0.003203631154913399, 0.0028002710739326773, 0.001591673993314853, 0.0027773785920431706, 0.0024020932525220242, 0.0008397919535970211, 0.0007630049117981413),
    57: (0.0029817181316883533, 0.002836173256550784, 0.002864986271490559, 0.0029607615685435006, 0.0019454450787252098, 0.00369038276102609, 0.0040598036523124305, 0.0012646621507439607, 0.003937904584385148, 0.0027719954334788255, 0.0007644622616464682, 0.0008390876703701519),
    58: (0.003190931403427026, 0.0026977415731364693, 0.0028983642331826456, 0.003003619447376456, 0.0019286690184963896, 0.003702437646635074, 0.003431264605972858, 0.0008316891224212725, 0.00438772098151635, 0.002176287723675404, 0.0008369203342371133, 0.0007610109174844355),
    59: (0.0032662438659450776, 0.003162164313609544, 0.0026539804512642476, 0.0015904213748552218, 0.00023325320162226177, 0.004399239131724294, 0.004491409180889158, 0.0002510949541410877, 0.002056552859437852, 0.0007595557704850559, 0.00019838626122057538, 0.00021877486614826428),
}
S30X = (1, 5, 9, 13, 17, 20, 24, 28, 32, 35, 39, 43, 47, 50, 54)   # S15[i] - 2: one more stratified block per stratum of 4
D15X_SETS = {"S60": tuple(range(60)), "S30": tuple(sorted(S15 + S30X)), "S15": S15}


def fits_table_x():
    """fits_table() + the 540 FITS_X entries: {tensor name: ('abs', sd)} for the 6 io + all 720 attn / MLP matrices (T249 D15X)"""
    out = fits_table()
    for b, vals in FITS_X.items():
        for mod, v in zip(BLOCK_MODULES, vals):
            out[f"transformer_blocks.{b}.{mod}.weight"] = ("abs", float(v))
    return out
