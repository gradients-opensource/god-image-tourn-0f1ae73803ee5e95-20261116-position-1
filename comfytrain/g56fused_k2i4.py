"""g56fused_k2i4 (T251, 01 Nov 2026): fused krea2 (Krea-2 single-stream DiT) block pieces for TRAINING - G56CT_K2FUSE.

ComfyUI 694815f's krea2 SingleStreamBlock (comfy/ldm/krea2/model.py) runs every non-GEMM piece as eager ops, and autograd mirrors each:
  norm   (1 + scale) * RMSNorm(x) + shift, twice per block; the norm in fp32 (x.float(), weight scale + 1, eps 1e-5, then .to(dtype))
  qkv    the per-head QK RMSNorm (fp32 the same way) + the training RoPE (_apply_rope1: fp32 copy, mul, addcmul_, cast back) + the GQA
         repeat_interleave of k and v (12 -> 48 heads)
  sgate  out * sigmoid(gate) after the attention
  swiglu silu(gate(x)).mul_(up(x)) in the MLP
  res    x + gate * y, twice per block (g56fused.gate_res: the same rounding as eager)
Each piece here is ONE Triton kernel forward and ONE backward, with eager's roundings kept: the fp32 norm output rounded to the input dtype,
the product with (1 + scale) rounded, the add of shift rounded; the RoPE in fp32 rounded once; the repeat as stores (forward) and an fp32 sum
rounded once (backward); sigmoid / silu rounded to the dtype before the product, as eager. Backward: the eager formulas in fp32 from the
saved inputs (norm: rstd g - x rstd^3 / C sum(g x), g = (dy * s1 rounded) * w; the modulation vectors' gradients = fp32 column sums of the
bf16-rounded products as eager's sum_to_size). Not bit-identical to eager (reduction order, FMA contraction): judged per op against an fp32
reference on the run's own activations (gate_begin / gate_end: the online per-op gate, T248's design) and by the run's score.

Norm weights are frozen base weights (the LoRA trains Linears only): no weight gradient. Only CUDA bf16 / fp16; anything else, or any
failed shape check, takes the eager branch of that call (counted in STATS[<piece>_eager]). install_krea2(parts) patches the
SingleStreamBlock class forward (TextFusionBlocks keep ComfyUI's); uninstall_krea2() restores it."""
import math
import os
import time

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  (no Triton: every piece stays eager)
    triton = None
    tl = None

PARTS = ("norm", "qkv", "res", "sgate", "swiglu")
FLAGS = {p: False for p in PARTS}
STATS = {}
_FALLBACK = {}
_INTERP = os.environ.get("TRITON_INTERPRET") == "1"  # the host CPU tests (Triton interpreter): CPU tensors allowed
_ROWS = 16
_EBLK = 1024


def _count(k):
    STATS[k] = STATS.get(k, 0) + 1


def _dev_ok(t):
    return t.is_cuda or (_INTERP and t.device.type == "cpu")


def _dt_ok(t):
    return t.dtype in (torch.bfloat16, torch.float16) or (_INTERP and t.dtype == torch.float32)


# ----------------------------------------------------------------------------------------------------------------------------- kernels
if triton is not None:
    @triton.jit
    def _k2_nm_fwd(X, s_xb, s_xl, W, S1, s_sb, SH, s_hb, OUT, RSTD, L, C, eps, BLOCK: tl.constexpr):
        l = tl.program_id(0)
        b = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        x = tl.load(X + b * s_xb + l * s_xl + cols, mask=m, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / C + eps)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        y = (x * rstd * w).to(OUT.dtype.element_ty).to(tl.float32)  # F.rms_norm(x.float(), w).to(dtype)
        s1 = tl.load(S1 + b * s_sb + cols, mask=m, other=0.0).to(tl.float32)
        t = (s1 * y).to(OUT.dtype.element_ty).to(tl.float32)  # (1 + scale) * n, rounded
        sh = tl.load(SH + b * s_hb + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + (b * L + l) * C + cols, (t + sh).to(OUT.dtype.element_ty), mask=m)  # + shift, rounded
        tl.store(RSTD + b * L + l, rstd)

    @triton.jit
    def _k2_nm_bwd(X, s_xb, s_xl, W, S1, s_sb, DO, RSTD, DX, PS, PH, L, C, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        pblk = tl.program_id(0)
        b = tl.program_id(1)
        nblk = tl.num_programs(0)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        s1 = tl.load(S1 + b * s_sb + cols, mask=m, other=0.0).to(tl.float32)
        acc_s = tl.zeros((BLOCK,), dtype=tl.float32)
        acc_h = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in range(ROWS):
            l = pblk * ROWS + r
            if l < L:
                x = tl.load(X + b * s_xb + l * s_xl + cols, mask=m, other=0.0).to(tl.float32)
                rstd = tl.load(RSTD + b * L + l)
                y = (x * rstd * w).to(DX.dtype.element_ty).to(tl.float32)
                d = tl.load(DO + (b * L + l) * C + cols, mask=m, other=0.0).to(tl.float32)
                acc_h += d  # d shift: the column sum (eager sum_to_size)
                acc_s += (d * y).to(DX.dtype.element_ty).to(tl.float32)  # d (1 + scale): the bf16 product, then the fp32 sum
                gy = tl.where(m, (d * s1).to(DX.dtype.element_ty).to(tl.float32) * w, 0.0)
                dx = rstd * gy - x * (rstd * rstd * rstd / C) * tl.sum(gy * x, axis=0)
                tl.store(DX + (b * L + l) * C + cols, dx.to(DX.dtype.element_ty), mask=m)
        tl.store(PS + (b * nblk + pblk) * C + cols, acc_s, mask=m)
        tl.store(PH + (b * nblk + pblk) * C + cols, acc_h, mask=m)

    @triton.jit
    def _k2_qkv_fwd(Q, s_qb, s_ql, K, s_kb, s_kl, V, s_vb, s_vl, WQ, WK, PE, pe_b, pe_l, QO, KO, VO, L, H, eps_q, eps_k,
                    REP: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr):
        pid = tl.program_id(0)
        g = tl.program_id(1)
        b = pid // L
        l = pid % L
        offs = tl.arange(0, D)
        i = tl.arange(0, HALF)
        pe = PE + b * pe_b + l * pe_l
        f00 = tl.load(pe + i * 4 + 0)
        f01 = tl.load(pe + i * 4 + 1)
        f10 = tl.load(pe + i * 4 + 2)
        f11 = tl.load(pe + i * 4 + 3)
        wq = tl.load(WQ + offs).to(tl.float32)
        wk = tl.load(WK + offs).to(tl.float32)
        k = tl.load(K + b * s_kb + l * s_kl + g * D + offs).to(tl.float32)
        rk = tl.rsqrt(tl.sum(k * k, axis=0) / D + eps_k)
        yk = (k * rk * wk).to(KO.dtype.element_ty).to(tl.float32)
        k0, k1 = tl.split(tl.reshape(yk, (HALF, 2)))
        ok = tl.reshape(tl.join(f00 * k0 + f01 * k1, f10 * k0 + f11 * k1), (D,)).to(KO.dtype.element_ty)
        v = tl.load(V + b * s_vb + l * s_vl + g * D + offs)
        for r in tl.static_range(REP):
            h = g * REP + r
            q = tl.load(Q + b * s_qb + l * s_ql + h * D + offs).to(tl.float32)
            rq = tl.rsqrt(tl.sum(q * q, axis=0) / D + eps_q)
            yq = (q * rq * wq).to(QO.dtype.element_ty).to(tl.float32)
            q0, q1 = tl.split(tl.reshape(yq, (HALF, 2)))
            oq = tl.reshape(tl.join(f00 * q0 + f01 * q1, f10 * q0 + f11 * q1), (D,))
            dst = ((b * H + h) * L + l) * D
            tl.store(QO + dst + offs, oq.to(QO.dtype.element_ty))
            tl.store(KO + dst + offs, ok)
            tl.store(VO + dst + offs, v.to(VO.dtype.element_ty))

    @triton.jit
    def _k2_qkv_bwd(Q, s_qb, s_ql, K, s_kb, s_kl, WQ, WK, PE, pe_b, pe_l, DQO, DKO, DVO, DQ, DK, DV, L, H, KVH, eps_q, eps_k,
                    REP: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr):
        pid = tl.program_id(0)
        g = tl.program_id(1)
        b = pid // L
        l = pid % L
        offs = tl.arange(0, D)
        i = tl.arange(0, HALF)
        pe = PE + b * pe_b + l * pe_l
        f00 = tl.load(pe + i * 4 + 0)
        f01 = tl.load(pe + i * 4 + 1)
        f10 = tl.load(pe + i * 4 + 2)
        f11 = tl.load(pe + i * 4 + 3)
        wq = tl.load(WQ + offs).to(tl.float32)
        wk = tl.load(WK + offs).to(tl.float32)
        acc_k = tl.zeros((D,), dtype=tl.float32)
        acc_v = tl.zeros((D,), dtype=tl.float32)
        for r in tl.static_range(REP):  # the repeat's backward: the REP head gradients summed in fp32, rounded once
            src = ((b * H + g * REP + r) * L + l) * D
            acc_k += tl.load(DKO + src + offs).to(tl.float32)
            acc_v += tl.load(DVO + src + offs).to(tl.float32)
        tl.store(DV + (b * L + l) * (KVH * D) + g * D + offs, acc_v.to(DV.dtype.element_ty))
        dkr = acc_k.to(DK.dtype.element_ty).to(tl.float32)
        d0, d1 = tl.split(tl.reshape(dkr, (HALF, 2)))
        dyk = tl.reshape(tl.join(f00 * d0 + f10 * d1, f01 * d0 + f11 * d1), (D,)).to(DK.dtype.element_ty).to(tl.float32)
        k = tl.load(K + b * s_kb + l * s_kl + g * D + offs).to(tl.float32)
        rk = tl.rsqrt(tl.sum(k * k, axis=0) / D + eps_k)
        gk = dyk * wk
        dk = rk * gk - k * (rk * rk * rk / D) * tl.sum(gk * k, axis=0)
        tl.store(DK + (b * L + l) * (KVH * D) + g * D + offs, dk.to(DK.dtype.element_ty))
        for r in tl.static_range(REP):
            h = g * REP + r
            dq = tl.load(DQO + ((b * H + h) * L + l) * D + offs).to(tl.float32)
            e0, e1 = tl.split(tl.reshape(dq, (HALF, 2)))
            dyq = tl.reshape(tl.join(f00 * e0 + f10 * e1, f01 * e0 + f11 * e1), (D,)).to(DQ.dtype.element_ty).to(tl.float32)
            q = tl.load(Q + b * s_qb + l * s_ql + h * D + offs).to(tl.float32)
            rq = tl.rsqrt(tl.sum(q * q, axis=0) / D + eps_q)
            gq = dyq * wq
            dqx = rq * gq - q * (rq * rq * rq / D) * tl.sum(gq * q, axis=0)
            tl.store(DQ + (b * L + l) * (H * D) + h * D + offs, dqx.to(DQ.dtype.element_ty))

    @triton.jit
    def _k2_sg_fwd(O, G, Y, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        o = tl.load(O + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        s = tl.sigmoid(g).to(Y.dtype.element_ty).to(tl.float32)  # F.sigmoid(gate), rounded
        tl.store(Y + offs, (o * s).to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _k2_sg_bwd(O, G, DY, DO, DG, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        o = tl.load(O + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        d = tl.load(DY + offs, mask=m, other=0.0).to(tl.float32)
        s = tl.sigmoid(g).to(DO.dtype.element_ty).to(tl.float32)
        tl.store(DO + offs, (d * s).to(DO.dtype.element_ty), mask=m)
        ds = (d * o).to(DG.dtype.element_ty).to(tl.float32)
        tl.store(DG + offs, (ds * (1.0 - s) * s).to(DG.dtype.element_ty), mask=m)  # sigmoid_backward(ds, s)

    @triton.jit
    def _k2_sw_fwd(A, Bm, Y, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
        bb = tl.load(Bm + offs, mask=m, other=0.0).to(tl.float32)
        sa = (a * tl.sigmoid(a)).to(Y.dtype.element_ty).to(tl.float32)  # F.silu(a), rounded
        tl.store(Y + offs, (sa * bb).to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _k2_sw_bwd(A, Bm, DY, DA, DB, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
        bb = tl.load(Bm + offs, mask=m, other=0.0).to(tl.float32)
        d = tl.load(DY + offs, mask=m, other=0.0).to(tl.float32)
        sg = tl.sigmoid(a)
        sa = (a * sg).to(DA.dtype.element_ty).to(tl.float32)
        tl.store(DB + offs, (d * sa).to(DB.dtype.element_ty), mask=m)
        dsa = (d * bb).to(DA.dtype.element_ty).to(tl.float32)
        tl.store(DA + offs, (dsa * sg * (1.0 + a * (1.0 - sg))).to(DA.dtype.element_ty), mask=m)  # silu_backward(dsa, a)


def _blk(C):
    return triton.next_power_of_2(C)


# ----------------------------------------------------------------------------------------------------------------------- autograd ops
class _NormMod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w32, s1, sh, eps):
        B, L, C = x.shape
        out = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
        rstd = torch.empty((B, L), dtype=torch.float32, device=x.device)
        _k2_nm_fwd[(L, B)](x, x.stride(0), x.stride(1), w32, s1, s1.stride(0), sh, sh.stride(0), out, rstd, L, C, eps,
                           BLOCK=_blk(C), num_warps=8)
        ctx.save_for_backward(x, w32, s1, rstd)
        return out

    @staticmethod
    def backward(ctx, do):
        x, w32, s1, rstd = ctx.saved_tensors
        B, L, C = x.shape
        do = do.contiguous()
        nblk = triton.cdiv(L, _ROWS)
        dx = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
        ps = torch.empty((B, nblk, C), dtype=torch.float32, device=x.device)
        ph = torch.empty_like(ps)
        _k2_nm_bwd[(nblk, B)](x, x.stride(0), x.stride(1), w32, s1, s1.stride(0), do, rstd, dx, ps, ph, L, C, ROWS=_ROWS,
                              BLOCK=_blk(C), num_warps=8)
        ds = ps.sum(dim=1, keepdim=True).to(s1.dtype) if ctx.needs_input_grad[2] else None
        dh = ph.sum(dim=1, keepdim=True).to(x.dtype) if ctx.needs_input_grad[3] else None
        return dx, None, ds, dh, None


class _QKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, wq, wk, pe, H, KVH, eps_q, eps_k):
        B, L, _ = q.shape
        D = q.shape[2] // H
        rep = H // KVH
        qo = torch.empty((B, H, L, D), dtype=q.dtype, device=q.device)
        ko = torch.empty_like(qo)
        vo = torch.empty_like(qo)
        _k2_qkv_fwd[(B * L, KVH)](q, q.stride(0), q.stride(1), k, k.stride(0), k.stride(1), v, v.stride(0), v.stride(1), wq, wk, pe,
                                   pe.stride(0) if pe.shape[0] > 1 else 0, pe.stride(2), qo, ko, vo, L, H, eps_q, eps_k,
                                   REP=rep, D=D, HALF=D // 2, num_warps=4)
        ctx.save_for_backward(q, k, wq, wk, pe)
        ctx.hk = (H, KVH, eps_q, eps_k)
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dqo, dko, dvo):
        q, k, wq, wk, pe = ctx.saved_tensors
        H, KVH, eps_q, eps_k = ctx.hk
        B, L, _ = q.shape
        D = q.shape[2] // H
        z = None
        dqo = torch.zeros((B, H, L, D), dtype=q.dtype, device=q.device) if dqo is None else dqo.contiguous()
        dko = torch.zeros_like(dqo) if dko is None else dko.contiguous()
        dvo = torch.zeros_like(dqo) if dvo is None else dvo.contiguous()
        dq = torch.empty((B, L, H * D), dtype=q.dtype, device=q.device)
        dk = torch.empty((B, L, KVH * D), dtype=k.dtype, device=k.device)
        dv = torch.empty_like(dk)
        _k2_qkv_bwd[(B * L, KVH)](q, q.stride(0), q.stride(1), k, k.stride(0), k.stride(1), wq, wk, pe,
                                   pe.stride(0) if pe.shape[0] > 1 else 0, pe.stride(2), dqo, dko, dvo, dq, dk, dv, L, H, KVH, eps_q, eps_k,
                                   REP=H // KVH, D=D, HALF=D // 2, num_warps=4)
        return dq, dk, dv, z, z, z, z, z, z, z


class _SGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, g):
        y = torch.empty_like(o)
        n = o.numel()
        _k2_sg_fwd[(triton.cdiv(n, _EBLK),)](o, g, y, n, BLOCK=_EBLK, num_warps=4)
        ctx.save_for_backward(o, g)
        return y

    @staticmethod
    def backward(ctx, dy):
        o, g = ctx.saved_tensors
        dy = dy.contiguous()
        do, dg = torch.empty_like(o), torch.empty_like(g)
        n = o.numel()
        _k2_sg_bwd[(triton.cdiv(n, _EBLK),)](o, g, dy, do, dg, n, BLOCK=_EBLK, num_warps=4)
        return do, dg


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        y = torch.empty_like(a)
        n = a.numel()
        _k2_sw_fwd[(triton.cdiv(n, _EBLK),)](a, b, y, n, BLOCK=_EBLK, num_warps=4)
        ctx.save_for_backward(a, b)
        return y

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        dy = dy.contiguous()
        da, db = torch.empty_like(a), torch.empty_like(b)
        n = a.numel()
        _k2_sw_bwd[(triton.cdiv(n, _EBLK),)](a, b, dy, da, db, n, BLOCK=_EBLK, num_warps=4)
        return da, db


# --------------------------------------------------------------------------------------------- eager references (= ComfyUI 694815f)
_W32 = {}


def _w32(norm, device):
    """RMSNorm.forward's weight: cast_to(scale, fp32) + 1.0 (frozen base weight: cached per tensor storage / version / device)"""
    sc = norm.scale
    key = (id(norm), sc.data_ptr(), getattr(sc, "_version", 0), str(device), sc.dtype)
    w = _W32.get(key)
    if w is None:
        if len(_W32) > 4096:
            _W32.clear()
        w = _W32[key] = sc.detach().to(device=device, dtype=torch.float32) + 1.0
    return w


def normmod_eager(norm, x, scale, shift):
    return (1 + scale) * norm(x) + shift


def qkv_eager(attn, q, k, v, freqs):
    """Attention.forward from the projections to the SDPA inputs (rearrange, qknorm, RoPE, GQA repeat)"""
    from einops import rearrange
    from comfy.ldm.flux.math import apply_rope
    q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if attn.kvheads != attn.heads:
        rep = attn.heads // attn.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return q, k, v


def sgate_eager(o, g):
    return o * F.sigmoid(g)


def swiglu_eager(a, b):
    return F.silu(a).mul_(b)


def res_eager(x, y, g):
    return x + g * y


# ------------------------------------------------------------------------------------------------- dispatch (fused when possible)
def _vec_ok(v, x):
    return (v is not None and v.dtype == x.dtype and v.device == x.device and v.dim() == 3 and v.shape[0] == x.shape[0]
            and v.shape[1] == 1 and v.shape[2] == x.shape[2] and v.stride(2) == 1)


def normmod_ok(norm, x, scale, shift):
    if triton is None:
        return "no triton"
    if not (_dev_ok(x) and _dt_ok(x) and x.dim() == 3 and x.stride(2) == 1):
        return "x"
    if not (_vec_ok(scale, x) and _vec_ok(shift, x)):
        return "modulation vector"
    sc = getattr(norm, "scale", None)
    if sc is None or tuple(sc.shape) != (x.shape[2],) or getattr(sc, "requires_grad", False) or not isinstance(getattr(norm, "eps", None), float):
        return "norm"
    if x.shape[2] > 16384:
        return "width"
    return None


def normmod(norm, x, scale, shift):
    """(1 + scale) * norm(x) + shift (krea2 RMSNorm, (B, 1, C) scale / shift)"""
    if FLAGS.get("norm") and normmod_ok(norm, x, scale, shift) is None:
        _count("norm")
        if _REC is not None:
            _REC.note("norm", (norm, x, scale, shift))
        return _NormMod.apply(x, _w32(norm, x.device), 1 + scale, shift, float(norm.eps))
    _count("norm_eager")
    return normmod_eager(norm, x, scale, shift)


def qkv_ok(attn, q, k, v, freqs):
    if triton is None:
        return "no triton"
    H, KVH = int(attn.heads), int(attn.kvheads)
    if H % KVH or H // KVH > 8:
        return "heads"
    B, L, HD = q.shape if q.dim() == 3 else (0, 0, 0)
    D = HD // H if H else 0
    if not (q.dim() == 3 and _dev_ok(q) and _dt_ok(q) and D in (64, 128, 256) and H * D == HD):
        return "q"
    for t in (k, v):
        if not (t.dim() == 3 and t.shape[:2] == q.shape[:2] and t.shape[2] == KVH * D and t.dtype == q.dtype and t.device == q.device
                and t.stride(2) == 1):
            return "k / v"
    if q.stride(2) != 1:
        return "q"
    if (freqs is None or freqs.dtype != torch.float32 or freqs.device != q.device or freqs.dim() != 6 or freqs.shape[0] not in (1, B)
            or freqs.shape[1] != 1 or freqs.shape[2] != L or tuple(freqs.shape[3:]) != (D // 2, 2, 2) or not freqs.is_contiguous()):
        return "freqs"
    for n in (attn.qknorm.qnorm, attn.qknorm.knorm):
        sc = getattr(n, "scale", None)
        if sc is None or tuple(sc.shape) != (D,) or getattr(sc, "requires_grad", False) or not isinstance(getattr(n, "eps", None), float):
            return "qk norm"
    return None


def qkv(attn, q, k, v, freqs):
    """the SDPA inputs (B, H, L, D) from the q / k / v projections"""
    if FLAGS.get("qkv") and qkv_ok(attn, q, k, v, freqs) is None:
        _count("qkv")
        if _REC is not None:
            _REC.note("qkv", (attn, q, k, v, freqs))
        nq, nk = attn.qknorm.qnorm, attn.qknorm.knorm
        return _QKV.apply(q, k, v, _w32(nq, q.device), _w32(nk, q.device), freqs, int(attn.heads), int(attn.kvheads), float(nq.eps),
                          float(nk.eps))
    _count("qkv_eager")
    return qkv_eager(attn, q, k, v, freqs)


def _ew_ok(a, b):
    if triton is None:
        return "no triton"
    if not (_dev_ok(a) and _dt_ok(a) and a.is_contiguous() and b.is_contiguous() and a.shape == b.shape and a.dtype == b.dtype
            and a.device == b.device):
        return "inputs"
    return None


def sgate(o, g):
    if FLAGS.get("sgate") and _ew_ok(o, g) is None:
        _count("sgate")
        if _REC is not None:
            _REC.note("sgate", (o, g))
        return _SGate.apply(o, g)
    _count("sgate_eager")
    return sgate_eager(o, g)


def swiglu(a, b):
    if FLAGS.get("swiglu") and _ew_ok(a, b) is None:
        _count("swiglu")
        if _REC is not None:
            _REC.note("swiglu", (a, b))
        return _SwiGLU.apply(a, b)
    _count("swiglu_eager")
    return swiglu_eager(a, b)


def _res_ok(x, y, g):
    if triton is None:
        return "no triton"
    if not (_dev_ok(x) and _dt_ok(x) and x.dim() == 3 and x.stride(2) == 1 and y.shape == x.shape and y.dtype == x.dtype
            and y.stride(2) == 1 and _vec_ok(g, x)):
        return "inputs"
    return None


def res(x, y, g):
    """x + g * y (g56fused.gate_res: y * g rounded, then the add rounded - eager's two roundings)"""
    if FLAGS.get("res") and _res_ok(x, y, g) is None:
        import g56fused
        _count("res")
        if _REC is not None:
            _REC.note("res", (x, y, g))
        return g56fused._GateRes.apply(x, y, g)
    _count("res_eager")
    return res_eager(x, y, g)


# ---------------------------------------------------------------------------------------------------------- the patched block
def _k2_attn(attn, x, freqs, mask, transformer_options):
    from comfy.ldm.krea2 import model as M
    q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
    q, k, v = qkv(attn, q, k, v, freqs)
    out = M.optimized_attention_masked(q, k, v, attn.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
    return attn.wo(sgate(out, gate))


def _k2_block_forward(self, x, vec, freqs, mask=None, transformer_options={}):  # noqa: B006  (ComfyUI's signature)
    prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
    a = _k2_attn(self.attn, normmod(self.prenorm, x, prescale, preshift), freqs, mask, transformer_options)
    x = res(x, a, pregate)
    h = normmod(self.postnorm, x, postscale, postshift)
    mlp = self.mlp
    x = res(x, mlp.down(swiglu(mlp.gate(h), mlp.up(h))), postgate)
    return x


_k2_block_forward._g56_fused = True


def install_krea2(parts):
    """set FLAGS for parts and patch SingleStreamBlock.forward (idempotent) -> the parts on"""
    from comfy.ldm.krea2 import model as M
    for p in PARTS:
        FLAGS[p] = p in parts
    if "krea2_block" not in _FALLBACK:
        _FALLBACK["krea2_block"] = M.SingleStreamBlock.forward
    M.SingleStreamBlock.forward = _k2_block_forward
    return [p for p in PARTS if FLAGS[p]]


def uninstall_krea2():
    _W32.clear()
    for p in PARTS:
        FLAGS[p] = False
    fn = _FALLBACK.pop("krea2_block", None)
    if fn is not None:
        from comfy.ldm.krea2 import model as M
        M.SingleStreamBlock.forward = fn


# ---------------------------------------------------------------------------------------------- self-check (synthetic, per part)
def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


class _Norm(torch.nn.Module):
    """krea2's RMSNorm stand-in (the same forward as comfy.ldm.krea2.model.RMSNorm)"""

    def __init__(self, n, eps=1e-5, device=None, dtype=None, gen=None):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter((0.1 * torch.randn(n, generator=gen)).to(device=device, dtype=dtype), requires_grad=False)

    def forward(self, x):
        w = self.scale.to(torch.float32) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=w, eps=self.eps).to(x.dtype)


def selfcheck(device, parts=PARTS, dtype=torch.bfloat16, seed=0, tol=2e-2, L=48, C=256, H=4, KVH=2, D=128):
    """fused vs ComfyUI's eager composition per part, forward + backward, synthetic shapes -> {part: (ok, {name: rel})}"""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def r(*s, scale=1.0):
        return (scale * torch.randn(*s, generator=gen)).to(device=device, dtype=dtype)

    out = {}
    saved = dict(FLAGS)
    try:
        for part in parts:
            try:
                rep = _selfcheck_one(part, device, dtype, r, gen, L, C, H, KVH, D)
                out[part] = (all(v <= tol for v in rep.values()), rep)
            except Exception as e:  # noqa: BLE001  (that part stays eager)
                out[part] = (False, {"error": f"{type(e).__name__}: {' '.join(str(e).split())[:120]}"})
    finally:
        FLAGS.update(saved)
    return out


def _grads(fn, leaves, gen, device):
    for t in leaves:
        t.grad = None
    y = fn()
    ys = list(y) if isinstance(y, (tuple, list)) else [y]
    ups = [(torch.randn(t.shape, generator=gen)).to(device=device, dtype=t.dtype) for t in ys]
    torch.autograd.backward(ys, ups)
    return [t.detach().clone() for t in ys], [None if t.grad is None else t.grad.detach().clone() for t in leaves]


def _selfcheck_one(part, device, dtype, r, gen, L, C, H, KVH, D):
    rep = {}
    gseed = torch.Generator(device="cpu")

    def run(fn, leaves):
        gseed.manual_seed(7)
        return _grads(fn, leaves, gseed, device)

    def cmp(fused_fn, eager_fn, leaves):
        lf = [t.detach().clone().requires_grad_(t.requires_grad) for t in leaves]
        le = [t.detach().clone().requires_grad_(t.requires_grad) for t in leaves]
        fo, fg = run(lambda: fused_fn(*lf), lf)
        eo, eg = run(lambda: eager_fn(*le), le)
        for i, (a, b) in enumerate(zip(fo, eo)):
            rep[f"out{i}"] = _rel(a, b)
        for i, (a, b) in enumerate(zip(fg, eg)):
            if a is not None or b is not None:
                rep[f"d{i}"] = _rel(a, b) if (a is not None and b is not None) else float("inf")

    if part == "norm":
        nrm = _Norm(C, device=device, dtype=dtype, gen=gen)
        x, s, h = r(1, L, C, scale=2.0).requires_grad_(), r(1, 1, C, scale=0.3).requires_grad_(), r(1, 1, C).requires_grad_()
        FLAGS["norm"] = True
        if normmod_ok(nrm, x, s, h) is not None:
            raise RuntimeError(f"refused: {normmod_ok(nrm, x, s, h)}")
        cmp(lambda a, b, c: normmod(nrm, a, b, c), lambda a, b, c: normmod_eager(nrm, a, b, c), [x, s, h])
    elif part == "qkv":
        class _QK(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qnorm, self.knorm = _Norm(D, device=device, dtype=dtype, gen=gen), _Norm(D, device=device, dtype=dtype, gen=gen)

            def forward(self, a, b):
                return self.qnorm(a), self.knorm(b)

        attn = torch.nn.Module()
        attn.heads, attn.kvheads, attn.qknorm = H, KVH, _QK()
        q, k, v = r(1, L, H * D, scale=2.0).requires_grad_(), r(1, L, KVH * D, scale=2.0).requires_grad_(), r(1, L, KVH * D).requires_grad_()
        ang = torch.rand(1, 1, L, D // 2, generator=gen) * 6.2832
        pe = torch.stack([torch.cos(ang), -torch.sin(ang), torch.sin(ang), torch.cos(ang)], dim=-1).reshape(1, 1, L, D // 2, 2, 2)
        pe = pe.to(device=device, dtype=torch.float32).contiguous()
        FLAGS["qkv"] = True
        if qkv_ok(attn, q, k, v, pe) is not None:
            raise RuntimeError(f"refused: {qkv_ok(attn, q, k, v, pe)}")
        cmp(lambda a, b, c: qkv(attn, a, b, c, pe), lambda a, b, c: _qkv_eager_local(attn, a, b, c, pe), [q, k, v])
    elif part == "sgate":
        o, g = r(1, L, C).requires_grad_(), r(1, L, C, scale=2.0).requires_grad_()
        FLAGS["sgate"] = True
        cmp(sgate, sgate_eager, [o, g])
    elif part == "swiglu":
        a, b = r(1, L, C, scale=2.0).requires_grad_(), r(1, L, C).requires_grad_()
        FLAGS["swiglu"] = True
        cmp(swiglu, lambda x, y: F.silu(x) * y, [a, b])
    elif part == "res":
        x, y, g = r(1, L, C).requires_grad_(), r(1, L, C).requires_grad_(), r(1, 1, C, scale=0.5).requires_grad_()
        FLAGS["res"] = True
        cmp(res, res_eager, [x, y, g])
    else:
        raise ValueError(part)
    return rep


def _qkv_eager_local(attn, q, k, v, pe):
    """qkv_eager without the comfy import (the same math: _apply_rope1 + rearrange + repeat_interleave), for the synthetic self-check"""
    B, L, _ = q.shape
    H, KVH = attn.heads, attn.kvheads
    D = q.shape[2] // H
    q = q.view(B, L, H, D).transpose(1, 2)
    k = k.view(B, L, KVH, D).transpose(1, 2)
    v = v.view(B, L, KVH, D).transpose(1, 2)
    q, k = attn.qknorm(q, k)

    def rope1(x):
        x_ = x.to(dtype=pe.dtype).reshape(*x.shape[:-1], -1, 1, 2)
        x_out = pe[..., 0] * x_[..., 0]
        x_out.addcmul_(pe[..., 1], x_[..., 1])
        return x_out.reshape(*x.shape).type_as(x)

    q, k = rope1(q), rope1(k)
    rep = H // KVH
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return q, k, v


# ----------------------------------------------------------------------------------------- the online per-op gate (T248's design)
GATE_TOL = 1e-2  # rel(fused, fp32) <= max(GATE_TOL, GATE_X x rel(eager, fp32)) for every checked output / input gradient
GATE_X = 1.5
GATE_PER = 64  # checked calls per (piece, input shapes)
GATE_BUDGET = 120.0  # seconds of checks per run
_REC = None


class _Rec(object):
    def __init__(self):
        self.res, self.seen, self.spent, self.checked, self.counted, self.busy = {}, {}, 0.0, 0, 0, False

    def note(self, name, a):
        if self.busy:
            return
        try:
            if torch._C._current_graph_task_id() != -1:  # a recompute inside a backward: its forward call was checked
                return
        except Exception:  # noqa: BLE001
            pass
        key = (name,) + tuple(tuple(t.shape) for t in a if isinstance(t, torch.Tensor))
        self.seen[key] = self.seen.get(key, 0) + 1
        if self.seen[key] > GATE_PER or self.spent > GATE_BUDGET:
            self.counted += 1
            return
        t0 = time.time()
        self.busy = True
        r = self.res.setdefault(name, {"ok": True, "n": 0, "worst": 0.0, "eager_at_worst": 0.0, "why": ""})
        try:
            if _check_call(name, a, r):
                self.checked += 1
        except Exception as e:  # noqa: BLE001  (a check that raises fails its piece)
            r["ok"], r["why"] = False, f"{name}: check raised {type(e).__name__}: {' '.join(str(e).split())[:120]}"
        finally:
            self.busy = False
            self.spent += time.time() - t0


def _replay(name, a, path):
    """one call replayed on fresh leaves: path 'fused' | 'eager' | 'fp32' -> (outputs, input gradients)"""
    def leaf(t, dt):
        return t.detach().to(dt).clone().requires_grad_(True)
    dt0 = next(t for t in a if isinstance(t, torch.Tensor)).dtype
    dt = torch.float32 if path == "fp32" else dt0
    saved = FLAGS.get(name)
    FLAGS[name] = path == "fused"
    try:
        if name == "norm":
            norm, x, s, h = a
            lv = [leaf(x, dt), leaf(s, dt), leaf(h, dt)]
            if path == "fp32":
                w = _w32(norm, x.device)
                out = [(1 + lv[1]) * F.rms_norm(lv[0], (lv[0].shape[-1],), weight=w, eps=float(norm.eps)) + lv[2]]
            else:
                out = [normmod(norm, *lv) if path == "fused" else normmod_eager(norm, *lv)]
        elif name == "qkv":
            attn, q, k, v, pe = a
            lv = [leaf(q, dt), leaf(k, dt), leaf(v, dt)]
            if path == "fp32":
                out = list(_qkv_fp32(attn, *lv, pe))
            else:
                out = list(qkv(attn, *lv, pe) if path == "fused" else _qkv_eager_local(attn, *lv, pe))
        elif name == "sgate":
            lv = [leaf(t, dt) for t in a]
            out = [sgate(*lv) if path == "fused" else sgate_eager(*lv)]
        elif name == "swiglu":
            lv = [leaf(t, dt) for t in a]
            out = [swiglu(*lv) if path == "fused" else F.silu(lv[0]) * lv[1]]
        elif name == "res":
            lv = [leaf(t, dt) for t in a]
            out = [res(*lv) if path == "fused" else res_eager(*lv)]
        else:
            raise ValueError(name)
    finally:
        FLAGS[name] = saved
    dev = out[0].device
    gen = torch.Generator(device=dev).manual_seed(4251)
    ups = [torch.randn(o.shape, generator=gen, device=dev, dtype=torch.float32).to(dt0).to(o.dtype) for o in out]
    torch.autograd.backward(out, ups)
    return [o.detach() for o in out], [t.grad for t in lv]


def _qkv_fp32(attn, q, k, v, pe):
    class _N(object):
        def __init__(self, n):
            self.w, self.eps = _w32(n, q.device), float(n.eps)

        def __call__(self, x):
            return F.rms_norm(x, (x.shape[-1],), weight=self.w, eps=self.eps)

    class _QK(object):
        def __init__(self):
            self.qn, self.kn = _N(attn.qknorm.qnorm), _N(attn.qknorm.knorm)

        def __call__(self, a, b):
            return self.qn(a), self.kn(b)

    st = type("A", (), {})()
    st.heads, st.kvheads, st.qknorm = attn.heads, attn.kvheads, _QK()
    return _qkv_eager_local(st, q, k, v, pe)


def _check_call(name, a, r):
    """fp32 reference, eager, fused twice -> updates the piece's result r; True when it verified the fused kernel"""
    with torch.inference_mode(False), torch.enable_grad():
        n0 = STATS.get(name, 0)
        ro, rg = _replay(name, a, "fp32")
        ref = ro + rg
        eo, eg = _replay(name, a, "eager")
        rel_e = [None if (te is None or tr is None) else _rel(te, tr) for te, tr in zip(eo + eg, ref)]
        del eo, eg
        fo, fg = _replay(name, a, "fused")
        fo2, fg2 = _replay(name, a, "fused")
        if STATS.get(name, 0) != n0 + 2:  # the 'fused' replays took the eager branch: that call verifies nothing
            STATS[name] = n0
            return False
        STATS[name] = n0  # (the replays are not training calls)
    r["n"] += 1
    for tf, tf2, re_, tr in zip(fo + fg, fo2 + fg2, rel_e, ref):
        if tf is None and tr is None:
            continue
        if tf is None or tf2 is None or tr is None or re_ is None or not bool(torch.isfinite(tf).all()):
            r["ok"], r["why"] = False, f"{name}: a missing / non-finite tensor"
            continue
        if not torch.equal(tf, tf2):
            r["ok"], r["why"] = False, f"{name} {tuple(tf.shape)}: two fused runs differ (nondeterministic)"
        rf = _rel(tf, tr)
        if rf > r["worst"]:
            r["worst"], r["eager_at_worst"] = rf, re_
        if not rf <= max(GATE_TOL, GATE_X * re_):
            r["ok"], r["why"] = False, f"{name} {tuple(tf.shape)}: rel vs fp32 {rf:.2e} > max({GATE_TOL:g}, {GATE_X:g} x eager {re_:.2e})"
    return True


def gate_begin():
    global _REC
    _REC = _Rec()
    return _REC


def gate_end():
    """-> (per piece {ok, n, worst, eager_at_worst, why}, checked calls, counted-only calls, seconds); the recorder off"""
    global _REC
    rec, _REC = _REC, None
    if rec is None:
        return {}, 0, 0, 0.0
    return rec.res, rec.checked, rec.counted, rec.spent


def gate_verdict(res, parts):
    """the pieces that pass (>= 1 verified call, every check ok) -> kept; the others' FLAGS off"""
    kept = []
    for p in parts:
        rr = res.get(p)
        if rr is not None and rr["ok"] and rr["n"] > 0 and FLAGS.get(p):
            kept.append(p)
        else:
            FLAGS[p] = False
    return kept
