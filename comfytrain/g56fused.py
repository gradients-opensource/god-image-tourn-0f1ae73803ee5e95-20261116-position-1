"""g56fused (T213, 28 Oct 2026): fused flux attention preparation for training - G56CT_FUSE_PREP.

ComfyUI 694815f's flux blocks prepare attention in eager pieces: the q / k / v views of the qkv projection (view + permute),
the two QK RMSNorms on those strided views, the txt + img concatenation (double blocks: three torch.cat), and the training-mode
RoPE (`_apply_rope1`: fp32 copy, mul, addcmul_, cast back) - and autograd mirrors each piece (stack / cat / strided copies /


`attn_prep(segments, pe, heads)` does the same math in ONE Triton kernel forward and ONE backward per segment:
  forward, per (token, head): q, k <- RMSNorm (fp32, rsqrt(mean(x^2) + eps) * w, rounded to the input dtype as F.rms_norm
  returns it), then RoPE in fp32 (out = f[.,0] * y0 + f[.,1] * y1 per pair, as _apply_rope1) rounded to the input dtype;
  v copied; written straight into the concatenated (B, H, L_total, D) q / k / v (contiguous: what SDPA reads best).
  backward: the transposed rotation, the result rounded to the input dtype (eager: the grad of the fp32 copy is cast back),
  the RMSNorm input gradient in fp32 (rstd * g - x * rstd^3 / D * sum(g x), g = dy * w), rounded; dv copied; written
  straight into a contiguous (B, L_seg, 3 H D) gradient of each segment's qkv rows (no stack / cat / strided copies).
The norm weights are frozen (base weights; the LoRA trains Linears only) - no weight gradient. Not bit-identical to eager:
the mean-of-squares and the backward sums reduce in another order (fp32), the FMA contraction may differ (checked against the
eager composition: g56fused.selfcheck).

Only bf16 / fp16 inputs on CUDA, D = 128, the pe of comfy's EmbedND (B or 1, 1, L_total, D / 2, 2, 2) fp32, the inner 3 H D
of every source row contiguous. Anything else: the caller keeps the eager path (attn_prep_ok)."""
import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  (no Triton: the knob stays off)
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _prep_fwd_kernel(SRC, s_b, s_l, WQ, WK, PE, pe_b, pe_l, QO, KO, VO, L_seg, L_tot, l_off, H, eps,
                         D: tl.constexpr, HALF: tl.constexpr):
        pid = tl.program_id(0)
        h = tl.program_id(1)
        b = pid // L_seg
        l = pid % L_seg
        src = SRC + b * s_b + l * s_l + h * D
        offs = tl.arange(0, D)
        q = tl.load(src + offs).to(tl.float32)
        k = tl.load(src + H * D + offs).to(tl.float32)
        v = tl.load(src + 2 * H * D + offs)
        wq = tl.load(WQ + offs).to(tl.float32)
        wk = tl.load(WK + offs).to(tl.float32)
        rq = tl.rsqrt(tl.sum(q * q, axis=0) / D + eps)
        rk = tl.rsqrt(tl.sum(k * k, axis=0) / D + eps)
        yq = (q * rq * wq).to(QO.dtype.element_ty).to(tl.float32)
        yk = (k * rk * wk).to(KO.dtype.element_ty).to(tl.float32)
        lt = l_off + l
        pe = PE + b * pe_b + lt * pe_l
        i = tl.arange(0, HALF)
        f00 = tl.load(pe + i * 4 + 0)
        f01 = tl.load(pe + i * 4 + 1)
        f10 = tl.load(pe + i * 4 + 2)
        f11 = tl.load(pe + i * 4 + 3)
        q0, q1 = tl.split(tl.reshape(yq, (HALF, 2)))
        k0, k1 = tl.split(tl.reshape(yk, (HALF, 2)))
        oq = tl.reshape(tl.join(f00 * q0 + f01 * q1, f10 * q0 + f11 * q1), (D,))
        ok = tl.reshape(tl.join(f00 * k0 + f01 * k1, f10 * k0 + f11 * k1), (D,))
        dst = (b * H + h) * L_tot * D + lt * D
        tl.store(QO + dst + offs, oq.to(QO.dtype.element_ty))
        tl.store(KO + dst + offs, ok.to(KO.dtype.element_ty))
        tl.store(VO + dst + offs, v)

    @triton.jit
    def _prep_bwd_kernel(SRC, s_b, s_l, WQ, WK, PE, pe_b, pe_l, DQO, DKO, DVO, DS, ds_b, ds_l, L_seg, L_tot, l_off, H, eps,
                         D: tl.constexpr, HALF: tl.constexpr):
        pid = tl.program_id(0)
        h = tl.program_id(1)
        b = pid // L_seg
        l = pid % L_seg
        src = SRC + b * s_b + l * s_l + h * D
        offs = tl.arange(0, D)
        q = tl.load(src + offs).to(tl.float32)
        k = tl.load(src + H * D + offs).to(tl.float32)
        wq = tl.load(WQ + offs).to(tl.float32)
        wk = tl.load(WK + offs).to(tl.float32)
        rq = tl.rsqrt(tl.sum(q * q, axis=0) / D + eps)
        rk = tl.rsqrt(tl.sum(k * k, axis=0) / D + eps)
        lt = l_off + l
        pe = PE + b * pe_b + lt * pe_l
        i = tl.arange(0, HALF)
        f00 = tl.load(pe + i * 4 + 0)
        f01 = tl.load(pe + i * 4 + 1)
        f10 = tl.load(pe + i * 4 + 2)
        f11 = tl.load(pe + i * 4 + 3)
        g = (b * H + h) * L_tot * D + lt * D
        dq0, dq1 = tl.split(tl.reshape(tl.load(DQO + g + offs).to(tl.float32), (HALF, 2)))
        dk0, dk1 = tl.split(tl.reshape(tl.load(DKO + g + offs).to(tl.float32), (HALF, 2)))
        dv = tl.load(DVO + g + offs)
        # RoPE transposed; the gradient of the fp32 copy of y is cast back to the input dtype (eager)
        dyq = tl.reshape(tl.join(f00 * dq0 + f10 * dq1, f01 * dq0 + f11 * dq1), (D,)).to(DS.dtype.element_ty).to(tl.float32)
        dyk = tl.reshape(tl.join(f00 * dk0 + f10 * dk1, f01 * dk0 + f11 * dk1), (D,)).to(DS.dtype.element_ty).to(tl.float32)
        gq = dyq * wq
        gk = dyk * wk
        dxq = rq * gq - q * (rq * rq * rq / D) * tl.sum(gq * q, axis=0)
        dxk = rk * gk - k * (rk * rk * rk / D) * tl.sum(gk * k, axis=0)
        ds = DS + b * ds_b + l * ds_l + h * D
        tl.store(ds + offs, dxq.to(DS.dtype.element_ty))
        tl.store(ds + H * D + offs, dxk.to(DS.dtype.element_ty))
        tl.store(ds + 2 * H * D + offs, dv)


if triton is not None:
    @triton.jit
    def _tanh(x):
        e = tl.exp(-2.0 * tl.abs(x))
        t = (1.0 - e) / (1.0 + e)
        return tl.where(x >= 0, t, -t)

    @triton.jit
    def _gelu_fwd_kernel(HH, s_b, s_l, off, W, OUT, L, BLOCK: tl.constexpr):
        l = tl.program_id(0)
        b = tl.program_id(1)
        c0 = tl.program_id(2) * BLOCK
        cols = c0 + tl.arange(0, BLOCK)
        m = cols < W
        x = tl.load(HH + b * s_b + l * s_l + off + cols, mask=m, other=0.0).to(tl.float32)
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)  # nn.GELU(approximate='tanh'): kBeta = sqrt(2 / pi)
        y = 0.5 * x * (1.0 + _tanh(inner))
        tl.store(OUT + (b * L + l) * W + cols, y.to(OUT.dtype.element_ty), mask=m)

    @triton.jit
    def _gelu_bwd_kernel(HH, s_b, s_l, off, W, DG, sg_b, sg_l, DH, sd_b, sd_l, L, BLOCK: tl.constexpr):
        l = tl.program_id(0)
        b = tl.program_id(1)
        c0 = tl.program_id(2) * BLOCK
        cols = c0 + tl.arange(0, BLOCK)
        m = cols < W
        x = tl.load(HH + b * s_b + l * s_l + off + cols, mask=m, other=0.0).to(tl.float32)
        dg = tl.load(DG + b * sg_b + l * sg_l + cols, mask=m, other=0.0).to(tl.float32)
        x2 = x * x
        inner = 0.7978845608028654 * (x + 0.044715 * x2 * x)
        t = _tanh(inner)
        dl = 0.5 * (1.0 + t)  # PyTorch's GeluBackward (tanh): left' + right'
        dr = 0.5 * x * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x2)
        tl.store(DH + b * sd_b + l * sd_l + off + cols, (dg * (dl + dr)).to(DH.dtype.element_ty), mask=m)


def _eps(norm, dtype):
    """the eps ComfyUI's eager norm uses: its own, or for eps None the default of torch 2.9.1's F.rms_norm = the ACCUMULATION type's
    epsilon, float32's 1.19e-07 for a bf16 / fp16 / fp32 input (CUDA _fused_rms_norm_cuda: eps.value_or(numeric_limits<float>::
    epsilon()); CPU rms_norm_composite: the opmath type's epsilon) - NOT finfo(input dtype).eps as torch's docs say. T248 fix (31 Oct
    2026): T213's original used finfo(dtype).eps, 7.8e-03 for bf16 = 65,536 x torch's, in the QK norms with eps None (z-image
    JointAttention q_norm / k_norm, flux QKNorm): rstd = rsqrt(mean(x^2) + eps) off by > 1 % wherever mean(x^2) < 0.4 (t248_check
    eps items: 0.31 rel at mean(x^2) ~ 9e-4, fp16). dtype is unused (kept for the call sites)."""
    e = getattr(norm, "eps", None)
    return float(torch.finfo(torch.float32).eps) if e is None else float(e)


class _AttnPrep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pe, heads, meta, tail_w, *tensors):
        srcs, wqs, wks = tensors[0::3], tensors[1::3], tensors[2::3]
        D = 128
        B = srcs[0].shape[0]
        lens = [s.shape[1] for s in srcs]
        L_tot = sum(lens)
        dt, dev = srcs[0].dtype, srcs[0].device
        qo = torch.empty((B, heads, L_tot, D), dtype=dt, device=dev)
        ko = torch.empty_like(qo)
        vo = torch.empty_like(qo)
        off = 0
        pe_b = pe.stride(0) if pe.shape[0] > 1 else 0
        for s, wq, wk, eps, L in zip(srcs, wqs, wks, meta, lens):
            _prep_fwd_kernel[(B * L, heads)](s, s.stride(0), s.stride(1), wq, wk, pe, pe_b, pe.stride(2), qo, ko, vo,
                                             L, L_tot, off, heads, eps, D=D, HALF=D // 2, num_warps=1)
            off += L
        ctx.heads, ctx.meta, ctx.lens, ctx.tail_w = heads, meta, lens, tail_w
        ctx.save_for_backward(pe, *tensors)
        if not tail_w:
            return qo, ko, vo
        s = srcs[0]  # the single block's tail: GELU(tanh) of linear1's mlp columns, contiguous (B, L, W)
        g = torch.empty((B, lens[0], tail_w), dtype=dt, device=dev)
        _gelu_fwd_kernel[(lens[0], B, triton.cdiv(tail_w, 1024))](s, s.stride(0), s.stride(1), 3 * heads * D, tail_w, g, lens[0],
                                                                 BLOCK=1024, num_warps=4)
        return qo, ko, vo, g

    @staticmethod
    def backward(ctx, dqo, dko, dvo, dg=None):
        pe, *tensors = ctx.saved_tensors
        srcs, wqs, wks = tensors[0::3], tensors[1::3], tensors[2::3]
        D, heads, L_tot = 128, ctx.heads, sum(ctx.lens)
        B = srcs[0].shape[0]
        z = None
        if dqo is None or dko is None or dvo is None:
            z = torch.zeros((B, heads, L_tot, D), dtype=srcs[0].dtype, device=srcs[0].device)
        dqo = (z if dqo is None else dqo).contiguous()
        dko = (z if dko is None else dko).contiguous()
        dvo = (z if dvo is None else dvo).contiguous()
        pe_b = pe.stride(0) if pe.shape[0] > 1 else 0
        grads, off = [], 0
        tw = ctx.tail_w
        for i, (s, wq, wk, eps, L) in enumerate(zip(srcs, wqs, wks, ctx.meta, ctx.lens)):
            ds = None
            if ctx.needs_input_grad[4 + 3 * i]:  # (pe, heads, meta, tail_w, then src / wq / wk per segment)
                ds = torch.empty((B, L, 3 * heads * D + tw), dtype=s.dtype, device=s.device)
                _prep_bwd_kernel[(B * L, heads)](s, s.stride(0), s.stride(1), wq, wk, pe, pe_b, pe.stride(2), dqo, dko, dvo, ds,
                                                 ds.stride(0), ds.stride(1), L, L_tot, off, heads, eps, D=D, HALF=D // 2,
                                                 num_warps=1)
                if tw:  # the tail's gradient in the same buffer: no split-backward cat of the qkv and mlp gradients
                    if dg is None:
                        ds[..., 3 * heads * D:].zero_()
                    else:
                        _gelu_bwd_kernel[(L, B, triton.cdiv(tw, 1024))](s, s.stride(0), s.stride(1), 3 * heads * D, tw, dg,
                                                                        dg.stride(0), dg.stride(1), ds, ds.stride(0), ds.stride(1),
                                                                        L, BLOCK=1024, num_warps=4)
            off += L
            grads += [ds, None, None]
        return (None, None, None, None, *grads)


def attn_prep_ok(segments, pe, heads):
    """None when attn_prep can run these (src, QKNorm) segments with this pe, else why not (the caller stays eager)."""
    if triton is None:
        return "no triton"
    if not segments:
        return "no segment"
    s0 = segments[0][0]
    if not s0.is_cuda or s0.dtype not in (torch.bfloat16, torch.float16):
        return "not a CUDA bf16 / fp16 input"
    L_tot = 0
    for s, norm in segments:
        if s.dim() != 3 or s.dtype != s0.dtype or s.device != s0.device or s.shape[0] != s0.shape[0]:
            return "segments differ"
        if s.shape[2] < 3 * heads * 128 or (s.shape[2] != 3 * heads * 128 and len(segments) != 1) or s.stride(2) != 1:
            return "qkv rows not (3 H 128 [+ the single block's tail]) contiguous"
        for n in (norm.query_norm, norm.key_norm):
            w = getattr(n, "weight", None)
            if w is None or tuple(w.shape) != (128,) or w.dtype != s0.dtype or w.device != s0.device or w.requires_grad:
                return "QKNorm weight not a frozen (128,) tensor of the input dtype"
            if getattr(n, "comfy_cast_weights", False) or len(getattr(n, "weight_function", [])) or len(getattr(n, "bias_function", [])):
                return "QKNorm with cast / weight functions"
            if tuple(getattr(n, "normalized_shape", (128,))) != (128,):
                return "QKNorm shape"
        L_tot += s.shape[1]
    if (pe is None or pe.dtype != torch.float32 or pe.dim() != 6 or tuple(pe.shape[1:]) != (1, L_tot, 64, 2, 2)
            or pe.shape[0] not in (1, s0.shape[0]) or not pe.is_cuda or pe.stride(-1) != 1 or pe.stride(-2) != 2 or pe.stride(-3) != 4):
        return "pe not comfy's EmbedND layout for these tokens"
    return None


def attn_prep(segments, pe, heads):
    """q, k, v (B, H, L_total, 128), contiguous: the segments' [txt, img] tokens concatenated, QK-normed and RoPE'd."""
    tensors, meta = [], []
    for s, norm in segments:
        tensors += [s, norm.query_norm.weight, norm.key_norm.weight]
        e1, e2 = _eps(norm.query_norm, s.dtype), _eps(norm.key_norm, s.dtype)
        if e1 != e2:
            raise ValueError("query / key norm eps differ")
        meta.append(e1)
    return _AttnPrep.apply(pe, heads, tuple(meta), 0, *tensors)


def attn_prep_single(h, norm, pe, heads, mlp_w):
    """The single block's linear1 output h (B, L, 3 H 128 + mlp_w) -> q, k, v (as attn_prep) and GELU-tanh of the mlp columns
    (B, L, mlp_w) contiguous; the backward writes the whole row gradient of h at once (no split-backward cat)."""
    e1, e2 = _eps(norm.query_norm, h.dtype), _eps(norm.key_norm, h.dtype)
    if e1 != e2:
        raise ValueError("query / key norm eps differ")
    return _AttnPrep.apply(pe, heads, (e1,), int(mlp_w), h, norm.query_norm.weight, norm.key_norm.weight)


def eager_reference(segments, pe, heads):
    """The ComfyUI 694815f composition attn_prep replaces (training mode): views, QKNorm, cat, _apply_rope1."""
    from comfy.ldm.flux.math import _apply_rope1
    qs, ks, vs = [], [], []
    for s, norm in segments:
        B, L = s.shape[0], s.shape[1]
        q, k, v = s.view(B, L, 3, heads, -1).permute(2, 0, 3, 1, 4)
        q, k = norm(q, k, v)
        qs.append(q)
        ks.append(k)
        vs.append(v)
    q = torch.cat(qs, dim=2) if len(qs) > 1 else qs[0]
    k = torch.cat(ks, dim=2) if len(ks) > 1 else ks[0]
    v = torch.cat(vs, dim=2) if len(vs) > 1 else vs[0]
    return _apply_rope1(q, pe), _apply_rope1(k, pe), v


def selfcheck(device, heads=24, lens=(512, 3008), dtype=torch.bfloat16, seed=0, tol=None):
    """fwd + bwd of attn_prep vs eager_reference on random inputs (two segments = a double block, one = a single block): the
    largest rel L2 error of q / k / v and of every source gradient; ok when all below tol (default 2e-2: bf16 rounding of
    the RMSNorm / RoPE intermediates in another order). Returns (ok, report dict)."""
    import comfy.ldm.flux.layers as L
    import comfy.ops
    tol = 2e-2 if tol is None else tol
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    ok = True
    for name, segl in (("double", lens), ("single", (sum(lens),))):
        segs, norms = [], []
        for Ls in segl:
            n = L.QKNorm(128, dtype=dtype, device=device, operations=comfy.ops.disable_weight_init)
            with torch.no_grad():
                n.query_norm.weight.copy_(1.0 + 0.1 * torch.randn(128, generator=g))
                n.key_norm.weight.copy_(1.0 + 0.1 * torch.randn(128, generator=g))
            n.requires_grad_(False)
            src = (torch.randn(1, Ls, 3 * heads * 128 + (64 if name == "single" else 0), generator=g) * 2.0).to(device, dtype)
            src = src[..., : 3 * heads * 128]  # single block: the qkv part of linear1's output (row stride > 3 H D)
            segs.append(src)
            norms.append(n)
        Lt = sum(segl)
        ids = torch.randn(1, Lt, 3, generator=g) * 40
        from comfy.ldm.flux.layers import EmbedND
        pe = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])(ids.to(device))
        go = [torch.randn(1, heads, Lt, 128, generator=g).to(device, dtype) for _ in range(3)]
        res = {}
        for tag, fn in (("fused", attn_prep), ("eager", eager_reference)):
            xs = [s.detach().clone().requires_grad_(True) for s in segs]
            why = attn_prep_ok(list(zip(xs, norms)), pe, heads) if tag == "fused" else None
            if why is not None:
                return False, {"why": why}
            o = fn(list(zip(xs, norms)), pe, heads)
            torch.autograd.backward(o, go)
            res[tag] = ([t.detach().float() for t in o], [x.grad.float() for x in xs])
        rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-30))  # noqa: E731
        r = {"q": rel(res["fused"][0][0], res["eager"][0][0]), "k": rel(res["fused"][0][1], res["eager"][0][1]),
             "v": rel(res["fused"][0][2], res["eager"][0][2]),
             "dsrc": max(rel(a, b) for a, b in zip(res["fused"][1], res["eager"][1]))}
        out[name] = r
        ok = ok and all(v_ <= tol for v_ in r.values())
    # the single block with its GELU tail: linear1's whole row in, q / k / v + GELU(mlp) out, the row gradient at once
    W = 96
    n = L.QKNorm(128, dtype=dtype, device=device, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        n.query_norm.weight.copy_(1.0 + 0.1 * torch.randn(128, generator=g))
        n.key_norm.weight.copy_(1.0 + 0.1 * torch.randn(128, generator=g))
    n.requires_grad_(False)
    Lt = sum(lens)
    hsrc = (torch.randn(1, Lt, 3 * heads * 128 + W, generator=g) * 2.0).to(device, dtype)
    from comfy.ldm.flux.layers import EmbedND
    pe = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])(torch.randn(1, Lt, 3, generator=g).to(device) * 40)
    go = [torch.randn(1, heads, Lt, 128, generator=g).to(device, dtype) for _ in range(3)] + [torch.randn(1, Lt, W, generator=g).to(device, dtype)]
    res = {}
    for tag in ("fused", "eager"):
        x = hsrc.detach().clone().requires_grad_(True)
        if tag == "fused":
            if attn_prep_ok([(x, n)], pe, heads) is not None:
                return False, {"why_tail": attn_prep_ok([(x, n)], pe, heads)}
            o = attn_prep_single(x, n, pe, heads, W)
        else:
            qkv, mlp = torch.split(x, [3 * heads * 128, W], dim=-1)
            o = list(eager_reference([(qkv, n)], pe, heads)) + [torch.nn.functional.gelu(mlp, approximate="tanh")]
        torch.autograd.backward(list(o), go)
        res[tag] = ([t.detach().float() for t in o], x.grad.float())
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-30))  # noqa: E731
    r = {"q": rel(res["fused"][0][0], res["eager"][0][0]), "k": rel(res["fused"][0][1], res["eager"][0][1]),
         "v": rel(res["fused"][0][2], res["eager"][0][2]), "gelu": rel(res["fused"][0][3], res["eager"][0][3]),
         "dh": rel(res["fused"][1], res["eager"][1])}
    out["single_tail"] = r
    ok = ok and all(v_ <= tol for v_ in r.values())
    return ok, out


# ----------------------------------------------------------------------------------------------------------------------
# flux block forwards with the fused preparation (ComfyUI 694815f's DoubleStreamBlock / SingleStreamBlock.forward with the
# trainer's out-of-place residual adds, attention through attn_prep + optimized_attention); any call the fused path cannot
# take (masks, modulation_dims, patches, another dtype / head size / pe layout) runs the previous forward unchanged.
# ----------------------------------------------------------------------------------------------------------------------
_FALLBACK = {}
STATS = {"double": 0, "single": 0, "fallback": 0, "why": {}}
FLAGS = {"prep": False, "mod": False}  # which fused pieces the installed forwards use (install_flux)


def _lnm(norm, x, shift, scale, dims):
    if FLAGS["mod"] and dims is None:
        return ln_mod(norm, x, shift, scale)
    from comfy.ldm.flux.layers import apply_mod
    return apply_mod(norm(x), (1 + scale), shift, dims)


def _gr(x, y, gate, dims):
    if FLAGS["mod"] and dims is None:
        return gate_res(x, y, gate)
    from comfy.ldm.flux.layers import apply_mod
    return x + apply_mod(y, gate, None, dims)


def _note_fallback(why):
    STATS["fallback"] += 1
    STATS["why"][why] = STATS["why"].get(why, 0) + 1


def _double_forward(self, img, txt, vec, pe, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None,
                    transformer_options={}):  # noqa: B006  (ComfyUI's signature)
    from comfy.ldm.flux.layers import apply_mod
    from comfy.ldm.modules.attention import optimized_attention
    why = None
    if attn_mask is not None or modulation_dims_img is not None or modulation_dims_txt is not None:
        why = "mask / modulation_dims"
    elif transformer_options.get("patches", {}):
        why = "transformer patches"
    if why is not None:
        _note_fallback(why)
        return _FALLBACK["double"](self, img, txt, vec, pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims_img,
                                   modulation_dims_txt=modulation_dims_txt, transformer_options=transformer_options)
    if self.modulation:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
    else:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec
    img_qkv = self.img_attn.qkv(_lnm(self.img_norm1, img, img_mod1.shift, img_mod1.scale, modulation_dims_img))
    txt_qkv = self.txt_attn.qkv(_lnm(self.txt_norm1, txt, txt_mod1.shift, txt_mod1.scale, modulation_dims_txt))
    segs = [(txt_qkv, self.txt_attn.norm), (img_qkv, self.img_attn.norm)]
    why = attn_prep_ok(segs, pe, self.num_heads) if FLAGS["prep"] else "FUSE_PREP off"
    if why is not None:  # the eager preparation from here on, exactly as ComfyUI's forward
        if FLAGS["prep"]:
            _note_fallback(why)
        from comfy.ldm.flux.math import attention
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del img_qkv
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del txt_qkv
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
    else:
        STATS["double"] += 1
        q, k, v = attn_prep(segs, pe, self.num_heads)
        del img_qkv, txt_qkv
        attn = optimized_attention(q, k, v, q.shape[1], skip_reshape=True, mask=attn_mask, transformer_options=transformer_options)
    del q, k, v
    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]
    img = _gr(img, self.img_attn.proj(img_attn), img_mod1.gate, modulation_dims_img)
    del img_attn
    img = _gr(img, self.img_mlp(_lnm(self.img_norm2, img, img_mod2.shift, img_mod2.scale, modulation_dims_img)), img_mod2.gate,
              modulation_dims_img)
    txt = _gr(txt, self.txt_attn.proj(txt_attn), txt_mod1.gate, modulation_dims_txt)
    del txt_attn
    txt = _gr(txt, self.txt_mlp(_lnm(self.txt_norm2, txt, txt_mod2.shift, txt_mod2.scale, modulation_dims_txt)), txt_mod2.gate,
              modulation_dims_txt)
    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)
    return img, txt


def _single_forward(self, x, vec, pe, attn_mask=None, modulation_dims=None, transformer_options={}):  # noqa: B006
    from comfy.ldm.flux.layers import apply_mod
    from comfy.ldm.modules.attention import optimized_attention
    why = None
    if attn_mask is not None or modulation_dims is not None:
        why = "mask / modulation_dims"
    elif transformer_options.get("patches", {}):
        why = "transformer patches"
    if why is not None:
        _note_fallback(why)
        return _FALLBACK["single"](self, x, vec, pe, attn_mask=attn_mask, modulation_dims=modulation_dims,
                                   transformer_options=transformer_options)
    if self.modulation:
        mod, _ = self.modulation(vec)
    else:
        mod = vec
    h = self.linear1(_lnm(self.pre_norm, x, mod.shift, mod.scale, modulation_dims))
    gelu_tail = (FLAGS["prep"] and not self.yak_mlp and isinstance(self.mlp_act, torch.nn.GELU)
                 and getattr(self.mlp_act, "approximate", "none") == "tanh" and h.shape[2] == 3 * self.hidden_size + self.mlp_hidden_dim_first)
    if gelu_tail and attn_prep_ok([(h, self.norm)], pe, self.num_heads) is None:
        STATS["single"] += 1
        STATS["single_tail"] = STATS.get("single_tail", 0) + 1
        q, k, v, mlp = attn_prep_single(h, self.norm, pe, self.num_heads, self.mlp_hidden_dim_first)
        del h
        attn = optimized_attention(q, k, v, q.shape[1], skip_reshape=True, mask=attn_mask, transformer_options=transformer_options)
        del q, k, v
        output = self.linear2(torch.cat((attn, mlp), 2))
        x = _gr(x, output, mod.gate, modulation_dims)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x
    qkv, mlp = torch.split(h, [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)
    segs = [(qkv, self.norm)]
    why = attn_prep_ok(segs, pe, self.num_heads) if FLAGS["prep"] else "FUSE_PREP off"
    if why is not None:
        if FLAGS["prep"]:
            _note_fallback(why)
        from comfy.ldm.flux.math import attention
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del qkv
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
    else:
        STATS["single"] += 1
        q, k, v = attn_prep(segs, pe, self.num_heads)
        del qkv
        attn = optimized_attention(q, k, v, q.shape[1], skip_reshape=True, mask=attn_mask, transformer_options=transformer_options)
    del q, k, v
    if self.yak_mlp:
        mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
    else:
        mlp = self.mlp_act(mlp)
    output = self.linear2(torch.cat((attn, mlp), 2))
    x = _gr(x, output, mod.gate, modulation_dims)
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def install_flux(prep=True, mod=False):
    """Patch comfy.ldm.flux.layers.DoubleStreamBlock / SingleStreamBlock.forward with the fused forwards (the forwards in place
    before = the fallback): prep = attn_prep, mod = ln_mod / gate_res. Idempotent. Returns the number of classes patched."""
    import comfy.ldm.flux.layers as L
    FLAGS["prep"], FLAGS["mod"] = bool(prep), bool(mod)
    n = 0
    for key, cls, fn in (("double", L.DoubleStreamBlock, _double_forward), ("single", L.SingleStreamBlock, _single_forward)):
        if getattr(cls.forward, "_g56_fused", False):
            continue
        _FALLBACK[key] = cls.forward
        fn._g56_fused = True
        cls.forward = fn
        n += 1
    return n


def uninstall_flux():
    import comfy.ldm.flux.layers as L
    for key, cls in (("double", L.DoubleStreamBlock), ("single", L.SingleStreamBlock)):
        if key in _FALLBACK and getattr(cls.forward, "_g56_fused", False):
            cls.forward = _FALLBACK.pop(key)


# ----------------------------------------------------------------------------------------------------------------------
# Modulation (T213, 28 Oct): flux's apply_mod(LayerNorm(x), 1 + scale, shift) and x + apply_mod(y, gate) as one kernel forward
# and one backward each. Eager (ComfyUI 694815f + the trainer's out-of-place residuals): native_layer_norm (no affine, the
# output rounded to the input dtype), 1 + scale (rounded), addcmul(shift, ln, 1 + scale) in fp32 -> rounded; y * gate
# (rounded) then x + it (rounded). The backward's column sums over the tokens (d shift, d scale, d gate: eager's broadcast
# reductions) are accumulated in fp32 per block of rows and summed over the blocks (another order than eager's reduce_kernel).
# ----------------------------------------------------------------------------------------------------------------------
if triton is not None:
    @triton.jit
    def _lnmod_fwd_kernel(X, s_xb, s_xl, SH, SC, s_mb, OUT, MEAN, RSTD, L, C, eps, BLOCK: tl.constexpr):
        l = tl.program_id(0)
        b = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        x = tl.load(X + b * s_xb + l * s_xl + cols, mask=m, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        xc = tl.where(m, x - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(xc * xc, axis=0) / C + eps)
        ln = (xc * rstd).to(OUT.dtype.element_ty).to(tl.float32)
        sc = tl.load(SC + b * s_mb + cols, mask=m, other=0.0).to(tl.float32)
        opsc = (1.0 + sc).to(OUT.dtype.element_ty).to(tl.float32)
        sh = tl.load(SH + b * s_mb + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + (b * L + l) * C + cols, (sh + ln * opsc).to(OUT.dtype.element_ty), mask=m)
        tl.store(MEAN + b * L + l, mean)
        tl.store(RSTD + b * L + l, rstd)

    @triton.jit
    def _lnmod_bwd_kernel(X, s_xb, s_xl, SC, s_mb, DO, MEAN, RSTD, DX, PSH, PSC, L, C, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        pblk = tl.program_id(0)
        b = tl.program_id(1)
        nblk = tl.num_programs(0)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        sc = tl.load(SC + b * s_mb + cols, mask=m, other=0.0).to(tl.float32)
        opsc = (1.0 + sc).to(DX.dtype.element_ty).to(tl.float32)
        acc_sh = tl.zeros((BLOCK,), dtype=tl.float32)
        acc_sc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in range(ROWS):
            l = pblk * ROWS + r
            if l < L:
                x = tl.load(X + b * s_xb + l * s_xl + cols, mask=m, other=0.0).to(tl.float32)
                mean = tl.load(MEAN + b * L + l)
                rstd = tl.load(RSTD + b * L + l)
                xhat = tl.where(m, (x - mean) * rstd, 0.0)
                ln = xhat.to(DX.dtype.element_ty).to(tl.float32)
                d = tl.load(DO + (b * L + l) * C + cols, mask=m, other=0.0).to(tl.float32)
                acc_sh += d
                acc_sc += (d * ln).to(DX.dtype.element_ty).to(tl.float32)  # eager: the bf16 product, then the fp32 sum
                dln = tl.where(m, (d * opsc).to(DX.dtype.element_ty).to(tl.float32), 0.0)
                dx = rstd * (dln - tl.sum(dln, axis=0) / C - xhat * (tl.sum(dln * xhat, axis=0) / C))
                tl.store(DX + (b * L + l) * C + cols, dx.to(DX.dtype.element_ty), mask=m)
        tl.store(PSH + (b * nblk + pblk) * C + cols, acc_sh, mask=m)
        tl.store(PSC + (b * nblk + pblk) * C + cols, acc_sc, mask=m)

    @triton.jit
    def _gres_fwd_kernel(X, s_xb, s_xl, Y, s_yb, s_yl, G, s_gb, OUT, L, C, BLOCK: tl.constexpr, SINGLE: tl.constexpr = False):
        l = tl.program_id(0)
        b = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        x = tl.load(X + b * s_xb + l * s_xl + cols, mask=m, other=0.0)
        y = tl.load(Y + b * s_yb + l * s_yl + cols, mask=m, other=0.0)
        g = tl.load(G + b * s_gb + cols, mask=m, other=0.0)
        if SINGLE:  # torch.addcmul(x, gate, y): x + gate * y in fp32 opmath, rounded once (qwen-image)
            tl.store(OUT + (b * L + l) * C + cols, (x.to(tl.float32) + y.to(tl.float32) * g.to(tl.float32)).to(OUT.dtype.element_ty), mask=m)
        else:
            t = (y.to(tl.float32) * g.to(tl.float32)).to(OUT.dtype.element_ty)  # eager: y * gate rounded, then the add rounded
            tl.store(OUT + (b * L + l) * C + cols, (x.to(tl.float32) + t.to(tl.float32)).to(OUT.dtype.element_ty), mask=m)

    @triton.jit
    def _gres_bwd_kernel(DO, Y, s_yb, s_yl, G, s_gb, DY, PG, L, C, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        pblk = tl.program_id(0)
        b = tl.program_id(1)
        nblk = tl.num_programs(0)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        g = tl.load(G + b * s_gb + cols, mask=m, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in range(ROWS):
            l = pblk * ROWS + r
            if l < L:
                d = tl.load(DO + (b * L + l) * C + cols, mask=m, other=0.0).to(tl.float32)
                y = tl.load(Y + b * s_yb + l * s_yl + cols, mask=m, other=0.0).to(tl.float32)
                acc += (d * y).to(DY.dtype.element_ty).to(tl.float32)
                tl.store(DY + (b * L + l) * C + cols, (d * g).to(DY.dtype.element_ty), mask=m)
        tl.store(PG + (b * nblk + pblk) * C + cols, acc, mask=m)

_ROWS = 16


def _blk(C):
    return triton.next_power_of_2(C)


class _LNMod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale, eps):
        B, L, C = x.shape
        out = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
        mean = torch.empty((B, L), dtype=torch.float32, device=x.device)
        rstd = torch.empty_like(mean)
        _lnmod_fwd_kernel[(L, B)](x, x.stride(0), x.stride(1), shift, scale, shift.stride(0), out, mean, rstd, L, C, eps,
                                  BLOCK=_blk(C), num_warps=8)
        ctx.save_for_backward(x, scale, mean, rstd)
        ctx.mb = shift.stride(0)
        return out

    @staticmethod
    def backward(ctx, do):
        x, scale, mean, rstd = ctx.saved_tensors
        B, L, C = x.shape
        do = do.contiguous()
        nblk = triton.cdiv(L, _ROWS)
        dx = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
        psh = torch.empty((B, nblk, C), dtype=torch.float32, device=x.device)
        psc = torch.empty_like(psh)
        _lnmod_bwd_kernel[(nblk, B)](x, x.stride(0), x.stride(1), scale, scale.stride(0), do, mean, rstd, dx, psh, psc, L, C,
                                     ROWS=_ROWS, BLOCK=_blk(C), num_warps=8)
        dsh = psh.sum(dim=1, keepdim=True).to(x.dtype)
        dsc = psc.sum(dim=1, keepdim=True).to(x.dtype)
        return dx, (dsh if ctx.needs_input_grad[1] else None), (dsc if ctx.needs_input_grad[2] else None), None


class _GateRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gate, single=False):
        B, L, C = x.shape
        out = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
        _gres_fwd_kernel[(L, B)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), gate, gate.stride(0), out, L, C,
                                 BLOCK=_blk(C), SINGLE=bool(single), num_warps=8)
        ctx.save_for_backward(y, gate)
        return out

    @staticmethod
    def backward(ctx, do):
        y, gate = ctx.saved_tensors
        B, L, C = y.shape
        do = do.contiguous()
        nblk = triton.cdiv(L, _ROWS)
        dy = torch.empty((B, L, C), dtype=y.dtype, device=y.device)
        pg = torch.empty((B, nblk, C), dtype=torch.float32, device=y.device)
        _gres_bwd_kernel[(nblk, B)](do, y, y.stride(0), y.stride(1), gate, gate.stride(0), dy, pg, L, C, ROWS=_ROWS,
                                    BLOCK=_blk(C), num_warps=8)
        dg = pg.sum(dim=1, keepdim=True).to(y.dtype) if ctx.needs_input_grad[2] else None
        return do, dy, dg, None


def _mod_ok(x, *vs):
    """None when the fused modulation can take x (B, L, C) and the (B, 1, C) modulation vectors vs, else why not."""
    if triton is None:
        return "no triton"
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16) or x.dim() != 3 or x.stride(2) != 1:
        return "x"
    for v in vs:
        if (v is None or v.dtype != x.dtype or v.device != x.device or v.dim() != 3 or v.shape[0] != x.shape[0] or v.shape[1] != 1
                or v.shape[2] != x.shape[2] or v.stride(2) != 1):
            return "modulation vector"
    return None


def ln_mod(norm, x, shift, scale):
    """apply_mod(norm(x), 1 + scale, shift) for flux's LayerNorm(no affine): fused when possible, else ComfyUI's own ops."""
    if (_mod_ok(x, shift, scale) is None and getattr(norm, "weight", None) is None and getattr(norm, "bias", None) is None
            and tuple(norm.normalized_shape) == (x.shape[2],)):
        STATS["lnmod"] = STATS.get("lnmod", 0) + 1
        return _LNMod.apply(x, shift, scale, float(norm.eps))
    from comfy.ldm.flux.layers import apply_mod
    STATS["lnmod_eager"] = STATS.get("lnmod_eager", 0) + 1
    return apply_mod(norm(x), (1 + scale), shift, None)


def gate_res(x, y, gate):
    """x + apply_mod(y, gate) (the trainer's out-of-place residual): fused when possible."""
    if _mod_ok(x, gate) is None and _mod_ok(y) is None and y.shape == x.shape:
        STATS["gres"] = STATS.get("gres", 0) + 1
        return _GateRes.apply(x, y, gate)
    from comfy.ldm.flux.layers import apply_mod
    STATS["gres_eager"] = STATS.get("gres_eager", 0) + 1
    return x + apply_mod(y, gate, None, None)


def mod_selfcheck(device, L=1000, C=3072, dtype=torch.bfloat16, seed=0, tol=2e-2):
    """ln_mod / gate_res vs ComfyUI's eager ops (fwd + every input gradient): rel L2 each; (ok, report)."""
    import comfy.ops
    from comfy.ldm.flux.layers import apply_mod
    g = torch.Generator(device="cpu").manual_seed(seed)
    norm = comfy.ops.disable_weight_init.LayerNorm(C, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
    x = (torch.randn(1, L, C, generator=g) * 3 + 0.5).to(device, dtype)
    mod = (torch.randn(1, 1, 3 * C, generator=g) * 0.5).to(device, dtype)
    y = torch.randn(1, L, C, generator=g).to(device, dtype)
    go = torch.randn(1, L, C, generator=g).to(device, dtype)
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))  # noqa: E731
    rep = {}
    for tag in ("fused", "eager"):
        xs, ms, ys = x.clone().requires_grad_(True), mod.clone().requires_grad_(True), y.clone().requires_grad_(True)
        sh, sc, gt = ms.chunk(3, dim=-1)
        if tag == "fused":
            if _mod_ok(xs, sh, sc, gt) is not None:
                return False, {"why": _mod_ok(xs, sh, sc, gt)}
            a = _LNMod.apply(xs, sh, sc, 1e-6)
            o = _GateRes.apply(xs, ys, gt)
        else:
            a = apply_mod(norm(xs), (1 + sc), sh, None)
            o = xs + apply_mod(ys, gt, None, None)
        torch.autograd.backward([a, o], [go, go])
        rep[tag] = (a.detach(), o.detach(), xs.grad, ms.grad, ys.grad)
    names = ("lnmod", "gres", "dx", "dmod", "dy")
    r = {n: rel(f, e) for n, f, e in zip(names, rep["fused"], rep["eager"])}
    return all(v <= tol for v in r.values()), r


# ----------------------------------------------------------------------------------------------------------------------
# Generic attention preparation (T213, 28 Oct 2026) for the other DiTs: gprep(segments, rope, heads, mode) does what attn_prep does
# for flux, for q / k / v given either as one fused projection output (columns [q | k | v], each H x D wide: ideogram4's qkv view
# (B, L, 3, H, D), z-image NextDiT's qkv split [H D, H D, H D] (n_kv_heads == n_heads)) or as three separate projections (qwen-image's
# to_q / to_k / to_v and add_q_proj / ...), each segment with its own two RMSNorms (frozen weights, own eps), and one of three RoPEs:
#   'pairs32' comfy.ldm.flux.math._apply_rope1 with an fp32 EmbedND pe (z-image: (1, L, 1, D/2, 2, 2) after movedim + cat):
#             out[2i + j] = f[i][j][0] y[2i] + f[i][j][1] y[2i + 1] in fp32, rounded to the input dtype
#   'pairs16' the same with the pe already in the input dtype (qwen-image: EmbedND(ids).to(x.dtype)): f[.,0] y0 rounded to the
#             input dtype first (bf16 mul), then the addcmul_ in fp32 opmath, rounded
#   'half'    comfy.text_encoders.llama.apply_rope (ideogram4: cos (1, L, D), sin[..., :D/2] (1, L, D/2), nsin (1, L, D/2), fp32):
#             out[:D/2] = y[:D/2] cos + y[D/2:] nsin, out[D/2:] = y[D/2:] cos + y[:D/2] sin in fp32, rounded
# The RMSNorm output is rounded to the input dtype before the RoPE (F.rms_norm returns the input dtype). Output: q / k / v
# (B, H, L_total, D) contiguous, the segments' tokens concatenated in the given order. Backward: the transposed rotation (fp32,
# rounded to the input dtype as the eager grad of the input-dtype norm output), the RMSNorm input gradient in fp32, rounded; a fused
# segment's gradient is ONE (B, L, 3 H D) buffer written in place (no split / unbind / view backward), a separate segment's three.
# Not bit-identical to eager (reduction order, FMA contraction); checked against the eager composition (gprep_selfcheck).
# ----------------------------------------------------------------------------------------------------------------------
GMODES = {"pairs32": 0, "pairs16": 1, "half": 2}

if triton is not None:
    @triton.jit
    def _grope_fwd(y, A, a_off, B_, b_off, C_, c_off, MODE: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, ODT: tl.constexpr):
        i = tl.arange(0, HALF)
        if MODE == 2:
            ya, yb = tl.split(tl.permute(tl.reshape(y, (2, HALF)), (1, 0)))
            ca = tl.load(A + a_off + i).to(tl.float32)
            cb = tl.load(A + a_off + HALF + i).to(tl.float32)
            s = tl.load(B_ + b_off + i).to(tl.float32)
            n = tl.load(C_ + c_off + i).to(tl.float32)
            return tl.reshape(tl.permute(tl.join(ya * ca + yb * n, yb * cb + ya * s), (1, 0)), (D,))
        else:
            f00 = tl.load(A + a_off + i * 4 + 0).to(tl.float32)
            f01 = tl.load(A + a_off + i * 4 + 1).to(tl.float32)
            f10 = tl.load(A + a_off + i * 4 + 2).to(tl.float32)
            f11 = tl.load(A + a_off + i * 4 + 3).to(tl.float32)
            y0, y1 = tl.split(tl.reshape(y, (HALF, 2)))
            if MODE == 1:  # pe in the input dtype: the first product rounded to it (bf16 mul), then the addcmul_ (fp32 opmath)
                t0 = (f00 * y0).to(ODT).to(tl.float32)
                t1 = (f10 * y0).to(ODT).to(tl.float32)
                return tl.reshape(tl.join(t0 + f01 * y1, t1 + f11 * y1), (D,))
            return tl.reshape(tl.join(f00 * y0 + f01 * y1, f10 * y0 + f11 * y1), (D,))

    @triton.jit
    def _grope_bwd(g, A, a_off, B_, b_off, C_, c_off, MODE: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr):
        i = tl.arange(0, HALF)
        if MODE == 2:
            ga, gb = tl.split(tl.permute(tl.reshape(g, (2, HALF)), (1, 0)))
            ca = tl.load(A + a_off + i).to(tl.float32)
            cb = tl.load(A + a_off + HALF + i).to(tl.float32)
            s = tl.load(B_ + b_off + i).to(tl.float32)
            n = tl.load(C_ + c_off + i).to(tl.float32)
            return tl.reshape(tl.permute(tl.join(ga * ca + gb * s, gb * cb + ga * n), (1, 0)), (D,))
        else:
            f00 = tl.load(A + a_off + i * 4 + 0).to(tl.float32)
            f01 = tl.load(A + a_off + i * 4 + 1).to(tl.float32)
            f10 = tl.load(A + a_off + i * 4 + 2).to(tl.float32)
            f11 = tl.load(A + a_off + i * 4 + 3).to(tl.float32)
            g0, g1 = tl.split(tl.reshape(g, (HALF, 2)))
            return tl.reshape(tl.join(f00 * g0 + f10 * g1, f01 * g0 + f11 * g1), (D,))

    @triton.jit
    def _gprep_fwd_kernel(QS, s_qb, s_ql, KS, s_kb, s_kl, VS, s_vb, s_vl, WQ, WK, RA, ra_b, ra_l, RB, rb_b, rb_l, RC, rc_b, rc_l,
                          QO, KO, VO, L_seg, L_tot, l_off, H, eps_q, eps_k, D: tl.constexpr, HALF: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        h = tl.program_id(1)
        b = pid // L_seg
        l = pid % L_seg
        offs = tl.arange(0, D)
        q = tl.load(QS + b * s_qb + l * s_ql + h * D + offs).to(tl.float32)
        k = tl.load(KS + b * s_kb + l * s_kl + h * D + offs).to(tl.float32)
        v = tl.load(VS + b * s_vb + l * s_vl + h * D + offs)
        wq = tl.load(WQ + offs).to(tl.float32)
        wk = tl.load(WK + offs).to(tl.float32)
        yq = (q * tl.rsqrt(tl.sum(q * q, axis=0) / D + eps_q) * wq).to(QO.dtype.element_ty).to(tl.float32)
        yk = (k * tl.rsqrt(tl.sum(k * k, axis=0) / D + eps_k) * wk).to(KO.dtype.element_ty).to(tl.float32)
        lt = l_off + l
        oq = _grope_fwd(yq, RA, b * ra_b + lt * ra_l, RB, b * rb_b + lt * rb_l, RC, b * rc_b + lt * rc_l, MODE, D, HALF,
                        QO.dtype.element_ty)
        ok = _grope_fwd(yk, RA, b * ra_b + lt * ra_l, RB, b * rb_b + lt * rb_l, RC, b * rc_b + lt * rc_l, MODE, D, HALF,
                        KO.dtype.element_ty)
        dst = (b * H + h) * L_tot * D + lt * D
        tl.store(QO + dst + offs, oq.to(QO.dtype.element_ty))
        tl.store(KO + dst + offs, ok.to(KO.dtype.element_ty))
        tl.store(VO + dst + offs, v)

    @triton.jit
    def _gprep_bwd_kernel(QS, s_qb, s_ql, KS, s_kb, s_kl, WQ, WK, RA, ra_b, ra_l, RB, rb_b, rb_l, RC, rc_b, rc_l,
                          DQO, DKO, DVO, DQ, sdq_b, sdq_l, DK, sdk_b, sdk_l, DV, sdv_b, sdv_l, L_seg, L_tot, l_off, H, eps_q, eps_k,
                          D: tl.constexpr, HALF: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        h = tl.program_id(1)
        b = pid // L_seg
        l = pid % L_seg
        offs = tl.arange(0, D)
        q = tl.load(QS + b * s_qb + l * s_ql + h * D + offs).to(tl.float32)
        k = tl.load(KS + b * s_kb + l * s_kl + h * D + offs).to(tl.float32)
        wq = tl.load(WQ + offs).to(tl.float32)
        wk = tl.load(WK + offs).to(tl.float32)
        rq = tl.rsqrt(tl.sum(q * q, axis=0) / D + eps_q)
        rk = tl.rsqrt(tl.sum(k * k, axis=0) / D + eps_k)
        lt = l_off + l
        g = (b * H + h) * L_tot * D + lt * D
        dyq = _grope_bwd(tl.load(DQO + g + offs).to(tl.float32), RA, b * ra_b + lt * ra_l, RB, b * rb_b + lt * rb_l, RC,
                         b * rc_b + lt * rc_l, MODE, D, HALF).to(DQ.dtype.element_ty).to(tl.float32)
        dyk = _grope_bwd(tl.load(DKO + g + offs).to(tl.float32), RA, b * ra_b + lt * ra_l, RB, b * rb_b + lt * rb_l, RC,
                         b * rc_b + lt * rc_l, MODE, D, HALF).to(DK.dtype.element_ty).to(tl.float32)
        gq = dyq * wq
        gk = dyk * wk
        dxq = rq * gq - q * (rq * rq * rq / D) * tl.sum(gq * q, axis=0)
        dxk = rk * gk - k * (rk * rk * rk / D) * tl.sum(gk * k, axis=0)
        tl.store(DQ + b * sdq_b + l * sdq_l + h * D + offs, dxq.to(DQ.dtype.element_ty))
        tl.store(DK + b * sdk_b + l * sdk_l + h * D + offs, dxk.to(DK.dtype.element_ty))
        tl.store(DV + b * sdv_b + l * sdv_l + h * D + offs, tl.load(DVO + g + offs))


def _grope_args(mode, rope, L_tot):
    """the 9 rope kernel arguments (A, a_b, a_l, B, b_b, b_l, C, c_b, c_l): element strides per batch (0 when broadcast) and token"""
    if mode == 2:
        out = []
        for t in rope:
            out += [t, t.stride(0) if t.shape[0] > 1 else 0, t.stride(-2)]
        return out
    pe = rope[0]
    tok = 2 if (pe.shape[2] == L_tot and pe.shape[1] == 1) else 1
    return [pe, pe.stride(0) if pe.shape[0] > 1 else 0, pe.stride(tok), pe, 0, 0, pe, 0, 0]


def _gsegs(meta, rest, heads):
    """per segment (qs, ks, vs, wq, wk, eps_q, eps_k, fused, src or None) from the flat tensor list"""
    segs, i = [], 0
    for fused, eq, ek in meta:
        if fused:
            src, wq, wk = rest[i:i + 3]
            hd = heads * wq.shape[0]
            segs.append((src[..., :hd], src[..., hd:2 * hd], src[..., 2 * hd:3 * hd], wq, wk, eq, ek, True, src))
            i += 3
        else:
            qs, ks, vs, wq, wk = rest[i:i + 5]
            segs.append((qs, ks, vs, wq, wk, eq, ek, False, None))
            i += 5
    return segs


class _GPrep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mode, heads, meta, n_rope, *tensors):
        rope, rest = tensors[:n_rope], tensors[n_rope:]
        segs = _gsegs(meta, rest, heads)
        D, B = segs[0][3].shape[0], segs[0][0].shape[0]
        lens = [s[0].shape[1] for s in segs]
        L_tot = sum(lens)
        qo = torch.empty((B, heads, L_tot, D), dtype=segs[0][0].dtype, device=segs[0][0].device)
        ko = torch.empty_like(qo)
        vo = torch.empty_like(qo)
        ra = _grope_args(mode, rope, L_tot)
        off = 0
        for (qs, ks, vs, wq, wk, eq, ek, _f, _s), L in zip(segs, lens):
            _gprep_fwd_kernel[(B * L, heads)](qs, qs.stride(0), qs.stride(1), ks, ks.stride(0), ks.stride(1), vs, vs.stride(0),
                                              vs.stride(1), wq, wk, *ra, qo, ko, vo, L, L_tot, off, heads, eq, ek,
                                              D=D, HALF=D // 2, MODE=mode, num_warps=1 if D <= 128 else 2)
            off += L
        ctx.mode, ctx.heads, ctx.meta, ctx.n_rope, ctx.lens = mode, heads, meta, n_rope, lens
        ctx.save_for_backward(*tensors)
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dqo, dko, dvo):
        tensors = ctx.saved_tensors
        n_rope, heads = ctx.n_rope, ctx.heads
        rope, rest = tensors[:n_rope], tensors[n_rope:]
        segs = _gsegs(ctx.meta, rest, heads)
        D, B, L_tot = segs[0][3].shape[0], segs[0][0].shape[0], sum(ctx.lens)
        dt, dev = segs[0][0].dtype, segs[0][0].device
        z = None
        if dqo is None or dko is None or dvo is None:
            z = torch.zeros((B, heads, L_tot, D), dtype=dt, device=dev)
        dqo, dko, dvo = ((z if t is None else t).contiguous() for t in (dqo, dko, dvo))
        ra = _grope_args(ctx.mode, rope, L_tot)
        grads, off, i = [], 0, 0
        for (qs, ks, vs, wq, wk, eq, ek, fused, src), L in zip(segs, ctx.lens):
            base = 4 + n_rope + i  # (mode, heads, meta, n_rope, *tensors)
            hd = heads * D
            if fused:
                need = ctx.needs_input_grad[base]
                buf = torch.empty((B, L, 3 * hd), dtype=dt, device=dev) if need else None
                if need:
                    dq, dk, dv = buf[..., :hd], buf[..., hd:2 * hd], buf[..., 2 * hd:]
                i += 3
            else:
                need = any(ctx.needs_input_grad[base:base + 3])
                if need:
                    dq, dk, dv = (torch.empty((B, L, hd), dtype=dt, device=dev) for _ in range(3))
                i += 5
            if need:
                _gprep_bwd_kernel[(B * L, heads)](qs, qs.stride(0), qs.stride(1), ks, ks.stride(0), ks.stride(1), wq, wk, *ra,
                                                  dqo, dko, dvo, dq, dq.stride(0), dq.stride(1), dk, dk.stride(0), dk.stride(1),
                                                  dv, dv.stride(0), dv.stride(1), L, L_tot, off, heads, eq, ek,
                                                  D=D, HALF=D // 2, MODE=ctx.mode, num_warps=1 if D <= 128 else 2)
            if fused:
                grads += [buf, None, None]
            else:
                grads += [dq if need and ctx.needs_input_grad[base] else None, dk if need and ctx.needs_input_grad[base + 1] else None,
                          dv if need and ctx.needs_input_grad[base + 2] else None, None, None]
            off += L
        return (None, None, None, None) + (None,) * n_rope + tuple(grads)


def _gnorm_ok(n, D, dtype, device):
    w = getattr(n, "weight", None)
    if w is None or tuple(w.shape) != (D,) or w.dtype != dtype or w.device != device or w.requires_grad:
        return "norm weight not a frozen (D,) tensor of the input dtype"
    if getattr(n, "comfy_cast_weights", False) or len(getattr(n, "weight_function", [])) or len(getattr(n, "bias_function", [])):
        return "norm with cast / weight functions"
    if tuple(getattr(n, "normalized_shape", (D,))) != (D,):
        return "norm shape"
    return None


def gprep_ok(segments, rope, heads, mode):
    """None when gprep can run these segments, else why not (the caller keeps its eager path). segments: dicts {'src': (B, L, 3 H D)}
    or {'q', 'k', 'v': (B, L, H D)}, each with 'nq' / 'nk' (the two RMSNorm modules)."""
    if triton is None:
        return "no triton"
    if mode not in GMODES or not segments:
        return "mode / no segment"
    t0 = segments[0]["src"] if "src" in segments[0] else segments[0]["q"]
    if not t0.is_cuda or t0.dtype not in (torch.bfloat16, torch.float16):
        return "not a CUDA bf16 / fp16 input"
    D = getattr(segments[0]["nq"], "weight", torch.empty(0)).shape[0] if getattr(segments[0]["nq"], "weight", None) is not None else 0
    if D not in (64, 128, 256):
        return f"head dim {D}"
    L_tot = 0
    for s in segments:
        ts = [s["src"]] if "src" in s else [s["q"], s["k"], s["v"]]
        for t in ts:
            if t.dim() != 3 or t.dtype != t0.dtype or t.device != t0.device or t.shape[0] != t0.shape[0] or t.stride(2) != 1:
                return "segments differ / inner dim not contiguous"
        if "src" in s and s["src"].shape[2] != 3 * heads * D:
            return "fused source not 3 H D wide"
        if "src" not in s and not (s["q"].shape == s["k"].shape == s["v"].shape and s["q"].shape[2] == heads * D):
            return "separate q / k / v shapes"
        for n in (s["nq"], s["nk"]):
            why = _gnorm_ok(n, D, t0.dtype, t0.device)
            if why:
                return why
        L_tot += ts[0].shape[1]
    m = GMODES[mode]
    if m == 2:
        if not isinstance(rope, (tuple, list)) or len(rope) != 3:
            return "half rope needs (cos, sin, nsin)"
        cos, sn, ns = rope
        for t, w in ((cos, D), (sn, D // 2), (ns, D // 2)):
            if (t.dim() != 3 or t.shape[0] != 1 or t.shape[1] != L_tot or t.shape[2] != w or t.stride(2) != 1 or t.dtype != torch.float32
                    or t.device != t0.device):
                return "half rope layout"
    else:
        pe = rope[0] if isinstance(rope, (tuple, list)) else rope
        want = torch.float32 if m == 0 else t0.dtype
        if (pe is None or pe.dtype != want or pe.dim() != 6 or tuple(pe.shape[3:]) != (D // 2, 2, 2) or pe.device != t0.device
                or pe.stride(-1) != 1 or pe.stride(-2) != 2 or pe.stride(-3) != 4 or pe.shape[0] not in (1, t0.shape[0])
                or sorted((pe.shape[1], pe.shape[2])) != [1, L_tot]):
            return "pe layout"
    return None


def gprep(segments, rope, heads, mode):
    """q, k, v (B, H, L_total, D) contiguous for gprep_ok segments (see the block comment)."""
    m = GMODES[mode]
    tensors, meta = [], []
    for s in segments:
        D = s["nq"].weight.shape[0]
        dt = (s["src"] if "src" in s else s["q"]).dtype
        meta.append(("src" in s, _eps(s["nq"], dt), _eps(s["nk"], dt)))
        tensors += ([s["src"]] if "src" in s else [s["q"], s["k"], s["v"]]) + [s["nq"].weight, s["nk"].weight]
    rt = tuple(rope) if m == 2 else ((rope[0],) if isinstance(rope, (tuple, list)) else (rope,))
    return _GPrep.apply(m, heads, tuple(meta), len(rt), *rt, *tensors)


def _grms(D, eps, dtype, device, g):
    import comfy.ops
    n = comfy.ops.disable_weight_init.RMSNorm(D, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
    with torch.no_grad():
        n.weight.copy_(1.0 + 0.1 * torch.randn(D, generator=g))
    n.requires_grad_(False)
    return n


def gprep_eager(family, segs, rope, heads):
    """ComfyUI 694815f's own composition that gprep replaces (training mode) -> q, k, v (B, H, L_total, D):
    z-image  JointAttention.forward: split [H D] x 3, view (B, L, H, D), q_norm / k_norm, flux math _apply_rope1 with the
             (1, L, 1, D/2, 2, 2) fp32 freqs_cis, movedim(1, 2) (the n_rep == 1 repeat is a copy);
    ideogram4 Ideogram4Attention.forward: view (B, L, 3, H, D).unbind(2), norm_q / norm_k, transpose(1, 2), llama apply_rope;
    qwen     Attention.forward: per stream view (B, L, H, D).transpose(1, 2).contiguous() (v without contiguous), the stream's
             norms, cat [txt, img] on the token dim, _apply_rope1 with the pe in the input dtype."""
    from comfy.ldm.flux.math import _apply_rope1
    qs, ks, vs = [], [], []
    for s in segs:
        if "src" in s:
            B, L, W = s["src"].shape
            D = W // (3 * heads)
            if family == "ideogram4":
                q, k, v = s["src"].view(B, L, 3, heads, D).unbind(dim=2)
            else:
                q, k, v = (t.view(B, L, heads, D) for t in torch.split(s["src"], [heads * D] * 3, dim=-1))
            q, k = s["nq"](q), s["nk"](k)
            qs.append(q), ks.append(k), vs.append(v)
        else:
            B, L, W = s["q"].shape
            D = W // heads
            q = s["nq"](s["q"].view(B, L, heads, D).transpose(1, 2).contiguous())
            k = s["nk"](s["k"].view(B, L, heads, D).transpose(1, 2).contiguous())
            v = s["v"].view(B, L, heads, D).transpose(1, 2)
            qs.append(q), ks.append(k), vs.append(v)
    if family == "z-image":  # one fused segment (the joint sequence)
        q, k = _apply_rope1(qs[0], rope[0]), _apply_rope1(ks[0], rope[0])
        return q.movedim(1, 2), k.movedim(1, 2), vs[0].movedim(1, 2)
    if family == "ideogram4":
        from comfy.text_encoders.llama import apply_rope
        q, k = apply_rope(qs[0].transpose(1, 2), ks[0].transpose(1, 2), rope)
        return q, k, vs[0].transpose(1, 2)
    q, k, v = torch.cat(qs, dim=2), torch.cat(ks, dim=2), torch.cat(vs, dim=2)  # qwen
    return _apply_rope1(q, rope[0]), _apply_rope1(k, rope[0]), v


def gprep_selfcheck(device, family, heads=2, lens=(6, 10), D=128, dtype=torch.bfloat16, seed=0, tol=2e-2):
    """fwd + bwd of gprep vs gprep_eager on random inputs shaped like the family's attention: rel L2 of q / k / v and of every
    input gradient; ok when all <= tol. -> (ok, report)"""
    from comfy.ldm.flux.layers import EmbedND
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = 1
    L_tot = sum(lens) if family == "qwen" else lens[0]
    mode = {"z-image": "pairs32", "ideogram4": "half", "qwen": "pairs16"}[family]
    eps = {"z-image": (None, None), "ideogram4": (1e-5, 1e-5), "qwen": (1e-6, 1e-6)}[family]
    segs = []
    for Ls in (lens if family == "qwen" else lens[:1]):
        nq, nk = _grms(D, eps[0], dtype, device, g), _grms(D, eps[1], dtype, device, g)
        if family == "qwen":
            segs.append({"q": (torch.randn(B, Ls, heads * D, generator=g) * 2).to(device, dtype),
                         "k": (torch.randn(B, Ls, heads * D, generator=g) * 2).to(device, dtype),
                         "v": (torch.randn(B, Ls, heads * D, generator=g) * 2).to(device, dtype), "nq": nq, "nk": nk})
        else:
            segs.append({"src": (torch.randn(B, Ls, 3 * heads * D, generator=g) * 2).to(device, dtype), "nq": nq, "nk": nk})
    if mode == "half":
        th = torch.randn(L_tot, D // 2, generator=g) * 3
        emb = torch.cat((th, th), dim=-1)
        cos, sin = emb.cos().unsqueeze(0).to(device), emb.sin().unsqueeze(0).to(device)
        rope = (cos, sin[..., : D // 2], -sin[..., D // 2:])
    else:
        ids = (torch.randn(B, L_tot, 3, generator=g) * 40).to(device)
        pe = EmbedND(dim=D, theta=256 if family == "z-image" else 10000, axes_dim=[D // 4, 3 * D // 8, 3 * D // 8])(ids)
        rope = (pe.movedim(1, 2),) if family == "z-image" else (pe.to(dtype).contiguous(),)
    gq = [torch.randn(B, heads, L_tot, D, generator=g).to(device, dtype) for _ in range(3)]
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))  # noqa: E731
    res = {}
    for tag in ("fused", "eager"):
        ins = []
        for s in segs:
            c = dict(s)
            for kk in ("src", "q", "k", "v"):
                if kk in c:
                    c[kk] = s[kk].detach().clone().requires_grad_(True)
                    ins.append(c[kk])
            if tag == "fused":
                c = {k2: v2 for k2, v2 in c.items()}
            ins_s = c
            s["_c_" + tag] = ins_s
        cs = [s["_c_" + tag] for s in segs]
        if tag == "fused":
            why = gprep_ok(cs, rope, heads, mode)
            if why is not None:
                return False, {"why": why}
            out = gprep(cs, rope, heads, mode)
        else:
            out = gprep_eager(family, cs, rope, heads)
        torch.autograd.backward(list(out), gq)
        res[tag] = ([o.detach() for o in out], [t.grad for t in ins])
    r = {n: rel(f, e) for n, f, e in zip(("q", "k", "v"), res["fused"][0], res["eager"][0])}
    for j, (f, e) in enumerate(zip(res["fused"][1], res["eager"][1])):
        r[f"d{j}"] = rel(f, e)
    for s in segs:
        s.pop("_c_fused", None), s.pop("_c_eager", None)
    return all(v <= tol for v in r.values()), r


# ----------------------------------------------------------------------------------------------------------------------
# gprep wired into the other families' attention forwards (T213, 28 Oct; G56CT_FUSE_PREP on z-image / ideogram4 / qwen-image).
# Each forward computes the projections exactly as ComfyUI 694815f, then either gprep + the same attention call, or (gprep_ok says
# no, an attn1 patch, a mask the fused path does not take) ComfyUI's own remainder on the SAME projections (no recompute).
# ----------------------------------------------------------------------------------------------------------------------
GFAMILIES = {"z-image": "z-image", "ideogram4": "ideogram4", "qwen-image": "qwen"}


def _zimage_attn_forward(self, x, x_mask, freqs_cis, transformer_options={}):  # noqa: B006  (ComfyUI's signature)
    from comfy.ldm.modules.attention import optimized_attention_masked
    bsz, seqlen, _ = x.shape
    qkv = self.qkv(x)
    if FLAGS.get("gprep"):
        why = None
        if x_mask is not None:
            why = "z-image: mask"
        elif self.n_local_kv_heads != self.n_local_heads:
            why = "z-image: GQA"
        else:
            why = gprep_ok([{"src": qkv, "nq": self.q_norm, "nk": self.k_norm}], (freqs_cis,), self.n_local_heads, "pairs32")
        if why is None:
            STATS["gprep"] = STATS.get("gprep", 0) + 1
            q, k, v = gprep([{"src": qkv, "nq": self.q_norm, "nk": self.k_norm}], (freqs_cis,), self.n_local_heads, "pairs32")
            out = optimized_attention_masked(q, k, v, self.n_local_heads, x_mask, skip_reshape=True,
                                             transformer_options=transformer_options)
            return self.out(out)
        _note_fallback(why)
    from comfy.ldm.flux.math import apply_rope
    xq, xk, xv = torch.split(qkv, [self.n_local_heads * self.head_dim, self.n_local_kv_heads * self.head_dim,
                                   self.n_local_kv_heads * self.head_dim], dim=-1)
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xq = self.q_norm(xq)
    xk = self.k_norm(xk)
    xq, xk = apply_rope(xq, xk, freqs_cis)
    n_rep = self.n_local_heads // self.n_local_kv_heads
    if n_rep >= 1:
        xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
    output = optimized_attention_masked(xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2), self.n_local_heads, x_mask,
                                        skip_reshape=True, transformer_options=transformer_options)
    return self.out(output)


def _ideogram4_attn_forward(self, x, attn_mask, freqs_cis, transformer_options={}):  # noqa: B006
    from comfy.ldm.modules.attention import optimized_attention_masked
    batch_size, seq_len, _ = x.shape
    qkv = self.qkv(x)
    if FLAGS.get("gprep"):
        rope = tuple(freqs_cis) if isinstance(freqs_cis, (tuple, list)) and len(freqs_cis) == 3 else None
        segs = [{"src": qkv, "nq": self.norm_q, "nk": self.norm_k}]
        why = "ideogram4: freqs_cis not (cos, sin, nsin)" if rope is None else gprep_ok(segs, rope, self.num_heads, "half")
        if why is None:
            STATS["gprep"] = STATS.get("gprep", 0) + 1
            q, k, v = gprep(segs, rope, self.num_heads, "half")
            out = optimized_attention_masked(q, k, v, self.num_heads, attn_mask, skip_reshape=True,
                                             transformer_options=transformer_options)
            return self.o(out)
        _note_fallback(why)
    from comfy.text_encoders.llama import apply_rope
    q, k, v = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim).unbind(dim=2)
    q = self.norm_q(q)
    k = self.norm_k(k)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    q, k = apply_rope(q, k, freqs_cis)
    out = optimized_attention_masked(q, k, v, self.num_heads, attn_mask, skip_reshape=True, transformer_options=transformer_options)
    return self.o(out)


def _qwen_attn_forward(self, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None, attention_mask=None,
                       image_rotary_emb=None, transformer_options={}):  # noqa: B006
    from comfy.ldm.modules.attention import optimized_attention_masked
    batch_size, seq_img, seq_txt = hidden_states.shape[0], hidden_states.shape[1], encoder_hidden_states.shape[1]
    patches = transformer_options.get("patches", {})
    iq, ik, iv = self.to_q(hidden_states), self.to_k(hidden_states), self.to_v(hidden_states)
    tq, tk, tv = self.add_q_proj(encoder_hidden_states), self.add_k_proj(encoder_hidden_states), self.add_v_proj(encoder_hidden_states)
    if encoder_hidden_states_mask is not None:
        attn_mask = torch.zeros((batch_size, 1, seq_txt + seq_img), dtype=hidden_states.dtype, device=hidden_states.device)
        attn_mask[:, 0, :seq_txt] = encoder_hidden_states_mask
    else:
        attn_mask = None
    joint = None
    if FLAGS.get("gprep"):
        segs = [{"q": tq, "k": tk, "v": tv, "nq": self.norm_added_q, "nk": self.norm_added_k},
                {"q": iq, "k": ik, "v": iv, "nq": self.norm_q, "nk": self.norm_k}]
        why = "qwen: attn1_patch" if "attn1_patch" in patches else gprep_ok(segs, (image_rotary_emb,), self.heads, "pairs16")
        if why is None:
            STATS["gprep"] = STATS.get("gprep", 0) + 1
            q, k, v = gprep(segs, (image_rotary_emb,), self.heads, "pairs16")
            joint = optimized_attention_masked(q, k, v, self.heads, attn_mask, transformer_options=transformer_options, skip_reshape=True)
        else:
            _note_fallback(why)
    if joint is None:  # ComfyUI 694815f's remainder on the same projections
        from comfy.ldm.flux.math import apply_rope1
        extra_options = transformer_options.copy()
        img_query = iq.view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
        img_key = ik.view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
        img_value = iv.view(batch_size, seq_img, self.heads, -1).transpose(1, 2)
        txt_query = tq.view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
        txt_key = tk.view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
        txt_value = tv.view(batch_size, seq_txt, self.heads, -1).transpose(1, 2)
        img_query, img_key = self.norm_q(img_query), self.norm_k(img_key)
        txt_query, txt_key = self.norm_added_q(txt_query), self.norm_added_k(txt_key)
        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)
        extra_options["img_slice"] = [txt_query.shape[2], joint_query.shape[2]]
        if "attn1_patch" in patches:
            for p in patches["attn1_patch"]:
                out = p(joint_query, joint_key, joint_value, pe=image_rotary_emb, attn_mask=encoder_hidden_states_mask,
                        extra_options=extra_options)
                joint_query, joint_key, joint_value, image_rotary_emb, encoder_hidden_states_mask = (
                    out.get("q", joint_query), out.get("k", joint_key), out.get("v", joint_value), out.get("pe", image_rotary_emb),
                    out.get("attn_mask", encoder_hidden_states_mask))
        joint_query = apply_rope1(joint_query, image_rotary_emb)
        joint_key = apply_rope1(joint_key, image_rotary_emb)
        joint = optimized_attention_masked(joint_query, joint_key, joint_value, self.heads, attn_mask,
                                           transformer_options=transformer_options, skip_reshape=True)
    txt_attn_output = joint[:, :seq_txt, :]
    img_attn_output = joint[:, seq_txt:, :]
    img_attn_output = self.to_out[0](img_attn_output)
    img_attn_output = self.to_out[1](img_attn_output)
    txt_attn_output = self.to_add_out(txt_attn_output)
    return img_attn_output, txt_attn_output


def install_family(fam, prep=True):
    """Patch the attention class of a non-flux family (z-image: lumina JointAttention, ideogram4: Ideogram4Attention, qwen-image:
    qwen_image Attention) with the gprep forward (the forward in place before = _FALLBACK). Idempotent; returns 1 when patched."""
    if fam == "z-image":
        import comfy.ldm.lumina.model as M
        cls, fn, key = M.JointAttention, _zimage_attn_forward, "zimage_attn"
    elif fam == "ideogram4":
        import comfy.ldm.ideogram4.model as M
        cls, fn, key = M.Ideogram4Attention, _ideogram4_attn_forward, "ideogram4_attn"
    elif fam == "qwen-image":
        import comfy.ldm.qwen_image.model as M
        cls, fn, key = M.Attention, _qwen_attn_forward, "qwen_attn"
    else:
        return 0
    FLAGS["gprep"] = bool(prep)
    if getattr(cls.forward, "_g56_fused", False):
        return 0
    _FALLBACK[key] = cls.forward
    fn._g56_fused = True
    cls.forward = fn
    return 1


def uninstall_family(fam):
    import importlib
    mods = {"z-image": ("comfy.ldm.lumina.model", "JointAttention", "zimage_attn"),
            "ideogram4": ("comfy.ldm.ideogram4.model", "Ideogram4Attention", "ideogram4_attn"),
            "qwen-image": ("comfy.ldm.qwen_image.model", "Attention", "qwen_attn")}
    if fam not in mods:
        return
    m, c, key = mods[fam]
    cls = getattr(importlib.import_module(m), c)
    if key in _FALLBACK and getattr(cls.forward, "_g56_fused", False):
        cls.forward = _FALLBACK.pop(key)
    FLAGS["gprep"] = False


def family_forward_selfcheck(device, fam, dtype=torch.bfloat16, seed=0, tol=2e-2):
    """The patched attention forward (install_family) vs ComfyUI's own on a small random module of the family (training mode):
    rel L2 of every output and input gradient, and how often gprep ran (STATS['gprep'] must grow: no silent fallback).
    -> (ok, report). Restores the original forward."""
    import comfy.ops
    import comfy.model_management
    from comfy.ldm.flux.layers import EmbedND
    ops = comfy.ops.disable_weight_init
    prev_training = comfy.model_management.in_training
    comfy.model_management.in_training = True
    g = torch.Generator(device="cpu").manual_seed(seed)

    def init(m):
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.dim() >= 2:
                    p.copy_((torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5).to(p.dtype))
                elif "norm" in n:
                    p.copy_((1.0 + 0.1 * torch.randn(p.shape, generator=g)).to(p.dtype))
                else:
                    p.copy_((0.02 * torch.randn(p.shape, generator=g)).to(p.dtype))
        return m.requires_grad_(False)
    L = 12
    if fam == "z-image":
        import comfy.ldm.lumina.model as M
        mod = init(M.JointAttention(256, 2, None, True, operation_settings={"operations": ops, "dtype": dtype, "device": device}))
        pe = EmbedND(dim=128, theta=256, axes_dim=[32, 48, 48])(torch.randn(1, L, 3, generator=g).to(device) * 20).movedim(1, 2)
        pe = torch.cat([pe[:, :5], pe[:, 5:]], dim=1)
        xs = [torch.randn(1, L, 256, generator=g).to(device, dtype)]
        call = lambda a: mod(a[0], None, pe)  # noqa: E731
    elif fam == "ideogram4":
        import comfy.ldm.ideogram4.model as M
        mod = init(M.Ideogram4Attention(512, 2, eps=1e-5, dtype=dtype, device=device, operations=ops))
        th = torch.randn(L, 128, generator=g) * 3
        emb = torch.cat((th, th), -1)
        cos, sin = emb.cos().unsqueeze(0).to(device), emb.sin().unsqueeze(0).to(device)
        fc = (cos, sin[..., :128], -sin[..., 128:])
        xs = [torch.randn(1, L, 512, generator=g).to(device, dtype)]
        call = lambda a: mod(a[0], None, fc)  # noqa: E731
    elif fam == "qwen-image":
        import comfy.ldm.qwen_image.model as M
        mod = init(M.Attention(query_dim=256, dim_head=128, heads=2, out_dim=256, bias=True, eps=1e-6, dtype=dtype, device=device,
                               operations=ops))
        pq = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])(torch.randn(1, L + 5, 3, generator=g).to(device) * 20)
        pq = pq.to(dtype).contiguous()
        xs = [torch.randn(1, L, 256, generator=g).to(device, dtype), torch.randn(1, 5, 256, generator=g).to(device, dtype)]
        call = lambda a: mod(a[0], a[1], None, None, pq)  # noqa: E731
    else:
        return False, {"why": f"family {fam}"}
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))  # noqa: E731
    outs, n0 = {}, STATS.get("gprep", 0)
    try:
        for tag in ("eager", "fused"):
            if tag == "fused":
                install_family(fam, prep=True)
            else:
                uninstall_family(fam)
            leaves = [x.clone().requires_grad_(True) for x in xs]
            o = call(leaves)
            o = o if isinstance(o, tuple) else (o,)
            gs = [torch.linspace(-1, 1, t.numel(), device=t.device).view_as(t).to(t.dtype) * 0.01 for t in o]
            torch.autograd.backward(list(o), gs)
            outs[tag] = ([t.detach() for t in o], [x.grad for x in leaves])
    finally:
        uninstall_family(fam)
        comfy.model_management.in_training = prev_training
    r = {f"o{i}": rel(a, b) for i, (a, b) in enumerate(zip(outs["fused"][0], outs["eager"][0]))}
    r.update({f"g{i}": rel(a, b) for i, (a, b) in enumerate(zip(outs["fused"][1], outs["eager"][1]))})
    r["gprep_calls"] = STATS.get("gprep", 0) - n0
    return all(v <= tol for k, v in r.items() if k != "gprep_calls") and r["gprep_calls"] > 0, r


# ----------------------------------------------------------------------------------------------------------------------
# RMSNorm modulation for ideogram4 / z-image (T213, 28 Oct; G56CT_FUSE_MOD there): their blocks are
#   h = RMSNorm1(x) * (1 + scale)              (F.rms_norm with weight: one rounding; the product rounded)
#   x = x + tanh(gate) * RMSNorm2(y)           (the norm rounded, gate * it rounded, the add rounded)
# rms_mod / rms_gate_res do each as one kernel forward and one backward (d scale / d gate: per-block fp32 column sums over the
# tokens, the bf16 product first as eager's MulBackward). (1 + scale) and tanh(gate) stay eager (tiny (B, 1, C) tensors), so their
# own rounding / backward is ComfyUI's. Norm weights frozen (base weights), as for flux.
# ----------------------------------------------------------------------------------------------------------------------
if triton is not None:
    @triton.jit
    def _rmsmod_fwd_kernel(X, s_xb, s_xl, W, S1, s_sb, OUT, RSTD, L, C, eps, BLOCK: tl.constexpr):
        l = tl.program_id(0)
        b = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        x = tl.load(X + b * s_xb + l * s_xl + cols, mask=m, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / C + eps)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        y = (x * rstd * w).to(OUT.dtype.element_ty).to(tl.float32)
        s1 = tl.load(S1 + b * s_sb + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + (b * L + l) * C + cols, (y * s1).to(OUT.dtype.element_ty), mask=m)
        tl.store(RSTD + b * L + l, rstd)

    @triton.jit
    def _rmsmod_bwd_kernel(X, s_xb, s_xl, W, S1, s_sb, DO, RSTD, DX, PS, L, C, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        pblk = tl.program_id(0)
        b = tl.program_id(1)
        nblk = tl.num_programs(0)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        s1 = tl.load(S1 + b * s_sb + cols, mask=m, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in range(ROWS):
            l = pblk * ROWS + r
            if l < L:
                x = tl.load(X + b * s_xb + l * s_xl + cols, mask=m, other=0.0).to(tl.float32)
                rstd = tl.load(RSTD + b * L + l)
                y = (x * rstd * w).to(DX.dtype.element_ty).to(tl.float32)
                d = tl.load(DO + (b * L + l) * C + cols, mask=m, other=0.0).to(tl.float32)
                acc += (d * y).to(DX.dtype.element_ty).to(tl.float32)  # eager: the bf16 product, then the fp32 sum
                gy = tl.where(m, (d * s1).to(DX.dtype.element_ty).to(tl.float32) * w, 0.0)
                dx = rstd * gy - x * (rstd * rstd * rstd / C) * tl.sum(gy * x, axis=0)
                tl.store(DX + (b * L + l) * C + cols, dx.to(DX.dtype.element_ty), mask=m)
        tl.store(PS + (b * nblk + pblk) * C + cols, acc, mask=m)

    @triton.jit
    def _rmsgres_fwd_kernel(X, s_xb, s_xl, Y, s_yb, s_yl, W, G, s_gb, OUT, RSTD, L, C, eps, BLOCK: tl.constexpr):
        l = tl.program_id(0)
        b = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        y = tl.load(Y + b * s_yb + l * s_yl + cols, mask=m, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(y * y, axis=0) / C + eps)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        z = (y * rstd * w).to(OUT.dtype.element_ty).to(tl.float32)
        g = tl.load(G + b * s_gb + cols, mask=m, other=0.0).to(tl.float32)
        t = (g * z).to(OUT.dtype.element_ty).to(tl.float32)
        x = tl.load(X + b * s_xb + l * s_xl + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + (b * L + l) * C + cols, (x + t).to(OUT.dtype.element_ty), mask=m)
        tl.store(RSTD + b * L + l, rstd)

    @triton.jit
    def _rmsgres_bwd_kernel(DO, Y, s_yb, s_yl, W, G, s_gb, RSTD, DY, PG, L, C, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        pblk = tl.program_id(0)
        b = tl.program_id(1)
        nblk = tl.num_programs(0)
        cols = tl.arange(0, BLOCK)
        m = cols < C
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + b * s_gb + cols, mask=m, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in range(ROWS):
            l = pblk * ROWS + r
            if l < L:
                y = tl.load(Y + b * s_yb + l * s_yl + cols, mask=m, other=0.0).to(tl.float32)
                rstd = tl.load(RSTD + b * L + l)
                z = (y * rstd * w).to(DY.dtype.element_ty).to(tl.float32)
                d = tl.load(DO + (b * L + l) * C + cols, mask=m, other=0.0).to(tl.float32)
                acc += (d * z).to(DY.dtype.element_ty).to(tl.float32)
                gz = tl.where(m, (d * g).to(DY.dtype.element_ty).to(tl.float32) * w, 0.0)
                dy = rstd * gz - y * (rstd * rstd * rstd / C) * tl.sum(gz * y, axis=0)
                tl.store(DY + (b * L + l) * C + cols, dy.to(DY.dtype.element_ty), mask=m)
        tl.store(PG + (b * nblk + pblk) * C + cols, acc, mask=m)


class _RMSMod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, s1, eps):
        B, L, C = x.shape
        out = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
        rstd = torch.empty((B, L), dtype=torch.float32, device=x.device)
        _rmsmod_fwd_kernel[(L, B)](x, x.stride(0), x.stride(1), w, s1, s1.stride(0), out, rstd, L, C, eps, BLOCK=_blk(C), num_warps=8)
        ctx.save_for_backward(x, w, s1, rstd)
        return out

    @staticmethod
    def backward(ctx, do):
        x, w, s1, rstd = ctx.saved_tensors
        B, L, C = x.shape
        do = do.contiguous()
        nblk = triton.cdiv(L, _ROWS)
        dx = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
        ps = torch.empty((B, nblk, C), dtype=torch.float32, device=x.device)
        _rmsmod_bwd_kernel[(nblk, B)](x, x.stride(0), x.stride(1), w, s1, s1.stride(0), do, rstd, dx, ps, L, C, ROWS=_ROWS,
                                      BLOCK=_blk(C), num_warps=8)
        ds = ps.sum(dim=1, keepdim=True).to(s1.dtype) if ctx.needs_input_grad[2] else None
        return dx, None, ds, None


class _RMSGateRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, w, g, eps):
        B, L, C = y.shape
        out = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
        rstd = torch.empty((B, L), dtype=torch.float32, device=x.device)
        _rmsgres_fwd_kernel[(L, B)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), w, g, g.stride(0), out, rstd, L, C, eps,
                                    BLOCK=_blk(C), num_warps=8)
        ctx.save_for_backward(y, w, g, rstd)
        return out

    @staticmethod
    def backward(ctx, do):
        y, w, g, rstd = ctx.saved_tensors
        B, L, C = y.shape
        do = do.contiguous()
        nblk = triton.cdiv(L, _ROWS)
        dy = torch.empty((B, L, C), dtype=y.dtype, device=y.device)
        pg = torch.empty((B, nblk, C), dtype=torch.float32, device=y.device)
        _rmsgres_bwd_kernel[(nblk, B)](do, y, y.stride(0), y.stride(1), w, g, g.stride(0), rstd, dy, pg, L, C, ROWS=_ROWS,
                                       BLOCK=_blk(C), num_warps=8)
        dg = pg.sum(dim=1, keepdim=True).to(g.dtype) if ctx.needs_input_grad[3] else None
        return do, dy, None, dg, None


def _rms_ok(norm, x, *vs):
    """None when the RMS kernels can take x (B, L, C), this RMSNorm (frozen (C,) weight of x's dtype) and the (B, 1, C) vectors."""
    why = _mod_ok(x, *vs)
    if why:
        return why
    return _gnorm_ok(norm, x.shape[2], x.dtype, x.device)


def rms_mod(norm, x, s1):
    """norm(x) * s1 (s1 = 1 + scale, (B, 1, C)) fused when possible, else ComfyUI's own two ops."""
    if FLAGS.get("rmsmod") and _rms_ok(norm, x, s1) is None:
        STATS["rmsmod"] = STATS.get("rmsmod", 0) + 1
        return _RMSMod.apply(x, norm.weight, s1, _eps(norm, x.dtype))
    STATS["rmsmod_eager"] = STATS.get("rmsmod_eager", 0) + 1
    return norm(x) * s1


def rms_gate_res(x, y, norm, g):
    """x + g * norm(y) (g = tanh(gate), (B, 1, C)) fused when possible."""
    if FLAGS.get("rmsmod") and _rms_ok(norm, y, g) is None and _mod_ok(x) is None and x.shape == y.shape:
        STATS["rmsgres"] = STATS.get("rmsgres", 0) + 1
        return _RMSGateRes.apply(x, y, norm.weight, g, _eps(norm, y.dtype))
    STATS["rmsgres_eager"] = STATS.get("rmsgres_eager", 0) + 1
    return x + g * norm(y)


def _ideogram4_block_forward(self, x, attn_mask, freqs_cis, adaln_input, transformer_options={}):  # noqa: B006
    mod = self.adaln_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
    gate_msa = torch.tanh(gate_msa)
    gate_mlp = torch.tanh(gate_mlp)
    scale_msa = 1.0 + scale_msa
    scale_mlp = 1.0 + scale_mlp
    attn_out = self.attention(rms_mod(self.attention_norm1, x, scale_msa), attn_mask, freqs_cis, transformer_options=transformer_options)
    x = rms_gate_res(x, attn_out, self.attention_norm2, gate_msa)
    x = rms_gate_res(x, self.feed_forward(rms_mod(self.ffn_norm1, x, scale_mlp)), self.ffn_norm2, gate_mlp)
    return x


def _zimage_block_forward(self, x, x_mask, freqs_cis, adaln_input=None, timestep_zero_index=None, transformer_options={}):  # noqa: B006
    if not self.modulation or timestep_zero_index is not None:
        return _FALLBACK["zimage_block"](self, x, x_mask, freqs_cis, adaln_input=adaln_input, timestep_zero_index=timestep_zero_index,
                                         transformer_options=transformer_options)
    from comfy.ldm.lumina.model import clamp_fp16  # fp16 only (nan_to_num), as ComfyUI's block
    scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
    attn = clamp_fp16(self.attention(rms_mod(self.attention_norm1, x, 1 + scale_msa.unsqueeze(1)), x_mask, freqs_cis,
                                     transformer_options=transformer_options))
    x = rms_gate_res(x, attn, self.attention_norm2, gate_msa.unsqueeze(1).tanh())
    x = rms_gate_res(x, clamp_fp16(self.feed_forward(rms_mod(self.ffn_norm1, x, 1 + scale_mlp.unsqueeze(1)))), self.ffn_norm2,
                     gate_mlp.unsqueeze(1).tanh())
    return x


def install_family_mod(fam, on=True):
    """Patch ideogram4 / z-image transformer blocks with the rms_mod / rms_gate_res forwards (the forward in place before = _FALLBACK)."""
    if fam == "z-image":
        import comfy.ldm.lumina.model as M
        cls, fn, key = M.JointTransformerBlock, _zimage_block_forward, "zimage_block"
    elif fam == "ideogram4":
        import comfy.ldm.ideogram4.model as M
        cls, fn, key = M.Ideogram4TransformerBlock, _ideogram4_block_forward, "ideogram4_block"
    else:
        return 0
    FLAGS["rmsmod"] = bool(on)
    if getattr(cls.forward, "_g56_fused", False):
        return 0
    _FALLBACK[key] = cls.forward
    fn._g56_fused = True
    cls.forward = fn
    return 1


def uninstall_family_mod(fam):
    import importlib
    mods = {"z-image": ("comfy.ldm.lumina.model", "JointTransformerBlock", "zimage_block"),
            "ideogram4": ("comfy.ldm.ideogram4.model", "Ideogram4TransformerBlock", "ideogram4_block")}
    if fam not in mods:
        return
    m, c, key = mods[fam]
    cls = getattr(importlib.import_module(m), c)
    if key in _FALLBACK and getattr(cls.forward, "_g56_fused", False):
        cls.forward = _FALLBACK.pop(key)
    FLAGS["rmsmod"] = False


def rms_mod_selfcheck(device, L=40, C=96, dtype=torch.bfloat16, seed=0, tol=2e-2):
    """rms_mod / rms_gate_res vs the eager composition (fwd + every input gradient): rel L2 each. -> (ok, report)"""
    import comfy.ops
    g = torch.Generator(device="cpu").manual_seed(seed)
    n1 = comfy.ops.disable_weight_init.RMSNorm(C, eps=1e-5, elementwise_affine=True, dtype=dtype, device=device)
    n2 = comfy.ops.disable_weight_init.RMSNorm(C, eps=1e-5, elementwise_affine=True, dtype=dtype, device=device)
    with torch.no_grad():
        n1.weight.copy_(1.0 + 0.1 * torch.randn(C, generator=g))
        n2.weight.copy_(1.0 + 0.1 * torch.randn(C, generator=g))
    n1.requires_grad_(False), n2.requires_grad_(False)
    x = (torch.randn(1, L, C, generator=g) * 3).to(device, dtype)
    y = (torch.randn(1, L, C, generator=g) * 2).to(device, dtype)
    mod = (torch.randn(1, 1, 2 * C, generator=g) * 0.5).to(device, dtype)
    go = torch.randn(1, L, C, generator=g).to(device, dtype)
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))  # noqa: E731
    rep, prev, n0 = {}, FLAGS.get("rmsmod", False), (STATS.get("rmsmod", 0), STATS.get("rmsgres", 0))
    try:
        for tag in ("fused", "eager"):
            FLAGS["rmsmod"] = tag == "fused"
            xs, ys, ms = x.clone().requires_grad_(True), y.clone().requires_grad_(True), mod.clone().requires_grad_(True)
            sc, gt = ms.chunk(2, dim=-1)
            a = rms_mod(n1, xs, 1.0 + sc)
            o = rms_gate_res(xs, ys, n2, torch.tanh(gt))
            torch.autograd.backward([a, o], [go, go])
            rep[tag] = (a.detach(), o.detach(), xs.grad, ms.grad, ys.grad)
    finally:
        FLAGS["rmsmod"] = prev
    names = ("rmsmod", "rmsgres", "dx", "dmod", "dy")
    r = {n: rel(f, e) for n, f, e in zip(names, rep["fused"], rep["eager"])}
    ran = STATS.get("rmsmod", 0) > n0[0] and STATS.get("rmsgres", 0) > n0[1]  # the fused path really ran (no silent eager)
    r["fused_ran"] = ran
    return all(v <= tol for k, v in r.items() if k != "fused_ran") and ran, r


def family_block_selfcheck(device, fam, dtype=torch.bfloat16, seed=0, tol=2e-2):
    """install_family_mod's block forward vs ComfyUI's own on a small random block of the family (training mode): rel L2 of the
    output and the gradients of x and the adaLN input; the fused kernels must have run. -> (ok, report). Restores the forward."""
    import comfy.ops
    import comfy.model_management
    ops = comfy.ops.disable_weight_init
    prev_training = comfy.model_management.in_training
    comfy.model_management.in_training = True
    g = torch.Generator(device="cpu").manual_seed(seed)

    def init(m):
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.dim() >= 2:
                    p.copy_((torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5).to(p.dtype))
                elif "norm" in n:
                    p.copy_((1.0 + 0.1 * torch.randn(p.shape, generator=g)).to(p.dtype))
                else:
                    p.copy_((0.1 * torch.randn(p.shape, generator=g)).to(p.dtype))
        return m.requires_grad_(False)
    L, C = 12, 256
    if fam == "ideogram4":
        import comfy.ldm.ideogram4.model as M
        blk = init(M.Ideogram4TransformerBlock(C, 512, 2, 1e-5, 64, dtype=dtype, device=device, operations=ops))
        x0 = torch.randn(1, L, C, generator=g).to(device, dtype)
        a0 = torch.randn(1, 1, 64, generator=g).to(device, dtype)
        th = torch.randn(L, 64, generator=g) * 3
        emb = torch.cat((th, th), -1)
        cos, sin = emb.cos().unsqueeze(0).to(device), emb.sin().unsqueeze(0).to(device)
        fc = (cos, sin[..., :64], -sin[..., 64:])
        call = lambda x, a: blk(x, None, fc, a)  # noqa: E731
    elif fam == "z-image":
        import comfy.ldm.lumina.model as M
        from comfy.ldm.flux.layers import EmbedND
        blk = init(M.JointTransformerBlock(0, C, 2, 2, 256, None, 1e-5, True, modulation=True, z_image_modulation=True,
                                           operation_settings={"operations": ops, "dtype": dtype, "device": device}))
        x0 = torch.randn(1, L, C, generator=g).to(device, dtype)
        a0 = torch.randn(1, min(C, 256), generator=g).to(device, dtype)
        pe = EmbedND(dim=128, theta=256, axes_dim=[32, 48, 48])(torch.randn(1, L, 3, generator=g).to(device) * 20).movedim(1, 2)
        pe = torch.cat([pe[:, :5], pe[:, 5:]], dim=1)
        call = lambda x, a: blk(x, None, pe, a)  # noqa: E731
    else:
        return False, {"why": f"family {fam}"}
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))  # noqa: E731
    outs, n0 = {}, (STATS.get("rmsmod", 0), STATS.get("rmsgres", 0))
    try:
        for tag in ("eager", "fused"):
            if tag == "fused":
                install_family_mod(fam, on=True)
            else:
                uninstall_family_mod(fam)
            x, a = x0.clone().requires_grad_(True), a0.clone().requires_grad_(True)
            o = call(x, a)
            o.backward(torch.linspace(-1, 1, o.numel(), device=o.device).view_as(o).to(o.dtype) * 0.01)
            outs[tag] = (o.detach(), x.grad, a.grad)
    finally:
        uninstall_family_mod(fam)
        comfy.model_management.in_training = prev_training
    r = {n: rel(f, e) for n, f, e in zip(("out", "dx", "dadaln"), outs["fused"], outs["eager"])}
    r["fused_ran"] = STATS.get("rmsmod", 0) - n0[0] >= 2 and STATS.get("rmsgres", 0) - n0[1] >= 2
    return all(v <= tol for k, v in r.items() if k != "fused_ran") and r["fused_ran"], r


# ----------------------------------------------------------------------------------------------------------------------
# qwen-image modulation (T213, 28 Oct; G56CT_FUSE_MOD on qwen-image): QwenImageTransformerBlock = flux's LayerNorm modulation
# (addcmul(shift, LN(x), 1 + scale): _LNMod exactly) + gated residuals as torch.addcmul(y, gate, x) (one rounding: _GateRes SINGLE)
# for the image stream and the text MLP, and encoder_hidden_states + gate * attn (two roundings: _GateRes) for the text attention.
# ----------------------------------------------------------------------------------------------------------------------
def gate_res1(x, y, gate):
    """torch.addcmul(x, gate, y) fused when possible (one rounding), else ComfyUI's own op."""
    if FLAGS.get("qmod") and _mod_ok(x, gate) is None and _mod_ok(y) is None and y.shape == x.shape:
        STATS["gres1"] = STATS.get("gres1", 0) + 1
        return _GateRes.apply(x, y, gate, True)
    STATS["gres1_eager"] = STATS.get("gres1_eager", 0) + 1
    return torch.addcmul(x, gate, y)


def _qln(norm, x, shift, scale):
    """qwen _modulate: addcmul(shift, norm(x), 1 + scale) (LayerNorm without affine) fused when possible."""
    if (FLAGS.get("qmod") and _mod_ok(x, shift, scale) is None and getattr(norm, "weight", None) is None
            and getattr(norm, "bias", None) is None and tuple(norm.normalized_shape) == (x.shape[2],)):
        STATS["qlnmod"] = STATS.get("qlnmod", 0) + 1
        return _LNMod.apply(x, shift, scale, float(norm.eps))
    STATS["qlnmod_eager"] = STATS.get("qlnmod_eager", 0) + 1
    return torch.addcmul(shift, norm(x), 1 + scale)


def _qgr2(x, y, gate):
    """x + gate * y (two roundings) fused when possible."""
    if FLAGS.get("qmod") and _mod_ok(x, gate) is None and _mod_ok(y) is None and y.shape == x.shape:
        STATS["gres"] = STATS.get("gres", 0) + 1
        return _GateRes.apply(x, y, gate, False)
    return x + gate * y


def _qwen_block_forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb=None,
                        timestep_zero_index=None, transformer_options={}):  # noqa: B006
    if timestep_zero_index is not None:  # the reference-latent split stays ComfyUI's
        return _FALLBACK["qwen_block"](self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb,
                                       image_rotary_emb=image_rotary_emb, timestep_zero_index=timestep_zero_index,
                                       transformer_options=transformer_options)
    img_mod_params = self.img_mod(temb)
    txt_mod_params = self.txt_mod(temb)
    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
    ish, isc, ig = (t.unsqueeze(1) for t in torch.chunk(img_mod1, 3, dim=-1))
    tsh, tsc, tg = (t.unsqueeze(1) for t in torch.chunk(txt_mod1, 3, dim=-1))
    img_attn_output, txt_attn_output = self.attn(hidden_states=_qln(self.img_norm1, hidden_states, ish, isc),
                                                 encoder_hidden_states=_qln(self.txt_norm1, encoder_hidden_states, tsh, tsc),
                                                 encoder_hidden_states_mask=encoder_hidden_states_mask, image_rotary_emb=image_rotary_emb,
                                                 transformer_options=transformer_options)
    hidden_states = gate_res1(hidden_states, img_attn_output, ig)  # _apply_gate: torch.addcmul(y, gate, x)
    encoder_hidden_states = _qgr2(encoder_hidden_states, txt_attn_output, tg)  # encoder_hidden_states + txt_gate1 * txt_attn_output
    ish2, isc2, ig2 = (t.unsqueeze(1) for t in torch.chunk(img_mod2, 3, dim=-1))
    hidden_states = gate_res1(hidden_states, self.img_mlp(_qln(self.img_norm2, hidden_states, ish2, isc2)), ig2)
    tsh2, tsc2, tg2 = (t.unsqueeze(1) for t in torch.chunk(txt_mod2, 3, dim=-1))
    encoder_hidden_states = gate_res1(encoder_hidden_states, self.txt_mlp(_qln(self.txt_norm2, encoder_hidden_states, tsh2, tsc2)), tg2)
    return encoder_hidden_states, hidden_states


def install_qwen_mod(on=True):
    import comfy.ldm.qwen_image.model as M
    FLAGS["qmod"] = bool(on)
    cls = M.QwenImageTransformerBlock
    if getattr(cls.forward, "_g56_fused", False):
        return 0
    _FALLBACK["qwen_block"] = cls.forward
    _qwen_block_forward._g56_fused = True
    cls.forward = _qwen_block_forward
    return 1


def uninstall_qwen_mod():
    import comfy.ldm.qwen_image.model as M
    cls = M.QwenImageTransformerBlock
    if "qwen_block" in _FALLBACK and getattr(cls.forward, "_g56_fused", False):
        cls.forward = _FALLBACK.pop("qwen_block")
    FLAGS["qmod"] = False


def qwen_block_selfcheck(device, dtype=torch.bfloat16, seed=0, tol=2e-2):
    """install_qwen_mod's block forward vs ComfyUI's QwenImageTransformerBlock (small random block, training mode): rel L2 of both
    outputs and the gradients of hidden_states, encoder_hidden_states and temb; the fused kernels must have run. -> (ok, report)"""
    import comfy.ops
    import comfy.model_management
    import comfy.ldm.qwen_image.model as M
    from comfy.ldm.flux.layers import EmbedND
    prev_training = comfy.model_management.in_training
    comfy.model_management.in_training = True
    g = torch.Generator(device="cpu").manual_seed(seed)
    blk = M.QwenImageTransformerBlock(256, 2, 128, eps=1e-6, dtype=dtype, device=device, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for n, p in blk.named_parameters():
            if p.dim() >= 2:
                p.copy_((torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5).to(p.dtype))
            elif "norm" in n:
                p.copy_((1.0 + 0.1 * torch.randn(p.shape, generator=g)).to(p.dtype))
            else:
                p.copy_((0.1 * torch.randn(p.shape, generator=g)).to(p.dtype))
    blk.requires_grad_(False)
    L, T = 12, 5
    h0 = torch.randn(1, L, 256, generator=g).to(device, dtype)
    e0 = torch.randn(1, T, 256, generator=g).to(device, dtype)
    t0 = torch.randn(1, 256, generator=g).to(device, dtype)
    pq = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])(torch.randn(1, L + T, 3, generator=g).to(device) * 20).to(dtype).contiguous()
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))  # noqa: E731
    outs, n0 = {}, (STATS.get("qlnmod", 0), STATS.get("gres1", 0), STATS.get("gres", 0))
    try:
        for tag in ("eager", "fused"):
            if tag == "fused":
                install_qwen_mod(True)
            else:
                uninstall_qwen_mod()
            h, e, t = (z.clone().requires_grad_(True) for z in (h0, e0, t0))
            eo, ho = blk(h, e, None, t, image_rotary_emb=pq)
            torch.autograd.backward([eo, ho], [torch.linspace(-1, 1, z.numel(), device=z.device).view_as(z).to(z.dtype) * 0.01
                                               for z in (eo, ho)])
            outs[tag] = (eo.detach(), ho.detach(), h.grad, e.grad, t.grad)
    finally:
        uninstall_qwen_mod()
        comfy.model_management.in_training = prev_training
    r = {n: rel(f, e_) for n, f, e_ in zip(("txt_out", "img_out", "dh", "de", "dtemb"), outs["fused"], outs["eager"])}
    r["fused_ran"] = (STATS.get("qlnmod", 0) - n0[0] >= 4 and STATS.get("gres1", 0) - n0[1] >= 3 and STATS.get("gres", 0) - n0[2] >= 1)
    return all(v <= tol for k, v in r.items() if k != "fused_ran") and r["fused_ran"], r
