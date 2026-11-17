#!/usr/bin/env python3
"""T063 RRF - readout refit (11 Oct 2026, sub-agent of the SN56 image session). An original post-training step: once the LoRA
is trained, solve in CLOSED FORM (ridge regression, lambda by K-fold cross-validation over the task's own TRAINING images) for
corrections of the model's final readout, so that the merged model's velocity predictions fit the task better. Nothing but the
task's own training images (fresh noises from our own RNG, the public sigma midpoints as the grid our trainer already uses; by
default the alternating half of the 16 midpoints per image, every band on half of the images) and
the base + LoRA the trainer already has; no evaluator noise / seed / rounding internals, no asset the trainer does not get.

Objective (the evaluator's metric on training data): per sample s = (image, sigma band, noise, caption | ""), x_t =
(1 - sigma) x0 + sigma eps, target v = eps - x0, loss = mean((v_pred - v)^2) over the latent, weighted 0.5 / 0.5 over caption /
"" and equally over images and cases. The model output is v_pred = P(y): y = the final Linear's per-token output, P the
unpatchify (a signed permutation onto the latent; z-image / ideogram4 negate). A forward hook makes y a float32 autograd leaf
(the same values), so P^T r (r = v - v_pred) comes exactly from autograd without knowing P; P^T u for a random +-1 latent u
gives the mask of the tokens that reach the output (text / padding tokens do not). Then, in token space,
    loss(dy) = sum_s w_s sum_t |g_t - dy_t|^2,   g = P^T r,
and every correction below is linear in its parameters -> weighted ridge regression on sufficient statistics.

Forms (the brief's order; availability per family from the ComfyUI 694815f final layers, see SPEC):
  B      output bias:  dy = db                                   -> '.diff_b' on the final Linear's bias (all families)
  BS<k>  + sigma-dependent shift through the final modulation's SHIFT rows, restricted to the top-k principal subspace of the
         modulation input c (= SiLU of the conditioning vector): dy_s = B phi_s, phi_s = V_k^T (c_s - c_mean); shipped as a
         rank-k LoRA on the modulation Linear (shift rows W_eff^+ B, other rows 0; down V_k^T) + the centring constant in db.
         flux (final_layer.adaLN_modulation.1: c = SiLU(time + guidance + pooled-text vec)) and qwen-image (norm_out.linear:
         c = SiLU(temb(t))) only - z-image / ideogram4 modulate the scale only, krea2's last modulation has no Linear.
  W      full ridge refit of the final Linear: dy_t = dW x_t + db (dW: O x H) -> '.diff' (no LoRA on that weight), or an
         exact rank concatenation [up | I_O], [down; dW], alpha = rank (z-image final_layer.linear, krea2 last.linear carry
         C0 LoRAs; never a LoRA and a '.diff' on one weight: comfy/lora.py:53-76 lets the '.diff' override silently).
         Only where the final Linear merges deterministically (bf16 / fp16 / fp32, round to nearest): NOT ideogram4 (fp8
         weight, stochastic rounding at the merge).
  WS<k>  W + BS<k>.
Model / lambda choice: K-fold CV over the training IMAGES (folds by image), relative ridge (lambda x the block's mean
diagonal), one-standard-error rule (the simplest model / largest lambda within one SE of the best), ship only when the chosen
configuration's mean CV gain is positive and above its SE; otherwise nothing is shipped (a no-op). A realized-effect gate
then merges the candidate file through ComfyUI's own LoRA loader + merge (fp16 compute, round to the weight dtype) and keeps
it only if >= half of its intended in-sample gain survives the rounding (else bias-only, else nothing).


  python rrf_box.py --family F --repo ORG/REPO --task TASK_ID --dataset ZIP --c0 LORA --out /cache/rrf [--json J]
Writes <out>/R63__<task8>.safetensors (C0 + the shipped correction, only when shipped), <out>/R63B__<task8>.safetensors
(C0 + the bias-only fit at its CV lambda: diagnostic, when it differs from R63), <out>/rrf_<task8>.json, and appends the
labels of the written files to <out>/ready.txt. Every check prints '[rrf] PASS|FAIL ...'; the last line is
'[rrf] RESULT PASS|FAIL (n/m checks) ...'.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

MUT = os.environ.get("RRF_MUTATE", "")  # rrf_check.py mutation switch (a deliberately broken variant); empty in real runs

# Final-layer anatomy per family (ComfyUI 694815f):
#   flux       comfy/ldm/flux/layers.py:358-372 LastLayer: shift, scale = adaLN_modulation(vec).chunk(2); x = norm(x) * (1 + scale)
#              + shift; linear(x). vec = time_in + guidance_in + vector_in(pooled text) (flux/model.py:170-178), output
#              rearranged (model.py:420), no sign flip.
#   qwen-image comfy/ldm/qwen_image/model.py:319-338 LastLayer: emb = linear(SiLU(temb)); scale, shift = chunk(2);
#              x = shift + norm(x) * (1 + scale); proj_out (model.py:563-568), no sign flip.
#   z-image    comfy/ldm/lumina/model.py:354-396 FinalLayer: scale only; output = -unpatchify (model.py:857-859).
#   ideogram4  comfy/ldm/ideogram4/model.py:118-127: scale only; output = -tokens_to_img (model.py:295-297).
#   krea2      comfy/ldm/krea2/model.py:168-178 LastLayer: scale, shift = t + lin (SimpleModulation, no Linear); last.linear.
SPEC = {
    "flux": dict(linear="final_layer.linear", mod="final_layer.adaLN_modulation.1", chunks=2, shift=0),
    "qwen-image": dict(linear="proj_out", mod="norm_out.linear", chunks=2, shift=1),
    "z-image": dict(linear="final_layer.linear", mod=None),
    "ideogram4": dict(linear="final_layer.linear", mod=None),
    "krea2": dict(linear="last.linear", mod=None),
}
PREFIX = "diffusion_model."
ORDER = ("B", "BS4", "BS8", "W", "WS8")  # complexity order (one-SE rule picks the earliest eligible)
LAMBDAS = (1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0)
RES = {"checks": [], "numbers": {}}


SAY = None  # the trainer (G56CT_RRF) routes these lines to its private log: the container stdout is public


def say(*a):
    if SAY is not None:
        SAY(*a)
        return
    print("[rrf]", *a, flush=True)


def ok(name, cond, detail=""):
    RES["checks"].append((name, bool(cond), detail))
    say(f"{'PASS' if cond else 'FAIL'} {name}" + (f" - {detail}" if detail else ""))
    return bool(cond)


def get_attr(obj, path):
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


# ------------------------------------------------------------------------------------------------------------ noise / folds
def rrf_noise(shape, seed, sha, band, k, stream):
    """Our own fresh noise: a generator seeded by sha256('rrf63|stream|seed|image sha|band|k') - never the evaluator's
    case_seed scheme (validator/evaluation/denoising_mse.py), never its noises."""
    import torch
    if MUT == "eval_seeds":  # (mutation only: the evaluator's own case_seed scheme - must be caught by rrf_check)
        from evalcore import case_seed
        return torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(case_seed(sha, band, k)),
                           dtype=torch.float32)
    h = hashlib.sha256(f"rrf63|{stream}|{seed}|{sha}|{band}|{k}".encode()).digest()
    g = torch.Generator(device="cpu").manual_seed(int.from_bytes(h[:8], "little") & 0x7FFFFFFFFFFFFFFF)
    return torch.randn(shape, generator=g, dtype=torch.float32)


def fold_of(shas, k):
    """image sha -> fold (balanced, by a hash order of the image digests)."""
    order = sorted(shas, key=lambda s: hashlib.sha256(f"rrf63fold|{s}".encode()).hexdigest())
    return {s: i % k for i, s in enumerate(order)}


# ------------------------------------------------------------------------------------------------------------------ hooks
class Tap:
    """Forward hook on the final Linear (input x, output y -> a float32 autograd leaf with the SAME values, so the tail
    unpatchify / sign / crop is recorded and P^T comes from autograd) and a pre-hook on the modulation Linear (its input c)."""

    def __init__(self, lin, mod=None):
        self.on, self.calls, self.x, self.y, self.c = False, 0, None, None, None
        self.h = [lin.register_forward_hook(self._out)]
        if mod is not None:
            self.h.append(mod.register_forward_pre_hook(self._cin))

    def _out(self, m, inp, out):
        if not self.on:
            return None
        self.calls += 1
        self.x = inp[0].detach()
        y = out.detach().float().requires_grad_(True)
        self.y = y
        return y

    def _cin(self, m, inp):
        if self.on:
            self.c = inp[0].detach()

    def remove(self):
        for h in self.h:
            h.remove()


def token_residual(tap, den, noisy, target, sigmas):
    """(g, tok, loss_s, err): g = P^T r per token (B, T, O) float32, tok = tokens that reach the output (B, T) bool,
    loss_s = per-sample evaluator MSE (B,), err = max relative |sum g^2 - sum r^2| (P orthogonal on used tokens)."""
    import torch
    sig = sigmas.reshape(-1, *([1] * (noisy.dim() - 1))).to(noisy.device, torch.float32)
    v = (noisy.float() - den.float()) / sig  # the evaluator's flow_prediction_mse velocity
    r = target.float() - v
    loss = 0.5 * (r * r).sum()
    (gy,) = torch.autograd.grad(loss, tap.y, retain_graph=True)
    g = -gy if MUT != "residual_sign" else gy  # dL/dy = -P^T r
    u = (torch.randint(0, 2, v.shape, generator=torch.Generator(device="cpu").manual_seed(631), dtype=torch.int64) * 2 - 1)
    (gu,) = torch.autograd.grad((v * u.to(v.device, v.dtype)).sum(), tap.y)
    a = gu.abs()
    el = a > 0.5
    tok = el.all(-1)
    partial = int((el.any(-1) & ~el.all(-1)).sum())
    dev1 = float((a[el] - 1.0).abs().max()) if bool(el.any()) else 1.0
    if partial or dev1 > 1e-3 or int(el.sum()) != v[0].numel() * v.shape[0]:
        raise RuntimeError(f"the output map is not a signed permutation of whole tokens onto the latent (partial tokens "
                           f"{partial}, |P| deviation {dev1:.1e}, mapped {int(el.sum())} of {v.numel()})")
    if MUT == "no_mask":
        tok = torch.ones_like(tok)
    rd = r.detach()
    loss_s = (rd * rd).flatten(1).mean(1)
    gg = (g * g).sum(-1)
    s_g = torch.stack([gg[b][tok[b]].sum() for b in range(g.shape[0])])
    s_r = (rd * rd).flatten(1).sum(1)
    err = float(((s_g - s_r).abs() / s_r.clamp_min(1e-30)).max())
    return g.detach(), tok, loss_s, err


# -------------------------------------------------------------------------------------------------------------- statistics
class Stats:
    """Weighted sufficient statistics of g (O) and x (H) over used tokens: per fold (x x^T, x g^T, sums) and per sample
    (S = sum g, N, Q = sum |g|^2, xs = sum x, c = modulation input). float64, on the given device."""

    def __init__(self, H, O, nf, device, full):
        import torch
        f = dict(dtype=torch.float64, device=device)
        self.H, self.O, self.nf, self.full, self.device = H, O, nf, full, device
        self.Axx = torch.zeros(nf, H, H, **f) if full else None
        self.Axg = torch.zeros(nf, H, O, **f) if full else None
        self.sx = torch.zeros(nf, H, **f)
        self.sg = torch.zeros(nf, O, **f)
        self.n = torch.zeros(nf, **f)
        self.q = torch.zeros(nf, **f)
        self.samples = []

    def add(self, fold, w, x, g, c, meta):
        """x (n, H), g (n, O): float64 rows of the used tokens of ONE sample; c (Hc,) or None; w its weight."""
        S, xs = g.sum(0), x.sum(0)
        Q = float((g * g).sum())
        if self.full:
            self.Axx[fold] += w * (x.T @ x)
            self.Axg[fold] += w * (x.T @ g)
        self.sx[fold] += w * xs
        self.sg[fold] += w * S
        self.n[fold] += w * x.shape[0]
        self.q[fold] += w * Q
        self.samples.append(dict(meta, fold=fold, w=w, N=x.shape[0], S=S.cpu(), Q=Q, xs=xs.cpu(),
                                 c=None if c is None else c.double().cpu()))


def phi_basis(st, k):
    """V (Hc x k) top-k principal directions of the centred modulation inputs (inputs only, no targets) and their mean."""
    import torch
    C = torch.stack([s["c"] for s in st.samples])
    cbar = C.mean(0)
    _, sv, Vh = torch.linalg.svd(C - cbar, full_matrices=False)
    V = Vh[:k].T.contiguous()
    for j in range(V.shape[1]):  # sign fixed: largest |entry| positive
        if V[V[:, j].abs().argmax(), j] < 0:
            V[:, j] = -V[:, j]
    return V, cbar, sv


def parse_model(m):
    return m.startswith("W"), int(m.split("S")[1]) if "S" in m else 0


def design(st, model, folds, V=None, cbar=None):
    """(A, Bm, q): normal matrix (D x D), right side (D x O) and sum w|g|^2 of the samples of `folds`; features of `model`
    in the order [x (H)], 1, [phi (k)]."""
    import torch
    hx, k = parse_model(model)
    H, O, dev = st.H, st.O, st.device
    D = (H if hx else 0) + 1 + k
    A = torch.zeros(D, D, dtype=torch.float64, device=dev)
    Bm = torch.zeros(D, O, dtype=torch.float64, device=dev)
    fl = list(folds)
    i1 = H if hx else 0
    if hx:
        A[:H, :H] = st.Axx[fl].sum(0)
        Bm[:H] = st.Axg[fl].sum(0)
        A[:H, i1] = A[i1, :H] = st.sx[fl].sum(0)
    A[i1, i1] = st.n[fl].sum()
    Bm[i1] = st.sg[fl].sum(0)
    if k:
        fs = set(fl)
        ss = [s for s in st.samples if s["fold"] in fs]
        P = torch.stack([(s["c"] - cbar) for s in ss]).to(dev) @ V[:, :k].to(dev)  # (n_s, k)
        w = torch.tensor([s["w"] for s in ss], dtype=torch.float64, device=dev)
        N = torch.tensor([float(s["N"]) for s in ss], dtype=torch.float64, device=dev)
        S = torch.stack([s["S"] for s in ss]).to(dev)
        ip = i1 + 1
        A[ip:, ip:] = (P * (w * N)[:, None]).T @ P
        A[i1, ip:] = A[ip:, i1] = (w * N) @ P
        Bm[ip:] = (P * w[:, None]).T @ S
        if hx:
            XS = torch.stack([s["xs"] for s in ss]).to(dev)
            A[:H, ip:] = (XS * w[:, None]).T @ P
            A[ip:, :H] = A[:H, ip:].T
    return A, Bm, float(st.q[fl].sum())


def ridge(A, model, lam, H):
    """lambda x the mean diagonal of each feature block (x / bias / phi)."""
    import torch
    hx, k = parse_model(model)
    d = torch.diagonal(A)
    reg = torch.zeros_like(d)
    i1 = H if hx else 0
    if hx:
        reg[:H] = lam * d[:H].mean()
    reg[i1] = lam * d[i1]
    if k:
        reg[i1 + 1:] = lam * d[i1 + 1:].mean()
    return A + torch.diag(reg)


def solve(A, Bm, model, lam, H):
    import torch
    return torch.linalg.solve(ridge(A, model, lam, H), Bm)


def gain(theta, A, Bm):
    """reduction of sum w|g - dy|^2 (positive = better)."""
    return float(2.0 * (theta * Bm).sum() - (theta * (A @ theta)).sum())


def cv_table(st, models, lambdas, V, cbar):
    """per (model, lambda): per-fold relative gains (gain / fold loss), mean, SE, pooled relative gain."""
    import torch  # noqa: F401
    nf = st.nf
    tab = {}
    for m in models:
        parts = [design(st, m, [f], V, cbar) for f in range(nf)]
        A_all = sum(p[0] for p in parts)
        B_all = sum(p[1] for p in parts)
        for lam in lambdas:
            rels, gsum, qsum = [], 0.0, 0.0
            for f, (Af, Bf, qf) in enumerate(parts):
                th = solve(A_all - Af, B_all - Bf, m, lam, st.H)
                gf = gain(th, Af, Bf)
                rels.append(gf / qf if qf > 0 else 0.0)
                gsum, qsum = gsum + gf, qsum + qf
            mean = sum(rels) / len(rels)
            se = (sum((x - mean) ** 2 for x in rels) / max(len(rels) - 1, 1) / len(rels)) ** 0.5
            tab[(m, lam)] = dict(model=m, lam=lam, folds=rels, mean=mean, se=se, pooled=gsum / max(qsum, 1e-300),
                                 pos=sum(1 for x in rels if x > 0))
    return tab


def choose(tab):
    """one-standard-error rule: the simplest model (ORDER), then the largest lambda, whose mean CV gain is within one SE
    of the best; shipped only when its mean CV gain is > 0 and > its SE."""
    best = max(tab.values(), key=lambda r: r["mean"])
    thr = best["mean"] - best["se"]
    elig = [r for r in tab.values() if r["mean"] >= thr]
    pick = min(elig, key=lambda r: (ORDER.index(r["model"]), -r["lam"]))
    if MUT == "no_one_se":
        pick = best
    ship = pick["mean"] > 0 and pick["mean"] > pick["se"]
    if MUT == "always_ship":
        ship = True
    return pick, best, ship


def theta_to_delta(theta, model, st, V, cbar, W_eff):
    """features -> parameter deltas: W (O x H), b (O), mod_shift_up (H x k) + mod_down (k x Hc) (float64)."""
    import torch
    hx, k = parse_model(model)
    H = st.H
    d = {}
    i = 0
    if hx:
        d["W"] = theta[:H].T.contiguous()
        i = H
    b = theta[i].clone()
    if k:
        Bs = theta[i + 1:i + 1 + k].T  # O x k
        Vk = V[:, :k].to(theta.device)
        b = b - Bs @ (Vk.T @ cbar.to(theta.device))
        We = W_eff + d["W"] if hx else W_eff
        Wp = torch.linalg.pinv(We)  # H x O, W_eff @ Wp = I
        if MUT == "pinv_transpose":
            Wp = We.T.contiguous()
        d["mod_shift_up"] = Wp @ Bs
        d["mod_down"] = Vk.T.contiguous()
        d["pinv_err"] = float((We @ Wp - torch.eye(We.shape[0], dtype=We.dtype, device=We.device)).abs().max())
    d["b"] = b
    return d


# ------------------------------------------------------------------------------------------------------------------ export
def merge_rrf(c0, fam, d, mod_rows=None):
    """C0 state dict + RRF deltas -> the file's state dict (C0 tensors shared and never modified) + a report. Never a LoRA
    and a '.diff' on one weight: a weight with a C0 LoRA gets an exact rank concatenation, a weight with a '.diff' gets the
    sum, any other a new '.diff' (full) or a new LoRA (low-rank)."""
    import torch
    sp = SPEC[fam]
    out = dict(c0)
    rep = {}
    pl = PREFIX + sp["linear"]
    if d.get("b") is not None:
        key = pl + ".diff_b"
        add = d["b"].detach().double().cpu()
        out[key] = ((out[key].double() + add) if key in out else add).float().contiguous()
        rep["diff_b"] = key
    if d.get("W") is not None:
        rep["W"] = _add_dense(out, pl, d["W"].detach().double().cpu())
    if d.get("mod_shift_up") is not None:
        up_s = d["mod_shift_up"].detach().double().cpu()
        H = up_s.shape[0]
        up = torch.zeros(sp["chunks"] * H, up_s.shape[1], dtype=torch.float64)
        row = sp["shift"] if MUT != "scale_rows" else 1 - sp["shift"]
        up[row * H:(row + 1) * H] = up_s
        rep["mod"] = _add_lowrank(out, PREFIX + sp["mod"], up, d["mod_down"].detach().double().cpu())
    return out, rep


def _add_dense(out, prefix, dW):
    import torch
    ku, kd, ka, kf = prefix + ".lora_up.weight", prefix + ".lora_down.weight", prefix + ".alpha", prefix + ".diff"
    if ku in out and MUT != "diff_beside_lora":
        up, down = out[ku], out[kd]
        r = down.shape[0]
        a = float(out[ka]) if ka in out else float(r)
        s = a / r
        O = up.shape[0]
        eye = torch.eye(O, dtype=torch.float64) / s
        out[ku] = torch.cat([up, eye.to(up.dtype)], 1).contiguous()
        out[kd] = torch.cat([down, dW.to(down.dtype)], 0).contiguous()
        out[ka] = torch.tensor(s * (r + O), dtype=torch.float32)
        return dict(kind="concat", key=prefix, rank=f"{r}+{O}", scale=s)
    if kf in out:
        out[kf] = (out[kf].double() + dW).float().contiguous()
        return dict(kind="diff+", key=kf)
    out[kf] = dW.float().contiguous()
    return dict(kind="diff", key=kf)


def _add_lowrank(out, prefix, up, down):
    import torch
    ku, kd, ka, kf = prefix + ".lora_up.weight", prefix + ".lora_down.weight", prefix + ".alpha", prefix + ".diff"
    k = down.shape[0]
    if ku in out:
        u0, d0 = out[ku], out[kd]
        r = d0.shape[0]
        a = float(out[ka]) if ka in out else float(r)
        s = a / r
        out[ku] = torch.cat([u0, (up / s).to(u0.dtype)], 1).contiguous()
        out[kd] = torch.cat([d0, down.to(d0.dtype)], 0).contiguous()
        out[ka] = torch.tensor(s * (r + k), dtype=torch.float32)
        return dict(kind="concat", key=prefix, rank=f"{r}+{k}")
    if kf in out:
        out[kf] = (out[kf].double() + up @ down).float().contiguous()
        return dict(kind="diff+", key=kf)
    out[ku] = up.float().contiguous()
    out[kd] = down.float().contiguous()
    out[ka] = torch.tensor(float(k), dtype=torch.float32)
    return dict(kind="lora", key=prefix, rank=k)


def lora_product(sd, prefix):
    """float64 delta of one weight in a state dict: alpha/rank x up @ down (+ '.diff'); None when absent."""
    import torch
    ku, kd, ka, kf = prefix + ".lora_up.weight", prefix + ".lora_down.weight", prefix + ".alpha", prefix + ".diff"
    out = None
    if ku in sd:
        up, down = sd[ku].double(), sd[kd].double()
        a = float(sd[ka]) if ka in sd else float(down.shape[0])
        out = (up @ down) * (a / down.shape[0])
    if kf in sd:
        out = sd[kf].double() if out is None else out + sd[kf].double()
    return out


def check_merge(c0, new, fam, d, tag):
    """the written state = C0 + the intended deltas: untouched C0 tensors byte-identical, concatenated rows byte-identical,
    per weight the product delta == intended (float64, rel 1e-6), no LoRA + '.diff' on one weight, alpha = scale x rank."""
    import torch
    sp = SPEC[fam]
    touched = set()
    pl = PREFIX + sp["linear"]
    pm = PREFIX + sp["mod"] if sp["mod"] else None
    for p in (pl, pm):
        if p:
            touched |= {p + s for s in (".lora_up.weight", ".lora_down.weight", ".alpha", ".diff", ".diff_b")}
    same = all(k in new and new[k].dtype == v.dtype and torch.equal(new[k], v) for k, v in c0.items() if k not in touched)
    errs = []
    rows_ok = True
    for p in (pl, pm):
        if not p:
            continue
        ku = p + ".lora_up.weight"
        if ku in c0 and ku in new:
            r = c0[p + ".lora_down.weight"].shape[0]
            rows_ok &= torch.equal(new[ku][:, :r], c0[ku]) and torch.equal(new[p + ".lora_down.weight"][:r], c0[p + ".lora_down.weight"])
        want = None
        if p == pl and d.get("W") is not None:
            want = d["W"].double().cpu()
        if p == pm and d.get("mod_shift_up") is not None:
            up = torch.zeros(sp["chunks"] * d["mod_shift_up"].shape[0], d["mod_shift_up"].shape[1], dtype=torch.float64)
            H = d["mod_shift_up"].shape[0]
            up[sp["shift"] * H:(sp["shift"] + 1) * H] = d["mod_shift_up"].double().cpu()
            want = up @ d["mod_down"].double().cpu()
        p0, p1 = lora_product(c0, p), lora_product(new, p)
        if want is not None:
            got = p1 - (p0 if p0 is not None else 0.0)
            errs.append(float((got - want).norm() / max(float(want.norm()), 1e-30)))
        elif p1 is not None or p0 is not None:
            errs.append(0.0 if (p0 is not None and p1 is not None and torch.equal(p0, p1)) else 1.0)
    kb = pl + ".diff_b"
    b_ok = True
    if d.get("b") is not None:
        want_b = d["b"].double().cpu() + (c0[kb].double() if kb in c0 else 0.0)
        b_ok = kb in new and float((new[kb].double() - want_b).abs().max()) <= 1e-6 * max(float(want_b.abs().max()), 1e-12)
    coll = [k for k in new if k.endswith(".diff") and k[:-5] + ".lora_up.weight" in new]
    emax = max(errs) if errs else 0.0
    return ok(f"merge {tag}: C0 tensors byte-identical, C0 rows of concatenated modules byte-identical, product delta == "
              f"intended (rel <= 1e-6), diff_b exact, no LoRA + '.diff' on one weight",
              same and rows_ok and emax <= 1e-6 and b_ok and not coll,
              f"product rel err max {emax:.1e}, collisions {coll[:2]}, untouched same {same}, rows {rows_ok}, bias {b_ok}")


def strict_load(mp, sd):
    """the evaluator's LoraLoader strictness (comfy.sd.load_lora_for_models + validator image_flow_adapter.apply_lora):
    convert_lora -> load_lora with 'lora key not loaded:' captured -> every patch key in the model ('NOT LOADED')."""
    import comfy.lora
    import comfy.lora_convert
    key_map = comfy.lora.model_lora_keys_unet(mp.model, {})
    msgs = []

    class Hd(logging.Handler):
        def emit(self, r):
            msgs.append(r.getMessage())
    h = Hd()
    logging.getLogger().addHandler(h)
    try:
        loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(dict(sd)), key_map)
    finally:
        logging.getLogger().removeHandler(h)
    msd = set(mp.model.state_dict().keys())
    missing = [m for m in msgs if m.startswith(("lora key not loaded:", "NOT LOADED "))]
    missing += [f"NOT LOADED {k}" for k in loaded if (k if isinstance(k, str) else k[0]) not in msd]
    return loaded, missing


def expected_patches(sd):
    mods = {k[: -len(".lora_up.weight")] for k in sd if k.endswith(".lora_up.weight")}
    return len(mods) + sum(1 for k in sd if k.endswith((".diff", ".diff_b")))


def merge_kind(model, key):
    """'det' when ComfyUI merges this parameter by a plain cast (bf16 / fp16 / fp32: round to nearest), 'sr' when by
    stochastic rounding (fp8 weights, quantized layers' set_weight) - model_patcher.py:846-875, float.py:65-84, ops.py:1307."""
    import torch
    import comfy.model_patcher
    w, set_func, _ = comfy.model_patcher.get_key_weight(model, key)
    op = get_attr(model, key.rsplit(".", 1)[0])
    if set_func is not None and getattr(op, "layout_type", None) is not None:
        return "sr"
    if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or type(w).__name__ == "QuantizedTensor":
        return "sr"
    return "det"


def realized_weight(mp, key, patch, base, device):
    """the evaluator's merged value of one parameter for the patch list `patch` (from comfy.lora.load_lora of the file):
    ModelPatcher.patch_weight_to_device(return_weight=True) on the BASE value (the live parameter is swapped for the
    base copy during the call and put back). float32 result on device."""
    import torch
    m = get_attr(mp.model, key.rsplit(".", 1)[0])
    pname = key.rsplit(".", 1)[1]
    tmp = mp.clone()
    tmp.patches = {}
    tmp.add_patches({key: patch}, 1.0, 1.0)
    live = m._parameters[pname]
    m._parameters[pname] = torch.nn.Parameter(base.to(device=live.device), requires_grad=False)
    try:
        w = tmp.patch_weight_to_device(key, device_to=device, return_weight=True)
    finally:
        m._parameters[pname] = live
    if hasattr(w, "dequantize"):
        w = w.dequantize()
    return w.float()


def realized_gain(st, dW, db, M, W_new):
    """exact change (positive = better) of sum w|g - dy|^2 for dy_t = dW x_t + db + W_new (M c_s): from the statistics.
    dW (O x H) needs the x x^T statistics (st.full); M (H x Hc) is the change of the modulation's shift rows."""
    import torch
    dev = st.device
    g = 0.0
    if dW is not None:
        if not st.full:
            raise ValueError("a weight change needs the full statistics")
        dWd = dW.to(dev, torch.float64)
        Axx, Axg = st.Axx.sum(0), st.Axg.sum(0)
        g += 2.0 * float((dWd.T * Axg).sum()) - float(((dWd @ Axx) * dWd).sum())
    ss = st.samples
    w = torch.tensor([s["w"] for s in ss], dtype=torch.float64)
    N = torch.tensor([float(s["N"]) for s in ss], dtype=torch.float64)
    S = torch.stack([s["S"] for s in ss])
    e = torch.zeros(len(ss), st.O, dtype=torch.float64)
    if db is not None:
        e = e + db.double().cpu()[None, :]
    if M is not None:
        C = torch.stack([s["c"] for s in ss])
        e = e + (C @ M.double().cpu().T) @ W_new.double().cpu().T
    g += float((w * (2.0 * (S * e).sum(1) - N * (e * e).sum(1))).sum())
    if dW is not None:
        XS = torch.stack([s["xs"] for s in ss])
        g -= float((w * 2.0 * (e * (XS @ dW.double().cpu().T)).sum(1)).sum())
    return g


# ---------------------------------------------------------------------------------------------------------- the collection
def collect(fwd, tap, items, empty_idx, bands, n_noise, seed, stream, st, folds, weight_of, device, batch, fold_override=None,
            log_every=0, target_fn=None, deadline=None, band_stride=1):
    """one pass over items x bands x {caption, ""} x n_noise fresh noises: the residual statistics of the loaded model.
    fwd(noisy, sigmas, cond_idx) -> denoised. target = eps - x0 (the evaluator's flow target); target_fn(noisy, sigmas,
    cond_idx, eps, x0) replaces it in rrf_check.py (a teacher model's velocity: the planted-correction tests).
    band_stride s > 1: image i gets the bands b with (b + i) % s == 0 (16 / s bands per image, every band on 1 / s of the
    images: the production form of G56CT_RRF, half or less of the forwards).
    Returns per-sample records (meta + loss) and the max sum-identity error."""
    import torch
    errs, out = 0.0, []
    t0 = time.time()
    n_done = 0
    for ii, it in enumerate(items):
        cases = [(b, k) for b in bands if (b + ii) % band_stride == 0 for k in range(n_noise)]
        x0 = it["scaled"]
        for mode, idx in (("text", it["cond_idx"]), ("no_text", empty_idx)):
            for j in range(0, len(cases), batch):
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError(f"rrf: the deadline came after {n_done} samples")
                cb = cases[j:j + batch]
                sig = torch.tensor([(b + 0.5) / 16.0 for b, _ in cb], dtype=torch.float32)
                eps = torch.cat([rrf_noise(x0.shape, seed, it["sha"], b, k, stream) for b, k in cb])
                x0b = x0.expand(len(cb), *x0.shape[1:]).float()
                sv = sig.reshape(-1, *([1] * (x0.dim() - 1)))
                noisy = ((1 - sv) * x0b + sv * eps).to(device)
                target = (eps - x0b).to(device) if target_fn is None else target_fn(noisy, sig.to(device), idx, eps, x0b)
                tap.on, tap.calls, tap.x, tap.y, tap.c = True, 0, None, None, None
                with torch.enable_grad():
                    den = fwd(noisy, sig.to(device), idx)
                    tap.on = False
                    if tap.calls != 1:
                        raise RuntimeError(f"final Linear called {tap.calls} times in one forward")
                    g, tok, loss_s, err = token_residual(tap, den, noisy, target, sig)
                errs = max(errs, err)
                fold = fold_override if fold_override is not None else folds[it["sha"]]
                C = tap.c.reshape(len(cb), -1) if tap.c is not None else None
                for bi, (b, k) in enumerate(cb):
                    xb = tap.x[bi][tok[bi]].double()
                    gb = g[bi][tok[bi]].double()
                    if MUT == "xg_rows_shift":
                        gb = torch.roll(gb, 1, 0)
                    w = weight_of(it, mode, xb.shape[0])
                    meta = dict(img=ii, band=b, k=k, mode=mode, loss=float(loss_s[bi]))
                    st.add(fold, w, xb, gb, None if C is None else C[bi], meta)
                    out.append(meta)
                n_done += len(cb)
                del den, g, tok, tap.x, tap.y
                tap.x = tap.y = None
        if log_every and (ii + 1) % log_every == 0:
            say(f"  {stream}: {ii + 1}/{len(items)} images, {n_done} samples, {time.time() - t0:.0f}s")
    return out, errs


def weighted_loss(records, n_img, n_cases):
    """the evaluator's aggregation: per image mean over cases per mode, 0.5 x text + 0.5 x no_text, mean over images."""
    per = {}
    for r in records:
        per.setdefault((r["img"], r["mode"]), []).append(r["loss"])
    imgs = sorted({i for i, _ in per})
    tot = 0.0
    for i in imgs:
        tot += 0.5 * sum(per[(i, "text")]) / len(per[(i, "text")]) + 0.5 * sum(per[(i, "no_text")]) / len(per[(i, "no_text")])
    return tot / max(len(imgs), 1)


# ------------------------------------------------------------------------------------------------------------- the fit
def fit(st, fam, has_mod, lin_det, mod_det, W_c0, k_max=8, lambdas=LAMBDAS):
    """CV over the available forms; returns (table, pick, best, ship, deltas of pick, deltas of the best bias-only, basis)."""
    models = ["B"]
    V = cbar = None
    sv = None
    if has_mod and mod_det and MUT != "no_mod_form":
        V, cbar, sv = phi_basis(st, k_max)
        V = V.to(st.device)
        models += ["BS4", "BS8"]
    if lin_det and st.full:
        models += ["W"]
        if has_mod and mod_det and V is not None:
            models += ["WS8"]
    tab = cv_table(st, models, lambdas, V, cbar)
    pick, best, ship = choose(tab)
    bias_best = max((r for r in tab.values() if r["model"] == "B"), key=lambda r: r["mean"])

    def deltas(r):
        A, Bm, _ = design(st, r["model"], range(st.nf), V, cbar)
        th = solve(A, Bm, r["model"], r["lam"], st.H)
        dd = theta_to_delta(th, r["model"], st, V, cbar, W_c0)
        dd["insample_gain"] = gain(th, A, Bm)
        return dd
    return dict(table=tab, pick=pick, best=best, ship=ship, d_pick=deltas(pick), bias=bias_best, d_bias=deltas(bias_best),
                V=V, cbar=cbar, sv=None if sv is None else [float(x) for x in sv[:12]], models=models)


# ------------------------------------------------------------------------------------------------------------- the run
def final_keys(fam):
    sp = SPEC[fam]
    return [PREFIX + sp["linear"] + ".weight", PREFIX + sp["linear"] + ".bias"] + \
        ([PREFIX + sp["mod"] + ".weight"] if sp["mod"] else [])


def base_values(model, keys):
    """the BASE values of the final-layer parameters (before any patch): realized_weight merges onto these."""
    vals = {}
    for k in keys:
        try:
            vals[k] = get_attr(model, k).detach().clone()
        except Exception as e:  # noqa: BLE001  (a quantized weight RRF never changes)
            say(f"base value of {k} not kept ({type(e).__name__}); RRF never changes it")
    return vals


def rrf_run(ctx):
    """the RRF on a loaded model (C0 merged by ComfyUI): hook identity, collection, CV fit, candidate files (merge check,
    strict load, realized effect + gate), writes, EFFECT on fresh validation noises. ctx: fam, mp (the loaded C0 patcher),
    fwd(noisy, sigmas, cond_idx) -> denoised, items (scaled, sha, cond_idx), empty_idx, base_vals, c0, out, tag8, seed,
    noises, nf, batch, val_ids, device; optional target_fn, no_full, lambdas, deadline, mode ('trainer': no file is
    written, no diagnostic arm, no EFFECT pass - returns {'R63': candidate with its state dict 'sd'} or {}).
    Returns {label: info} of the written files."""
    import torch
    from safetensors.torch import save_file
    fam, mp_c0, fwd, items, dev = ctx["fam"], ctx["mp"], ctx["fwd"], ctx["items"], ctx["device"]
    sp, keys, tag8 = SPEC[fam], final_keys(fam), ctx["tag8"]
    out = Path(ctx["out"]) if ctx.get("out") else None
    tm = RES["numbers"].setdefault("seconds", {})
    dm = mp_c0.model.diffusion_model
    lin = get_attr(dm, sp["linear"])
    mod = get_attr(dm, sp["mod"]) if sp["mod"] else None
    lin_det = merge_kind(mp_c0.model, keys[0]) == "det" and merge_kind(mp_c0.model, keys[1]) == "det"
    mod_det = mod is not None and merge_kind(mp_c0.model, keys[2]) == "det"
    H, O = lin.weight.shape[1], lin.weight.shape[0]
    say(f"final Linear {sp['linear']} {tuple(lin.weight.shape)} {lin.weight.dtype} merge {'det' if lin_det else 'sr'}; "
        f"modulation {sp['mod']} {('det' if mod_det else 'sr') if mod is not None else '-'}")
    W_c0 = (lin.weight.dequantize() if hasattr(lin.weight, "dequantize") else lin.weight).detach().double()
    tap = Tap(lin, mod if mod_det else None)
    seed, tfn, trainer, dl = ctx["seed"], ctx.get("target_fn"), ctx.get("mode") == "trainer", ctx.get("deadline")
    try:
        # hook identity: the hooked forward (float32 tail) returns the very same denoised values as the plain one
        it0 = items[0]
        eps = rrf_noise(it0["scaled"].shape, seed, it0["sha"], 8, 0, "id")
        nz = (0.5 * it0["scaled"] + 0.5 * eps).to(dev)
        sig = torch.tensor([0.5], dtype=torch.float32, device=dev)
        with torch.no_grad():
            d_plain = fwd(nz, sig, it0["cond_idx"]).float()
            tap.on = True
            d_hook = fwd(nz, sig, it0["cond_idx"]).float()
            tap.on = False
        ok("hooked forward == plain forward (bit for bit)", torch.equal(d_plain, d_hook),
           f"max |diff| {float((d_plain - d_hook).abs().max()):.2e}")
        bands = list(range(16))
        bpi = int(ctx.get("bands_per_image", 16))
        if bpi not in (1, 2, 4, 8, 16):
            raise ValueError(f"bands_per_image {bpi} must divide 16")
        n_cases = bpi * ctx["noises"]
        n_img = len(items)
        folds = fold_of([it["sha"] for it in items], ctx["nf"])

        def w_fit(it, mode, n_tok):
            return 0.5 / (n_img * n_cases * n_tok * O)
        st = Stats(H, O, ctx["nf"], dev, full=lin_det and not ctx.get("no_full"))
        t0 = time.time()
        recs, err = collect(fwd, tap, items, ctx["empty_idx"], bands, ctx["noises"], seed, "fit", st, folds, w_fit, dev,
                            ctx["batch"], log_every=max(1, n_img // 4), target_fn=tfn, deadline=dl, band_stride=16 // bpi)
        tm["collect"] = round(time.time() - t0, 1)
        RES["numbers"]["samples"] = len(recs)
        L0 = weighted_loss(recs, n_img, n_cases)
        ok("token residual identity: sum |P^T r|^2 == sum r^2 per sample (output map orthogonal on the used tokens)",
           err < 1e-4, f"max rel err {err:.1e}, {len(recs)} samples, C0 loss on the training images (fresh noises) {L0:.6f}")
        q_tot = float(st.q.sum())
        ok("statistics loss == the evaluator-form loss of the same samples", abs(q_tot - L0) <= 1e-5 * L0,
           f"{q_tot:.8f} vs {L0:.8f}")
        qn = max(q_tot, 1e-300)
        t0 = time.time()
        F = fit(st, fam, mod is not None, lin_det, mod_det, W_c0, lambdas=ctx.get("lambdas", LAMBDAS))
        tm["fit"] = round(time.time() - t0, 1)
        tab = F["table"]
        for m in F["models"]:
            r = max((x for x in tab.values() if x["model"] == m), key=lambda x: x["mean"])
            say(f"  CV {m:4s} best lambda {r['lam']:g}: mean {100 * r['mean']:+.4f}% se {100 * r['se']:.4f}% pooled "
                f"{100 * r['pooled']:+.4f}% positive folds {r['pos']}/{st.nf}")
        pick, ship = F["pick"], F["ship"]
        say(f"one-SE pick {pick['model']} lambda {pick['lam']:g} (mean {100 * pick['mean']:+.4f}%, se "
            f"{100 * pick['se']:.4f}%) -> {'SHIP' if ship else 'no-op'}")
        RES["numbers"]["cv"] = [dict(r, folds=[round(x, 8) for x in r["folds"]]) for r in tab.values()]
        RES["numbers"]["pick"] = {k: v for k, v in pick.items() if k != "folds"}
        RES["numbers"]["ship"] = ship
        RES["numbers"]["phi_singular_values"] = F["sv"]
        RES["numbers"]["loss_c0_fit"] = L0
        RES["numbers"]["fit"] = F  # (for rrf_check.py; json.dumps(default=str) prints the tensors' repr)

        def realize(sd):
            loaded_f, miss_f = strict_load(mp_c0, sd)
            vals = {}
            for k in keys:
                pk = [x for x in loaded_f if (x if isinstance(x, str) else x[0]) == k]
                if pk and k in ctx["base_vals"] and merge_kind(mp_c0.model, k) == "det":
                    vals[k] = realized_weight(mp_c0, k, loaded_f[pk[0]], ctx["base_vals"][k], dev)
            return loaded_f, miss_f, vals

        def deltas_real(vals):
            live = {}
            for k in keys:
                v = get_attr(mp_c0.model, k).detach()
                live[k] = (v.dequantize() if hasattr(v, "dequantize") else v).double()
            dW = (vals[keys[0]].double() - live[keys[0]]) if keys[0] in vals else None
            db = (vals[keys[1]].double() - live[keys[1]]) if keys[1] in vals else None
            W_new = vals[keys[0]].double() if keys[0] in vals else live[keys[0]]
            M = None
            if sp["mod"] and keys[2] in vals:
                dM = vals[keys[2]].double() - live[keys[2]]
                M = dM[sp["shift"] * H:(sp["shift"] + 1) * H]
            if dW is not None and float(dW.abs().max()) == 0.0:
                dW = None
            return dW, db, M, W_new

        def candidate(lab, d, model):
            """the file state, merge check, strict load, realized deltas and gains (intended vs after the merge)."""
            sd = merge_rrf(ctx["c0"], fam, d)[0]
            check_merge(ctx["c0"], sd, fam, d, f"{lab} ({model})")
            loaded_f, miss_f, vals = realize(sd)
            ok(f"{lab} ({model}): the evaluator's strict load - 0 unmatched keys, one patch per module / diff",
               not miss_f and len(loaded_f) == expected_patches(sd), f"{len(loaded_f)} patches, unmatched {miss_f[:2]}")
            dW, db, M, W_new = deltas_real(vals)
            g_int = d["insample_gain"]
            g_real = realized_gain(st, dW, db, M, W_new)
            frac = g_real / g_int if g_int > 0 else float("nan")
            RES["numbers"].setdefault("files", {})[f"{lab}:{model}"] = dict(
                model=model, insample_rel=g_int / qn, realized_rel=g_real / qn, realized_frac=frac,
                pinv_err=d.get("pinv_err"))
            say(f"{lab} ({model}): in-sample gain intended {100 * g_int / qn:+.4f}%, after the evaluator's merge "
                f"{100 * g_real / qn:+.4f}% ({frac:.2f} of it)")
            return dict(sd=sd, model=model, vals=vals, d=d, dW=dW, db=db, M=M, W_new=W_new,
                        gate=g_real > 0 and frac >= 0.5)

        def write(lab, c):
            path = out / f"{lab}__{tag8}.safetensors"
            t0 = time.time()
            tmp = path.with_name(path.name + ".tmp")
            save_file(c["sd"], str(tmp), metadata={})
            os.replace(tmp, path)
            tm[f"write_{lab}"] = round(time.time() - t0, 1)
            say(f"wrote {path} ({path.stat().st_size} B, {tm[f'write_{lab}']}s)")
            return dict({k: v for k, v in c.items() if k != "sd"}, path=str(path))

        written = {}
        prod = None
        if ship:  # the production decision (R63): the CV pick if its gain survives the merge, else the bias-only fit
            c = candidate("R63", F["d_pick"], pick["model"])
            if c["gate"]:
                prod = c
            else:
                say(f"R63 gate: {pick['model']} keeps < half of its in-sample gain after the merge -> not shipped")
                fb = F["bias"]
                if pick["model"] != "B" and fb["mean"] > 0 and fb["mean"] > fb["se"]:
                    c = candidate("R63", F["d_bias"], "B")
                    prod = c if c["gate"] else None
        RES["numbers"]["decision"] = prod["model"] if prod else "no-op"
        if trainer:  # G56CT_RRF: the caller writes the state (save_lora); no diagnostic arm, no EFFECT pass
            return {"R63": prod} if prod else {}
        if prod:
            written["R63"] = write("R63", prod)
        if prod is None or prod["model"] != "B":  # diagnostic arm: the bias-only fit at its CV lambda, always written
            written["R63B"] = write("R63B", candidate("R63B", F["d_bias"], "B"))
        # ---- EFFECT on fresh validation noises: predicted (statistics, realized deltas) == measured (merged parameters)
        vitems = [items[i] for i in ctx["val_ids"]]
        nvi = len(vitems)

        def w_val(it, mode, n_tok):
            return 0.5 / (nvi * 16 * n_tok * O)
        sv_ = Stats(H, O, 1, dev, full=st.full)
        t0 = time.time()
        vrec, _ = collect(fwd, tap, vitems, ctx["empty_idx"], bands, 1, seed, "val", sv_, None, w_val, dev, ctx["batch"],
                          fold_override=0, target_fn=tfn)
        Lv0 = weighted_loss(vrec, nvi, 16)
        for lab, wr in written.items():
            saved = []
            for k, v in wr["vals"].items():
                m_ = get_attr(mp_c0.model, k.rsplit(".", 1)[0])
                pn = k.rsplit(".", 1)[1]
                live = m_._parameters[pn]
                saved.append((m_, pn, live))
                m_._parameters[pn] = torch.nn.Parameter(v.to(live.dtype), requires_grad=False)
            try:
                vrec1 = []
                with torch.no_grad():
                    for ii, it in enumerate(vitems):
                        x0 = it["scaled"]
                        for mode, idx in (("text", it["cond_idx"]), ("no_text", ctx["empty_idx"])):
                            for j in range(0, 16, ctx["batch"]):
                                cb = list(range(j, min(16, j + ctx["batch"])))
                                sg_ = torch.tensor([(b + 0.5) / 16.0 for b in cb], dtype=torch.float32)
                                eps = torch.cat([rrf_noise(x0.shape, seed, it["sha"], b, 0, "val") for b in cb])
                                x0b = x0.expand(len(cb), *x0.shape[1:]).float()
                                svv = sg_.reshape(-1, *([1] * (x0b.dim() - 1)))
                                noisy = ((1 - svv) * x0b + svv * eps).to(dev)
                                # (tests: target_fn = a teacher's velocity; it restores the parameters it swaps)
                                target = (eps - x0b).to(dev) if tfn is None else tfn(noisy, sg_.to(dev), idx, eps, x0b)
                                den = fwd(noisy, sg_.to(dev), idx).float()
                                v = (noisy - den) / svv.to(dev)
                                ls = ((v - target.float()) ** 2).flatten(1).mean(1)
                                for bi, b in enumerate(cb):
                                    vrec1.append(dict(img=ii, band=b, k=0, mode=mode, loss=float(ls[bi])))
            finally:
                for m_, pn, p0 in saved:
                    m_._parameters[pn] = p0
            Lv1 = weighted_loss(vrec1, nvi, 16)
            pred = realized_gain(sv_, wr["dW"], wr["db"], wr["M"], wr["W_new"])
            meas = Lv0 - Lv1
            tol = 0.15 * abs(pred) + 3e-6 * Lv0
            Lvn = max(Lv0, 1e-300)
            ok(f"{lab} ({wr['model']}) EFFECT: loss reduction on fresh validation noises measured through the merged model "
               f"== predicted from the statistics", abs(meas - pred) <= tol and (pred <= 0 or meas > 0),
               f"measured {100 * meas / Lvn:+.4f}% predicted {100 * pred / Lvn:+.4f}% (C0 {Lv0:.6f}, {len(vrec1)} samples)")
            RES["numbers"].setdefault("effect", {})[lab] = dict(model=wr["model"], measured_rel=meas / Lvn,
                                                                 predicted_rel=pred / Lvn, loss_c0=Lv0, loss_rrf=Lv1)
        tm["effect"] = round(time.time() - t0, 1)
        return written
    finally:
        tap.remove()


# ------------------------------------------------------------------------------------------------------------- box main
def load_trainer(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("g56t", str(path))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    sys.path.insert(0, str(T.COMFY_ROOT))
    sys.path.insert(0, str(Path(path).resolve().parent))
    return T


class Job:
    """a comfy sampler: TrainGuider.sample() hands it the guider (= the model wrapper: the evaluator's CFGGuider path) with
    the processed conds after loading the patched model (prepare_sampling -> load_models_gpu: ComfyUI's own LoRA merge)."""

    def __init__(self, fn):
        self.fn = fn
        self.result = None

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        import torch
        with torch.inference_mode(False):
            self.result = self.fn(model_wrap, model_wrap.conds["positive"], extra_args, noise.device)
        return latent_image

    def max_denoise(self, model_wrap, sigmas):
        return False


def guided(mp_c0, conds, lat0, seed, make_ctx):
    """rrf_run inside ComfyUI's sampling machinery, as the trainer trains: TrainGuider(C0 patcher) -> prepare_sampling ->
    load_models_gpu (the evaluator's LoRA merge) -> Job.sample(guider, processed conds); forward = the guider itself
    (CFGGuider cfg 1.0 -> sampling_function -> calc_cond_batch -> apply_model), one processed cond per prompt."""
    import torch
    from comfy_extras.nodes_train import TrainGuider
    guider = TrainGuider(mp_c0, offloading=False)
    guider.set_conds(conds)

    def run(model_wrap, conds_p, extra_args, dev):
        def fwd(noisy, sigmas, idx):
            model_wrap.conds["positive"] = [conds_p[idx]]
            return model_wrap(noisy, sigmas, **extra_args)
        return rrf_run(make_ctx(fwd, dev))
    job = Job(run)
    guider.sample(torch.zeros_like(lat0), torch.zeros_like(lat0), job, torch.tensor([1.0, 0.0]), disable_pbar=True, seed=seed)
    return job.result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(SPEC))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--c0", required=True)
    ap.add_argument("--out", default="/cache/rrf")
    ap.add_argument("--json", default="")
    ap.add_argument("--trainer", default=os.environ.get("RRF_TRAINER", "/workspace/comfytrain/g56_comfy_train.py"))
    ap.add_argument("--seed", type=int, default=63)
    ap.add_argument("--noises", type=int, default=1, help="fresh noises per (image, band, prompt)")
    ap.add_argument("--bands-per-image", type=int, default=8,
                    help="sigma bands per (image, prompt): 8 = the alternating half of the 16 midpoints (the G56CT_RRF form)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4, help="bands per forward")
    ap.add_argument("--val-images", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--no-full", action="store_true", help="skip the full-weight form W (x x^T statistics)")
    a = ap.parse_args(argv)
    t_all = time.time()
    fam, tag8 = a.family, a.task[:8]
    T = load_trainer(a.trainer)
    import torch
    import comfy.model_management as mm
    import comfy.sd
    import nodes
    import node_helpers
    from safetensors.torch import load_file
    if os.environ.get("G56CT_FP8_COMPUTE", "0") != "1":  # the trainer's main(): A100 evaluator numerics on any GPU
        mm.supports_fp8_compute = lambda device=None: False
    T.apply_family_defaults(fam)
    device = mm.get_torch_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tm = RES["numbers"].setdefault("seconds", {})
    say(f"T063 RRF {fam} task {a.task} c0 {a.c0}; device {device}, torch {torch.__version__}, trainer {a.trainer}")
    # ---- data, base, latents, conditioning: the trainer's own loaders, as its main() runs them
    t0 = time.time()
    items = T.decode_items(T.read_rows(a.dataset))
    if a.max_images:
        items = items[: a.max_images]
    mp = T.load_base_model(fam, a.repo, device)
    tm["load_base"] = round(time.time() - t0, 1)
    base_vals = base_values(mp.model, final_keys(fam))
    t0 = time.time()
    vae = T.load_vae(fam, a.repo)
    for it in items:
        lat = T.encode_latent(vae, it["image"], device)
        it["scaled"] = mp.model.process_latent_in(lat.clone()).float().cpu()
        it["image"] = None
    del vae
    mm.soft_empty_cache()
    clip, _ = T.load_clip(fam, device, a.repo)
    spec = T.FAMILIES[fam]
    prompts = sorted(set(it["caption"] for it in items))
    conds, pidx = [], {}
    for p in prompts + [""]:
        cnd = nodes.CLIPTextEncode().encode(clip, p)[0]
        if len(cnd) != 1:
            raise RuntimeError(f"expected one conditioning entry per prompt, got {len(cnd)}")
        if spec["guidance"] is not None:
            cnd = node_helpers.conditioning_set_values(cnd, {"guidance": spec["guidance"]})
        pidx[p] = len(conds)
        conds.extend(cnd)
    del clip
    mm.unload_all_models()
    mm.soft_empty_cache()
    for it in items:
        it["cond_idx"] = pidx[it["caption"]]
    tm["latents_conds"] = round(time.time() - t0, 1)
    say(f"{len(items)} training images, {len(prompts)} captions, latent {tuple(items[0]['scaled'].shape)}")
    # ---- C0 through the evaluator's loader (LoraLoader = load_lora_for_models, strength 1.0)
    c0 = load_file(a.c0)
    loaded, miss = strict_load(mp, c0)
    ok("C0 loads strictly (0 unmatched keys)", not miss and len(loaded) == expected_patches(c0), f"{len(loaded)} patches")
    mp_c0 = comfy.sd.load_lora_for_models(mp, None, c0, 1.0, 1.0)[0]
    shas = [it["sha"] for it in items]
    val_ids = sorted(range(len(items)), key=lambda i: hashlib.sha256(f"rrf63val|{shas[i]}".encode()).hexdigest())[: a.val_images]
    t0 = time.time()
    written = guided(mp_c0, conds, items[0]["scaled"], a.seed, lambda fwd, dev: dict(
        fam=fam, mp=mp_c0, fwd=fwd, items=items, empty_idx=pidx[""], base_vals=base_vals, c0=c0, out=out, tag8=tag8,
        seed=a.seed, noises=a.noises, nf=max(2, min(a.folds, len(items))), batch=a.batch, val_ids=val_ids, device=dev,
        no_full=a.no_full, bands_per_image=a.bands_per_image)) or {}
    tm["run"] = round(time.time() - t0, 1)
    tm["total"] = round(time.time() - t_all, 1)
    with open(out / "ready.txt", "a") as fh:
        for lab, wr in written.items():
            fh.write(f"{lab}__{tag8}\t{fam}\t{wr['model']}\t{wr['path']}\n")
    RES["numbers"]["written"] = {lab: dict(model=wr["model"], path=wr["path"]) for lab, wr in written.items()}
    RES["numbers"].pop("fit", None)
    return finish(a.json or str(out / f"rrf_{tag8}.json"))


def finish(json_path=None):
    n_ok = sum(1 for _, g, _ in RES["checks"] if g)
    good = n_ok == len(RES["checks"]) and len(RES["checks"]) > 0
    if json_path:
        try:
            Path(json_path).write_text(json.dumps(RES, indent=1, default=str))
        except Exception as e:  # noqa: BLE001
            say(f"json not written: {e}")
    say(f"RESULT {'PASS' if good else 'FAIL'} ({n_ok}/{len(RES['checks'])} checks) seconds "
        f"{json.dumps(RES['numbers'].get('seconds', {}))}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
