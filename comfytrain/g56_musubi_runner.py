#!/usr/bin/env python3
"""




  




  
  

  
  
  













"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def fingerprint(tensor):
    """The evaluator's sample fingerprint (god_hidream_objective.py:34-35)."""
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def latent_keys(cache_dirs, remove_dtype_suffix):
    """{fingerprint of a cached target: the pair's stem} over the cache folders of every dataset block (the evaluator's
    _latent_fingerprints, :38-57, for several folders). The text block and the no-text block cache the same targets under the
    same names: one entry. The same target under two DIFFERENT stems (two pairs with one target image) cannot be told apart in a
    batch -> None: those pairs train on Musubi's own loss."""
    from safetensors import safe_open
    keys = {}
    for d in cache_dirs:
        for path in sorted(Path(d).glob("*.safetensors")):
            if path.name.endswith("_te.safetensors"):
                continue
            with safe_open(str(path), framework="pt") as handle:
                names = [k for k in handle.keys() if k.startswith("latents_") and remove_dtype_suffix(k).rsplit("_", 1)[0] == "latents"]
                if len(names) != 1:
                    continue
                fp = fingerprint(handle.get_tensor(names[0]))
            stem = path.name.split("_", 1)[0]
            keys[fp] = stem if keys.get(fp, stem) == stem else None
    return keys


def conv_as_linear(conv, orig, counts, name):
    """The forward of the Conv3d / Conv2d `conv` (its forward so far: `orig`) as ONE F.linear when its kernel covers its whole input:
    kernel == stride, no padding, dilation 1, groups 1, zero padding mode, and an input whose spatial size IS the kernel's - the
    output (N, out, 1, ...) is then x.reshape(N, -1) @ W.reshape(out, -1).T + b, the same terms (C, T, P, P in the same order). Any
    other input -> `orig` (counted under '<name> (conv)'). -> the forward, or None when `conv` is not of that kind."""
    import torch.nn.functional as F
    k, pad = tuple(conv.kernel_size), conv.padding
    if (tuple(conv.stride) != k or conv.groups != 1 or any(d != 1 for d in conv.dilation) or getattr(conv, "padding_mode", "zeros") != "zeros"
            or not (pad == "valid" or (not isinstance(pad, str) and all(p == 0 for p in pad)))):
        return None
    out_ch, ones = conv.out_channels, (1,) * len(k)

    def forward(x):
        if x.dim() != len(k) + 2 or tuple(x.shape[2:]) != k:
            counts[name + " (conv)"] = counts.get(name + " (conv)", 0) + 1
            return orig(x)
        counts[name] = counts.get(name, 0) + 1
        return F.linear(x.reshape(x.shape[0], -1), conv.weight.reshape(out_ch, -1), conv.bias).reshape(x.shape[0], out_ch, *ones)
    return forward


def conv_backend(conv, n):
    """the backend torch selects for `conv` on an input of n items of its kernel's size, + the cuDNN switches (why a conv is slow)"""
    import torch
    try:
        x = torch.zeros(n, conv.in_channels, *conv.kernel_size, device=conv.weight.device, dtype=conv.weight.dtype)
        pad = [0] * len(conv.kernel_size) if isinstance(conv.padding, str) else list(conv.padding)
        return str(torch._C._select_conv_backend(x, conv.weight, conv.bias, list(conv.stride), pad, list(conv.dilation), False,
                                                 [0] * len(conv.kernel_size), conv.groups, None))
    except Exception as e:  # noqa: BLE001
        return f"unknown ({type(e).__name__}: {str(e)[:80]})"


TF32_TOL = 5e-3   # the production-precision gate of a float32 conv that may run in TF32 (10 mantissa bits: unit roundoff 4.9e-4)


def tf32_allowed(device):
    """True when a float32 conv or matmul on `device` may run in TF32: CUDA with torch.backends.cudnn.allow_tf32 (convolutions;
    PyTorch's default True) or torch.backends.cuda.matmul.allow_tf32 / a float32 matmul precision other than 'highest' (matmuls)"""
    import torch
    if getattr(device, "type", device) != "cuda":
        return False
    return bool(torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32
                or torch.get_float32_matmul_precision() != "highest")


def check_tol(out_dtype, tf32):
    """patchlinear_install's production-precision tolerance - the rounding of what the conv really computed in: an output in bf16 /
    fp16 -> 2e-2; a float32 output that may have been computed in TF32 (tf32: tf32_allowed of its device) -> TF32_TOL; else 1e-5"""
    import torch
    if out_dtype in (torch.bfloat16, torch.float16):
        return 2e-2
    return TF32_TOL if tf32 and out_dtype == torch.float32 else 1e-5


def _conv64(conv, gen):
    """a float64 TWIN of `conv` (its kind and hyper-parameters) with random weights drawn from `gen` - patchlinear_install's algebra
    check: the linear form's algebra does not depend on the weights, and a module initialized to zeros (Musubi's lora_up) would make a
    copy of its own weights a trivial check (0 against 0). Built without its init (torch.nn.utils.skip_init: no draw from the global
    RNG)."""
    import torch
    c = torch.nn.utils.skip_init(type(conv), conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                                 padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None,
                                 padding_mode=conv.padding_mode, device=conv.weight.device, dtype=torch.float64)
    with torch.no_grad():
        c.weight.copy_(torch.randn(c.weight.shape, generator=gen, device=c.weight.device, dtype=torch.float64))
        if conv.bias is not None:
            c.bias.copy_(torch.randn(c.bias.shape, generator=gen, device=c.bias.device, dtype=torch.float64))
    return c


def patchlinear_install(transformer, network, counts, tf32=None):
    """G56NF_MUSUBI_PATCHLINEAR: every conv LoRA module of `network` whose base is a Conv3d (Musubi networks/lora.py: apply_to keeps
    the base's forward as org_forward) -> its base conv, lora_down and lora_up as conv_as_linear; a Qwen3VLVisionPatchEmbed.proj of
    `transformer` without a LoRA -> its own forward. Each replacement first checked against the conv on a random input of its kernel's
    size, two gates (else left as it was): the algebra - conv_as_linear on a float64 twin of the conv with random weights (_conv64)
    against the twin itself, autocast off, relative difference <= 1e-9 ('check64'); the production precision - orig vs the linear form
    on the module as it is, under the ambient autocast, within the rounding of what the conv really computed in (check_tol: its output
    bf16 / fp16 2e-2; float32 on CUDA while TF32 is allowed - tf32_allowed, or `tf32` when given (tests) - TF32_TOL; else 1e-5;
    'check', the tolerance per module in 'tol', the TF32 switches in 'tf32'; a check whose conv output is all zeros - Musubi's lora_up
    at init - is named in 'trivial').
    A patch embed whose forward is already wrapped by something outside the network is never replaced. The checks draw from private
    generators only (seed 0 on the conv's device): the training's global RNG streams - its noise, timesteps, data order - are where they
    would be without the knob. -> report dict"""
    import torch
    rep = dict(replaced=[], skipped=[], check=0.0, check64=0.0, backend={}, cudnn=None, tf32=None, tol={}, trivial=[])
    targets, based = [], set()
    for m in (network.modules() if network is not None else []):
        down, up, orgf = getattr(m, "lora_down", None), getattr(m, "lora_up", None), getattr(m, "org_forward", None)
        base = getattr(orgf, "__self__", None)
        if isinstance(down, torch.nn.Conv3d) and isinstance(up, torch.nn.Conv3d) and isinstance(base, torch.nn.Conv3d):
            nm = str(getattr(m, "lora_name", "lora"))
            based.add(id(base))
            targets += [(nm + " base", base, orgf, lambda f, m=m: setattr(m, "org_forward", f)),
                        (nm + " lora_down", down, down.forward, lambda f, d=down: setattr(d, "forward", f)),
                        (nm + " lora_up", up, up.forward, lambda f, u=up: setattr(u, "forward", f))]
    for name, mod in (transformer.named_modules() if transformer is not None else []):
        proj = getattr(mod, "proj", None)
        if type(mod).__name__ == "Qwen3VLVisionPatchEmbed" and isinstance(proj, torch.nn.Conv3d) and id(proj) not in based:
            if "forward" in vars(proj):   # wrapped by something we do not know (a LoRA outside `network`?): never replaced
                rep["skipped"].append(f"{name}.proj: its forward is already wrapped (not by a LoRA module of the network)")
                continue
            targets.append((name + ".proj", proj, proj.forward, lambda f, p=proj: setattr(p, "forward", f)))
    if targets:
        conv0 = targets[0][1]
        rep["backend"] = {"n 2": conv_backend(conv0, 2), "n 8192": conv_backend(conv0, 8192)}
        cd = torch.backends.cudnn
        rep["cudnn"] = dict(enabled=bool(cd.enabled), available=bool(cd.is_available()), version=cd.version() if cd.is_available() else None,
                            deterministic=bool(cd.deterministic), benchmark=bool(cd.benchmark),
                            deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()))
        rep["tf32"] = dict(cudnn=bool(cd.allow_tf32), matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                           matmul_precision=torch.get_float32_matmul_precision(),
                           used=bool(tf32_allowed(conv0.weight.device) if tf32 is None else tf32))
    for name, conv, orig, setf in targets:
        fwd, probe = conv_as_linear(conv, orig, counts, name), conv_as_linear(conv, orig, {}, name)
        if fwd is None:
            rep["skipped"].append(f"{name}: not a conv over exactly its kernel (kernel {tuple(conv.kernel_size)}, stride {tuple(conv.stride)}, "
                                  f"padding {conv.padding}, dilation {tuple(conv.dilation)}, groups {conv.groups})")
            continue
        try:
            with torch.no_grad(), torch.autocast(device_type=conv.weight.device.type, enabled=False):
                gen = torch.Generator(device=conv.weight.device)   # private: the global streams never move
                gen.manual_seed(0)
                c64 = _conv64(conv, gen)
                x64 = torch.randn(2, conv.in_channels, *conv.kernel_size, device=conv.weight.device, dtype=torch.float64, generator=gen)
                a64, b64 = c64(x64), conv_as_linear(c64, c64.forward, {}, name)(x64)
                rel64 = float((b64 - a64).norm() / max(float(a64.norm()), 1e-300))
            if not rel64 <= 1e-9:
                rep["skipped"].append(f"{name}: the linear form's algebra differs from the conv in float64 by {rel64:.2e} (> 1e-09)")
                continue
            with torch.no_grad():
                x = torch.randn(2, conv.in_channels, *conv.kernel_size, device=conv.weight.device, dtype=conv.weight.dtype,
                                generator=gen)
                a, b = orig(x), probe(x)
                tol = check_tol(a.dtype, tf32_allowed(conv.weight.device) if tf32 is None else tf32)   # what the conv computed in
                a, b = a.float(), b.float()
                rel = float((b - a).norm() / max(float(a.norm()), 1e-30))
                if float(a.norm()) == 0.0:
                    rep["trivial"].append(name)
            if not rel <= tol:
                rep["skipped"].append(f"{name}: the linear form differs from the conv by {rel:.2e} (> {tol:g})")
                continue
        except Exception as e:  # noqa: BLE001
            rep["skipped"].append(f"{name}: check failed ({type(e).__name__}: {str(e)[:80]})")
            continue
        setf(fwd)
        rep["replaced"].append(name)
        rep["tol"][name] = tol
        rep["check"], rep["check64"] = max(rep["check"], rel), max(rep["check64"], rel64)
    torch.clear_autocast_cache()   # autocast caches its casts of trainable leaves: the checks' casts, made under no_grad, must
    return rep                     # not stand in for the training's (they would carry no gradient for the rest of that autocast)


class G56PatchRules:
    """
    """

    def _g56_plin_setup(self, status):
        self._g56_plin = dict(status=status, rep=None, counts={}, steps=0)

    def _g56_plin_write(self):
        p = getattr(self, "_g56_plin", None)
        if p is None or not p["status"]:
            return
        try:
            tmp = Path(str(p["status"]) + ".tmp")
            tmp.write_text(json.dumps(dict(p["rep"] or {}, calls=p["counts"], steps=p["steps"])))
            os.replace(tmp, p["status"])
        except Exception:  # noqa: BLE001
            pass

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise, *rest):
        p = getattr(self, "_g56_plin", None)
        if p is not None:
            p["steps"] += 1
            if p["rep"] is None:
                try:
                    p["rep"] = patchlinear_install(transformer, network, p["counts"])
                except Exception as e:
                    p["rep"] = dict(replaced=[], skipped=[], off=f"{type(e).__name__}: {str(e)[:120]}")
                print(f"g56 patchlinear: {json.dumps(p['rep'])[:900]}", flush=True)
                self._g56_plin_write()
            elif p["steps"] % 25 == 0:
                self._g56_plin_write()
        return super().process_batch(args, accelerator, transformer, network, batch, latents, noise, *rest)


class _G56Stop(Exception):
    """G56NF_MUSUBI_MEMBERS: the member rules end the training here (Musubi's loop is left through it; main() returns, exit 0)."""


def _member_rule():
    """member_plan and RESET_OPTIMIZERS of the AI Toolkit runner (g56_aitk_runner.py next to this file; standard library only at
    import): ONE member rule for both paths."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import g56_aitk_runner as AR
    return AR.member_plan, AR.RESET_OPTIMIZERS


def _plan5(spec, length):
    """PLAN5's state of the AI Toolkit runner (g56_aitk_runner.Plan5: ONE planner for both paths) for the plan the driver hands over
    (G56_PLAN5, JSON); None when there is none."""
    if not spec:
        return None
    _member_rule()
    import g56_aitk_runner as AR
    return AR.Plan5(json.loads(spec), length)


def _save_dtype(args):
    """Musubi's own save dtype (trainer_base.py: --save_precision; unset = the network's own dtype)"""
    import torch
    p = getattr(args, "save_precision", None)
    return None if p is None else {"float": torch.float32, "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[p]


class G56MemberRules:
    """
    
    
    
    
    
    
    
    """

    def _g56_mem_setup(self, env=None):
        env = os.environ if env is None else env
        k = int(env.get("G56_MEMBERS") or 1)
        self._g56_mem = None
        if k <= 1 and not env.get("G56_PLAN5"):
            return
        self._g56_mem = dict(k_max=k, L=int(env["G56_MEMBER_LEN"]), deadline=float(env["G56_MEMBERS_DEADLINE"]),
                             seed=int(env.get("G56_MEMBERS_SEED") or 42), status=env.get("G56_MEMBERS_STATUS"),
                             dir=Path(env["G56_MEMBERS_DIR"]), k=1, step0=0, t0=None, t_train0=None, last=None, times=[], log=[],
                             opt=None, dts=[], save_s=10.0)
        m = self._g56_mem
        m["len"] = m["L"]   # the length of the member being trained (PLAN5 may plan another one than L)
        m["p5"] = _plan5(env.get("G56_PLAN5"), m["L"])   # PLAN5 (None = off: member_plan, the deadline above, as before)
        if m["p5"] is not None:
            m["k_max"], m["deadline"], m["n"] = m["p5"].khi, m["p5"].deadline(), 0

    def _g56_mem_write(self, phase):
        m = getattr(self, "_g56_mem", None)
        if m is None or not m["status"]:
            return
        try:
            tmp = Path(str(m["status"]) + ".tmp")
            extra = {} if m.get("p5") is None else dict(plan5=dict(m["p5"].info, t_train0=m["t_train0"], deadline=m["deadline"]))
            tmp.write_text(json.dumps(dict(phase=phase, members_done=len(m["times"]), member_log=m["log"], **extra)))
            os.replace(tmp, m["status"])
        except Exception:  # noqa: BLE001
            pass

    def on_train_start(self, args, accelerator, network, transformer, optimizer):
        super().on_train_start(args, accelerator, network, transformer, optimizer)
        if getattr(self, "_g56_mem", None) is not None:
            self._g56_mem["opt"] = optimizer

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise, *rest):
        m = getattr(self, "_g56_mem", None)
        if m is not None and m["t_train0"] is None:   # the first training step starts: the members' window
            m["t_train0"] = m["t0"] = m["last"] = time.time()
        return super().process_batch(args, accelerator, transformer, network, batch, latents, noise, *rest)

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step):
        super().on_post_optimizer_step(args, accelerator, network, transformer, sync_gradients, global_step)
        m = getattr(self, "_g56_mem", None)
        if m is None or not sync_gradients:
            return
        now = time.time()
        p5 = m.get("p5")
        if m["last"] is not None:
            m["dts"] = (m["dts"] + [now - m["last"]])[-20:]
            if p5 is not None:   # PLAN5: the step times of the plan
                p5.step(now - m["last"])
        m["last"] = now
        local = int(global_step) + 1 - m["step0"]   # this optimizer step, counted in the member (Musubi adds it after the hook)
        if p5 is not None:
            m["n"] += 1
            if m["n"] % 25 == 0:
                m["deadline"] = min(m["deadline"], p5.tick())
            if not p5.probed and m["k"] == 1 and p5.probe <= local < m["len"]:   # the first plan, while member 1 trains
                p5.probed = True
                go, length, k_fit, why = p5.plan(0, now, m["save_s"], credit=local)
                if go:
                    m["len"] = m["len0"] = max(int(length), local + 1)
                m["deadline"] = p5.deadline()
                line = f"plan5 probe at step {local} of member 1: length {m['len']}, K_fit {k_fit}, K planned {p5.k_plan} ({why})"
                print("g56 " + line, flush=True)
                m["log"].append(line)
                self._g56_mem_write("train")
        if p5 is not None and p5.probed:   # PLAN5: while this member's length is not final, the plan is checked against the clock
            beat = local < m["len"] - 1 and m["n"] % 25 == 0
            if p5.k_plan > m["k"] and (local >= m["len"] - 1 or beat):   # do the members after this one still fit?
                new = p5.rest(m["k"], now, m["save_s"], local, m["len"], now - m["t0"])
                m["deadline"] = p5.deadline()
                if new is not None:   # none does: this member is the plan's last - never shortened for a member that is not trained
                    line = f"plan5: member {m['k']} at step {local}: no member fits after it any more - it is the plan's last, its length {m['len']} -> {new} steps"
                    print("g56 " + line, flush=True)
                    m["log"].append(line)
                    m["len"] = m["len0"] = new
                    self._g56_mem_write("train")
            elif p5.k_plan <= m["k"] and local < m["len"] and (local == m["len"] - 1 or beat):   # the plan's last member: what fits now
                new = p5.refit(m["k"], now, m["save_s"], local, m["len"], p5.keep * m.get("len0", m["len"]))
                if new is not None:
                    if abs(new - m.get("said", -10 ** 9)) >= max(10, m["L"] // 20):
                        m["said"] = new
                        line = f"plan5: member {m['k']} at step {local}: the plan's last member, its length {m['len']} -> {new} steps (what fits before the deadline now)"
                        print("g56 " + line, flush=True)
                        m["log"].append(line)
                        self._g56_mem_write("train")
                    m["len"] = new
        if local >= m["len"]:
            self._g56_member_done(args, accelerator, network, local)
            return
        dts = sorted(m["dts"]) or [0.0]
        if (m["k"] > 1 or p5 is not None) and now + 1.5 * dts[len(dts) // 2] + 2.0 * m["save_s"] > m["deadline"]:   # the deadline brake
            if p5 is not None and p5.cut > 0 and local >= p5.cut * m.get("len0", m["len"]):   # PLAN5 'cut': far enough into its PLANNED length - kept
                self._g56_member_done(args, accelerator, network, local, cut=True)
            if m["k"] == 1:   # (PLAN5: member 1 trains on as Musubi's own run does - its step saves ship, the driver stops it)
                return
            line = f"member {m['k']}/{m['k_max']} cut by the deadline at step {local} of {m['len']}: no file"
            print("g56 " + line, flush=True)
            m["log"].append(line)
            self._g56_mem_write("done")
            raise _G56Stop()

    def _g56_member_done(self, args, accelerator, network, local, cut=False):
        """Member k reached its length: saved, timed, and the plan for the next one; the end (_G56Stop) when none starts. cut (PLAN5):
        the deadline stopped it far enough into its planned length - it is saved and the training ends."""
        m = self._g56_mem
        member_plan, _ = _member_rule()
        net = accelerator.unwrap_model(network) if accelerator is not None else network
        t = time.time()
        m["dir"].mkdir(parents=True, exist_ok=True)
        tmp, dst = m["dir"] / f".tmp_member_{m['k']:02d}.safetensors", m["dir"] / f"member_{m['k']:02d}.safetensors"
        net.save_weights(str(tmp), _save_dtype(args), {"g56_member": str(m["k"]), "g56_step": str(local),
                                                       **({"g56_len": str(int(m["len"]))} if m.get("p5") is not None else {})})
        os.replace(tmp, dst)
        now = time.time()
        m["save_s"] = max(m["save_s"] * 0.5, now - t)
        m["times"].append(now - m["t0"])
        window = m["deadline"] - m["t_train0"]
        length = m["L"]
        if cut:
            line = f"member {m['k']}/{m['k_max']} cut by the deadline at step {local} of {m['len']}: kept, at least {m['p5'].cut:g} of its length"
            print("g56 " + line, flush=True)
            m["log"].append(line)
            self._g56_mem_write("done")
            raise _G56Stop()
        if m.get("p5") is not None:
            go, length, k_fit, why = m["p5"].plan(m["k"], now, m["save_s"], member_wall=m["times"][-1])
            m["deadline"] = m["p5"].deadline()
            window = m["deadline"] - m["t_train0"]
        else:
            go, k_fit, why = member_plan(m["k"], m["k_max"], m["times"], window, m["deadline"] - now)
        if go:
            try:
                self._g56_reset_member(args, accelerator, network, m["k"] + 1)
            except Exception as re_:  # noqa: BLE001  (no reset = no further member; the finished ones are shipped)
                go, why = False, f"reset refused: {type(re_).__name__}: {str(re_)[:120]}"
        line = (f"member {m['k']}/{m['k_max']} done: steps {local}, t {m['times'][-1]:.1f} s, planner: window {window:.0f} s, longest "
                f"member {max(m['times']):.1f} s, K_fit {k_fit}, next: {'start' if go else 'stop'} ({why})")
        print("g56 " + line, flush=True)
        m["log"].append(line)
        if not go:
            self._g56_mem_write("done")
            raise _G56Stop()
        m["k"], m["step0"], m["t0"], m["len"], m["len0"] = m["k"] + 1, m["step0"] + local, now, int(length), int(length)
        self._g56_mem_write("train")

    def _g56_reset_member(self, args, accelerator, network, k):
        """Member k starts from scratch in this process: every LoRA module re-initialized as Musubi's LoRAModule.__init__ does
        (networks/lora.py: kaiming_uniform_(lora_down.weight, a=sqrt(5)), zeros_(lora_up.weight)), the optimizer's state dropped (built
        again from zero at the next step), the RNGs re-seeded with seed + k - 1. Refused (raises, nothing touched) unless every LoRA
        module is lora_down / lora_up weights only (Linear / Conv2d / Conv3d, no bias), the optimizer trains exactly those weights and is
        one of RESET_OPTIMIZERS, and the LR schedule is constant without warmup (no state that would have to start again)."""
        import math
        import random
        import numpy as np
        import torch
        _, reset_ok = _member_rule()
        m = self._g56_mem
        net = accelerator.unwrap_model(network) if accelerator is not None else network
        kinds = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)
        loras = [mm for mm in net.modules() if hasattr(mm, "lora_down") and hasattr(mm, "lora_up")]
        weights = []
        for mm in loras:
            d, u = mm.lora_down, mm.lora_up
            if not (isinstance(d, kinds) and isinstance(u, kinds)) or d.bias is not None or u.bias is not None:
                raise RuntimeError(f"a LoRA module that is not lora_down / lora_up weights only ({type(mm).__name__})")
            weights += [d.weight, u.weight]
        opt = getattr(m["opt"], "optimizer", m["opt"])
        if opt is None:
            raise RuntimeError("no optimizer seen at on_train_start")
        trained = {id(p) for g in opt.param_groups for p in g["params"]}
        if not weights or trained != {id(w) for w in weights}:
            raise RuntimeError(f"the optimizer trains {len(trained)} tensors, the LoRA modules hold {len(weights)} factors")
        kind = (type(opt).__module__.split(".")[0], type(opt).__name__)
        if kind not in reset_ok:
            raise RuntimeError(f"the optimizer is {kind[0]} {kind[1]}: not one whose cleared state is known to start again")
        sched, warm = getattr(args, "lr_scheduler", "constant"), getattr(args, "lr_warmup_steps", 0)
        if sched != "constant" or warm:
            raise RuntimeError(f"the LR schedule is {sched!r} (warmup {warm}): its state would have to start again too")
        seed = int(m["seed"]) + (k - 1)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            for mm in loras:
                torch.nn.init.kaiming_uniform_(mm.lora_down.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(mm.lora_up.weight)
        opt.state.clear()
        opt.zero_grad(set_to_none=True)


class G56MaskRules:
    """The mask rules, mixed in front of Musubi's HiDreamO1NetworkTrainer (module level, so tools/newfam_check.py runs them against
    a stand-in without Musubi). `self._g56_unpatchify` / `self._g56_patch` = Musubi's hidream_o1_utils.unpatchify_pixels / PATCH_SIZE."""

    def _g56_mask_setup(self, mask_dir, keys, status, unpatchify, patch):
        self._g56_mask_dir, self._g56_keys, self._g56_status = Path(mask_dir), keys, status
        self._g56_unpatchify, self._g56_patch = unpatchify, int(patch)
        self._g56_key = None
        self._g56_weights = {}   # (stem, width, height) -> the normalized weights [1, 1, H, W] on the loss's device | None
        self._g56_stats = dict(steps=0, weighted=0, mass=[], off=None)
        self._g56_seen = {}      # stem -> has a mask

    def _g56_mask_off(self, why):
        if self._g56_stats["off"] is None:
            self._g56_stats["off"] = why
            print(f"g56 edit mask: OFF from step {self._g56_stats['steps']} ({why}): Musubi's own loss", flush=True)
            self._g56_mask_write()

    def _g56_mask_write(self):
        s = self._g56_stats
        mass = sorted(s["mass"])
        try:
            if self._g56_status:
                tmp = Path(str(self._g56_status) + ".tmp")
                tmp.write_text(json.dumps(dict(steps=s["steps"], attached=s["weighted"], weighted=s["weighted"], items=len(self._g56_seen),
                                               masked=sum(1 for v in self._g56_seen.values() if v),
                                               mass_median=mass[len(mass) // 2] if mass else None, off=s["off"])))
                os.replace(tmp, self._g56_status)
        except Exception:  # noqa: BLE001
            pass

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise, *rest):
        self._g56_key = None
        if self._g56_stats["off"] is None:
            try:
                self._g56_stats["steps"] += 1
                if batch["latents"].shape[0] == 1:
                    self._g56_key = self._g56_keys.get(fingerprint(batch["latents"][0]))
                if self._g56_stats["steps"] % 25 == 0:
                    self._g56_mask_write()
            except Exception as e:
                self._g56_mask_off(f"{type(e).__name__}: {str(e)[:120]}")
        return super().process_batch(args, accelerator, transformer, network, batch, latents, noise, *rest)

    def _g56_item(self, output):
        """-> (prediction, target, normalized weights) in pixels for the batch's pair | None (no mask: Musubi's own loss)."""
        import torch
        from PIL import Image
        import g56_editmask as EM
        stem = self._g56_key
        hp, wp = output.extra["pixel_grid_hw"]
        height, width = hp * self._g56_patch, wp * self._g56_patch
        key = (stem, width, height)
        if key not in self._g56_weights:
            f = self._g56_mask_dir / f"{stem}.png"
            self._g56_weights[key] = None
            self._g56_seen[stem] = self._g56_seen.get(stem, False) or f.is_file()
            if f.is_file():
                aligned = EM.cover_and_center_crop(Image.open(f), width, height)
                weights = EM.mask_weights(aligned)
                self._g56_stats["mass"].append(EM.changed_mass(weights, EM.mask_as_float(aligned)))
                w = torch.from_numpy(weights).to(output.pred.device)[None, None]
                self._g56_weights[key] = w / w.mean()
        w = self._g56_weights[key]
        if w is None:
            return None
        return (self._g56_unpatchify(output.pred.float(), height, width), self._g56_unpatchify(output.target.float(), height, width), w)

    def compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
        item = None
        if self._g56_key is not None and self._g56_stats["off"] is None:
            try:
                if args.weighting_scheme != "none":   # the evaluator's own condition (:143-144)
                    raise ValueError(f"weighting scheme {args.weighting_scheme!r}")
                item = self._g56_item(output)
            except Exception as e:  # noqa: BLE001
                self._g56_mask_off(f"{type(e).__name__}: {str(e)[:120]}")
        if item is None:
            return super().compute_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step)
        pred, target, weights = item
        self._g56_stats["weighted"] += 1
        return ((pred - target) ** 2 * weights).mean(), {}


def main():
    from musubi_tuner.hidream_o1 import hidream_o1_utils
    from musubi_tuner.hidream_o1_train_network import HiDreamO1NetworkTrainer
    from musubi_tuner.hidream_o1_train_network import hidream_o1_setup_parser
    from musubi_tuner.hv_train_network import read_config_from_file
    from musubi_tuner.hv_train_network import setup_parser_common
    from musubi_tuner.utils.model_utils import remove_dtype_suffix

    class G56MaskedTrainer(G56MemberRules, G56PatchRules, G56MaskRules, HiDreamO1NetworkTrainer):
        def __init__(self):
            super().__init__()
            self._g56_mem_setup()
            self._g56_plin = None
            if os.environ.get("G56_PATCHLINEAR") == "1":
                self._g56_plin_setup(os.environ.get("G56_PATCHLINEAR_STATUS"))
            mask_dir, status = os.environ.get("G56_EDIT_MASK_DIR", ""), os.environ.get("G56_EDIT_MASK_STATUS")
            self._g56_mask_setup(mask_dir, {}, status, hidream_o1_utils.unpatchify_pixels, hidream_o1_utils.PATCH_SIZE)
            if not mask_dir:   # the runner without the mask knob (G56NF_MUSUBI_PATCHLINEAR alone): Musubi's own loss, no word
                self._g56_stats["off"] = "no edit mask"
                return
            try:
                dirs = [d for d in os.environ.get("G56_EDIT_CACHE_DIRS", "").split(os.pathsep) if d]
                self._g56_keys = latent_keys(dirs, remove_dtype_suffix)
                if not mask_dir or not self._g56_keys:
                    raise ValueError(f"mask folder {mask_dir!r}, {len(self._g56_keys)} cached targets in {len(dirs)} folders")
                print(f"g56 edit mask: {len(self._g56_keys)} cached targets in {len(dirs)} cache folders", flush=True)
            except Exception as e:  # noqa: BLE001
                self._g56_mask_off(f"{type(e).__name__}: {str(e)[:120]}")

    parser = hidream_o1_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    trainer = G56MaskedTrainer()
    stopped = False
    try:
        trainer.train(args)
    except _G56Stop:   # G56NF_MUSUBI_MEMBERS: the member rules ended the training (their files and lines are written)
        stopped = True
    finally:
        trainer._g56_mask_write()
        trainer._g56_plin_write()
    if stopped:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)   # no teardown of CUDA / dataloader workers (the member files are on disk)


if __name__ == "__main__":
    main()
