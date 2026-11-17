#!/usr/bin/env python3
"""





  
  
  
  

  

  
  
  

  
  
  

  
  

  
  
  

  
  
  

  
  
  

  
  

  
  
  

  
  
  

  
  

  

  
  

  
  


"""
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path


def snapshot_due(n, now, t_snap, every, first_steps):
    """
    
    
    """
    if first_steps and n == first_steps:
        return True
    return now - t_snap > every


def member_plan(done, k_max, times, window, left):
    """
    
    
    
    """
    t = max(times)
    k_fit = int((0.97 * window - 20.0) // t) if t > 0 else k_max
    if done >= k_max:
        return False, k_fit, "K_max reached"
    if done >= k_fit:
        return False, k_fit, "no time for another member"
    if t > left:
        return False, k_fit, "the next member would end after the deadline"
    return True, k_fit, "fits"


def _status(path, **kw):
    try:
        p = Path(path)
        tmp = p.parent / f".{p.name}.tmp"
        tmp.write_text(json.dumps(dict(kw, t=time.time())))
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001
        pass


# The optimizers whose state the member reset may clear - (top-level package, class) - because their per-parameter state is built
# again from zero at the next step and nothing else of the optimizer carries a member's history: bitsandbytes AdamW8bit (the job's
# own), torch AdamW (tools/newfam_check.py case P: a member after the reset == a fresh run, bit for bit) and torch Adam (the same
# state rules). Any other optimizer - prodigy, for one, keeps its step-size estimate in the param groups - refuses the reset: the
# finished members are shipped and no further one starts. The driver lets only their names through (g56_newfam.CFG_OPTIMIZERS).
RESET_OPTIMIZERS = frozenset({("bitsandbytes", "AdamW8bit"), ("torch", "AdamW"), ("torch", "Adam")})


FILL_SAFETY = 1.03   # G56NF_FILL_RHO: the planning seconds per step = the worst rolling FILL_ROLL-step mean x this
FILL_ROLL = 100


def fill_plan(done, k_max, L, left, s_plan, overhead, rho):
    """
    
    
    
    
    """
    if done >= k_max:
        return False, 0, done, "K_max reached"
    avail = 0.97 * left - 20.0
    t_full = L * s_plan + overhead
    n = max(0, min(k_max - done, int(avail // t_full))) if t_full > 0 else k_max - done
    k_fit = done + n
    if n >= k_max - done:
        return True, L, k_fit, "fits"
    r = rho.get(k_fit)
    if r and s_plan > 0:
        m = n + 1
        lp = min(L, int((avail / m - overhead) / s_plan))
        if lp >= r * L:
            return True, lp, k_fit, f"fill: {m} member(s) of {lp} steps fit ({lp / L:.3f} of {L} >= rho {r:g} at K_fit {k_fit})"
    if n >= 1:
        return True, L, k_fit, "fits"
    return False, 0, k_fit, "no time for another member" + (f" (fill: {n + 1} member(s) would be below rho {r:g} at K_fit {k_fit})" if r else "")


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
PLAN5_SAFETY = 1.03
PLAN5_ROLL = 100
PLAN5_WORST = (25, 20)   # the worst rolling mean: of the rolling means read every 25 steps, the last 20 (500 steps - a slow start ages out)
PLAN5_BRAKE = (2.0, 2.0, 1.0)   # the plan's own reserve before the deadline: planning steps, saves, seconds (the brake: 1.5 steps + 2 saves)
PLAN5_POST_A, PLAN5_POST_X, PLAN5_POLL_S = 15.0, 1.25, 2.0


def plan5_lo(lo, k_fit):
    """The lower bound of a member's length, a fraction of L, at K_fit: a number, or a list read at K_fit (its first entry at K_fit
    <= 1, its last one past its end); an empty list = 1 (no shorter member)."""
    if isinstance(lo, (list, tuple)):
        return float(lo[min(len(lo), max(1, int(k_fit))) - 1]) if lo else 1.0
    return float(lo)


def plan5(done, k_max, L, avail, s_plan, overhead, lo, hi, tie="fill"):
    """
    
    
    
    
    
    
    
    """
    if done >= k_max:
        return False, 0, done, "K_max reached", 0, ""
    t_full = L * s_plan + overhead
    n = max(0, min(k_max - done, int(avail // t_full))) if t_full > 0 else k_max - done
    k_fit = done + n
    cands = []
    if n >= 1 and s_plan > 0:
        ls = min(int(hi * L), int((avail / n - overhead) / s_plan))
        if ls >= L:
            cands.append((n * (ls * s_plan + overhead), tie == "stretch", ls, n, "stretch" if ls > L else "full"))
    m = min(n + 1, k_max - done)
    if m > n and s_plan > 0:
        lf = min(L, int((avail / m - overhead) / s_plan))
        if lf >= 1 and lf >= plan5_lo(lo, k_fit) * L:
            cands.append((m * (lf * s_plan + overhead), tie != "stretch", lf, m, "fill"))
    if not cands:
        if n >= 1:
            return True, L, k_fit, "fits", n, "full"
        return False, 0, k_fit, "no time for another member", 0, ""
    if len(cands) == 2 and abs(cands[0][0] - cands[1][0]) < max(cands[0][3], cands[1][3]) * s_plan:   # a tie: `tie` decides
        cands = [c for c in cands if c[1]]
    _, _, length, members, kind = max(cands)
    return True, length, k_fit, f"{kind}: {members} members of {length} steps", members, kind


def plan5_cost(path):
    """
    """
    try:
        c = json.loads(Path(path).read_text())
        return dict(form_s=float(c.get("form_s") or 0.0), merge_s=float(c.get("merge_s") or 0.0))
    except Exception:  # noqa: BLE001
        return None


def plan5_post(a, b, x, cost, saves, last=1):
    """
    
    
    """
    prior = float(a) + float(b) * saves
    if not cost:
        return prior
    return max(prior, PLAN5_POLL_S + float(x) * (cost["form_s"] * last + cost["merge_s"] * saves))


class Plan5:
    """
    
    
    
    
    
    """

    def __init__(self, p, L):
        self.end, self.khi, self.L = float(p["end"]), max(1, int(p.get("khi") or 1)), int(L)
        self.lo, self.hi = p.get("lo", 1.0), max(1.0, float(p.get("hi") or 1.0))
        self.tie = "stretch" if p.get("tie") == "stretch" else "fill"
        self.safety, self.probe, self.cut = float(p.get("safety") or PLAN5_SAFETY), int(p.get("probe") or 0), float(p.get("cut") or 0.0)
        self.keep = self.cut if 0.0 < self.cut <= 1.0 else 1.0   # the share of its planned length below which the last member is never shortened
        self.a, self.b, self.x = float(p.get("post_a", PLAN5_POST_A)), float(p.get("post_b") or 0.0), float(p.get("post_x") or PLAN5_POST_X)
        self.cost_path, self.saves = p.get("cost"), max(1, int(p.get("saves") or 1))
        self.roll, self.worst, self.mem_dt, self.overs, self.skip, self.n, self.hist, self.first = [], 0.0, 0.0, [], False, 0, [], 0.0
        self.k_plan, self.probed, self.cost, self.info, self.cap, self.last = 1, self.probe <= 0, None, {}, None, 0

    def step(self, dt):
        """one trained step's seconds; the step after a member boundary is left out (its time holds the save and the reset)"""
        if self.skip:
            self.skip = False
            return
        self.mem_dt += dt
        self.roll.append(dt)
        self.n += 1
        if len(self.roll) > PLAN5_ROLL:
            self.roll.pop(0)
        if len(self.roll) == PLAN5_ROLL and (self.n % PLAN5_WORST[0] == 0 or not self.hist):
            self.hist = (self.hist + [self.s_now()])[-PLAN5_WORST[1]:]
            self.worst = max(self.hist)

    def s_now(self):
        """the seconds per step of the last PLAN5_ROLL steps (before that many: of the later half of the steps so far): their mean with
        every step counted at no more than 3 x their median - one long step (a stall) is not the speed, wherever in the window it fell;
        a speed that stays slower moves the median and counts in full. (The runner's own saves are not in the step times: step().)"""
        v = self.roll if len(self.roll) >= PLAN5_ROLL else self.roll[len(self.roll) // 2:]   # (a short history: its later half - not the warm-up)
        if len(v) < 5:
            return sum(v) / len(v) if v else 0.0
        top = 3.0 * sorted(v)[len(v) // 2]
        return sum(min(x, top) for x in v) / len(v)

    def s_plan(self):
        """the planning seconds per step of a plan's last member: the worst rolling mean of the last 500 steps (PLAN5_WORST; before
        PLAN5_ROLL steps: s_now) x the safety factor"""
        return self.safety * (self.worst or self.s_now())

    def post(self, k_total):
        return plan5_post(self.a, self.b, self.x, self.cost, max(1, k_total) * self.saves, self.saves)

    def deadline(self):
        """the runner's deadline of the plan in force: the end minus the post-training work of a file of its members"""
        return self.end - self.post(self.k_plan)

    def tick(self):
        """the driver's costs read again -> the deadline of the plan in force"""
        self.cost = plan5_cost(self.cost_path) or self.cost
        return self.deadline()

    def _terms(self, save_s):
        """
        
        """
        s, se = self.s_plan(), self.s_now()
        r = se / s if s > 0 else 1.0
        lo_s = [x * r for x in self.lo] if isinstance(self.lo, (list, tuple)) else float(self.lo) * r
        over = max(self.overs[-2:]) if self.overs else self.saves * save_s + 0.5
        return s, se, r, lo_s, over, PLAN5_BRAKE[0] * s + PLAN5_BRAKE[1] * save_s + PLAN5_BRAKE[2]

    def _decide(self, done, left, credit, s, se, r, over):
        """
        
        
        
        
        """
        go, length, k_fit, why, members, kind = plan5(done, self.khi, self.L, left + credit * se, se, over, self.lo, self.hi, self.tie)
        if go and self.last:
            alt = plan5(done, self.khi, self.L, left + credit * se, se, over, self.lo, self.hi, "fill" if self.tie == "stretch" else "stretch")
            if alt[0] and abs(alt[1] - self.last) < abs(length - self.last):
                go, length, k_fit, why, members, kind = alt
        cap = None
        if go and members == 1:
            cap = self.L if kind == "fill" else int(self.hi * self.L)
            length = min(cap, int((left + credit * s - over) / s))
            if length < max(1, credit + 1) or length < min(1.0, plan5_lo(self.lo, k_fit)) * r * self.L - max(1.0, 0.01 * self.L):
                go, length, why, members = False, 0, "no time for another member", 0
            else:
                why = f"{kind}: 1 members of {length} steps"
        return go, length, k_fit, why, members, kind, cap

    def _overhead(self, wall):
        """the overhead of the member that took `wall` seconds (the runner's snapshots already taken out): what its steps do not
        explain - its saves, the reset, the step after the boundary; the first member's first step counts as one step of the speed"""
        return max(0.0, wall - self.mem_dt - (max(0.0, self.first - self.s_now()) if not self.overs else 0.0))

    def plan(self, done, now, save_s, credit=0, member_wall=None):
        """
        
        
        """
        if member_wall is not None:
            self.overs.append(self._overhead(member_wall))
            self.mem_dt = 0.0
        self.cost = plan5_cost(self.cost_path) or self.cost
        s, se, r, lo_s, over, brake = self._terms(save_s)
        go, length, k_fit, why, k_guess, members, cap = done < self.khi, self.L, done, "no step time measured yet", done + 1, 1, None
        for _ in range(4 if s > 0 and se > 0 else 0):   # the deadline depends on the members of the plan (the merge of more saves takes longer)
            go, length, k_fit, why, members, kind, cap = self._decide(done, self.end - self.post(k_guess) - now - brake, credit, s, se, r, over)
            if done + members <= k_guess:
                break
            k_guess = done + members
        if go and members == 1:
            self.cap = cap
        self.k_plan = max(1, k_guess if go else done + (1 if credit else 0))
        left = self.deadline() - now
        self.skip = go and not credit
        if go and members >= 2:
            self.last = length   # the length the plan's members have
        self.info = dict(k_plan=self.k_plan, s_plan=round(s, 5), s_now=round(se, 5), overhead=round(over, 2), post=round(self.post(self.k_plan), 1),
                         left=round(left, 1), why=why)
        return go, length, k_fit, (f"{why}; plan5 s/step {se:.4f} (the last member {s:.4f}), overhead {over:.1f} s, left {left:.0f} s, post "
                                   f"{self.post(self.k_plan):.0f} s")

    def rest(self, k, now, save_s, local, length, elapsed):
        """
        
        
        
        
        
        """
        if self.k_plan <= k:
            return None
        s, se, r, lo_s, over, brake = self._terms(save_s)
        if s <= 0 or se <= 0:
            return None
        ahead = self.saves * save_s                       # member k's saves still to come (its window saves counted as if none was taken)
        mine = self._overhead(elapsed) + ahead
        nxt = max(self.overs[-1], mine) if self.overs else mine
        left = self.end - self.post(self.k_plan) - now - brake - (max(0, length - local) * se + ahead)
        if self._decide(k, left, 0, s, se, r, nxt)[0]:
            return None
        self.k_plan, self.cap = k, int(self.hi * self.L)   # the plan ends with member k: the deadline is that file's
        new = min(self.cap, local + int((self.end - self.post(k) - now - brake - ahead) / s))
        return new if new > length else None

    def refit(self, k, now, save_s, local, length, floor):
        """
        
        
        
        
        """
        if self.k_plan > k or self.cap is None:
            return None
        s, se, r, lo_s, over, brake = self._terms(save_s)
        if s <= 0:
            return None
        new = min(self.cap, local + int((self.deadline() - now - brake - (self.saves * save_s + 0.5)) / s))
        new = max(new, int(floor), local + 1)
        return new if abs(new - length) >= 2 else None


PROFILE_TOP = 40
# G56NF_PROFILE: kernel groups by name (the first pattern a kernel's name holds, lower case); everything else is 'other'
PROFILE_GROUPS = (("attention", ("flash", "fmha", "attention", "sdpa", "attn")), ("gemm", ("gemm", "cutlass", "xmma", "cublas", "matmul", "::mm", "addmm", "bmm")),
                  ("conv", ("conv", "winograd", "implicit")), ("copy / cast", ("copy", "cast", "to_copy", "convert")),
                  ("norm", ("norm",)), ("elementwise", ("elementwise", "vectorized", "unrolled", "reduce")))


def profile_group(name):
    n = name.lower()
    return next((g for g, pats in PROFILE_GROUPS if any(p in n for p in pats)), "other")


#
LORA_NOOP_STATS = None   # lora_noop_install's counters; None = not installed (nothing of it runs)
BNB_NOSYNC_VERSION = "0.50.2"


def _is_one(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and v == 1.0


def lora_noop_install(NM):
    """
    
    
    
    
    
    
    
    
    
    
    
    """
    global LORA_NOOP_STATS
    import torch
    orig, qtensor = NM.ToolkitModuleMixin.forward, NM.QTensor
    st = dict(fast=0, original={}, at_skip=0, multiplier_reads=0, modules=set(), seen=[None, False])

    def why(self, x):
        if type(self).__name__ != "LoRAModule":
            return "class"
        net = self.network_ref()
        if net is None:
            return "network"
        if net.is_lorm:
            return "lorm"
        m = net._multiplier
        num = not isinstance(m, bool) and isinstance(m, (int, float))
        if not (num or (isinstance(m, (list, tuple)) and len(m) > 0
                        and all(not isinstance(v, bool) and isinstance(v, (int, float)) for v in m))):
            return "multiplier"
        if not net.is_active or net.is_merged_in or (num and m == 0):
            return ""
        if not (_is_one(m) if num else all(_is_one(v) for v in m)):
            return "multiplier"
        t = net.torch_multiplier
        if not (isinstance(t, torch.Tensor) and t.dim() == 1 and t.numel() > 0 and t.is_floating_point()):
            return "multiplier tensor"
        if st["seen"][0] is not t:   # (the reference held: no other tensor can take its id)
            st["seen"] = [t, bool((t == 1.0).all())]
            st["multiplier_reads"] += 1
        if not st["seen"][1]:
            return "multiplier tensor"
        if type(self.scale) is not float or self.scale != 1.0:
            return "scale"
        if self.dropout is not None:
            return "dropout"
        if self.module_dropout is not None:
            return "module dropout"
        if self.rank_dropout:
            return "rank dropout"
        if hasattr(self, "scalar"):
            return "scalar"
        if getattr(self, "lora_mid", None) is not None:
            return "lora_mid"
        if x.dim() < 2 or x.shape[0] < t.shape[0] or x.shape[0] % t.shape[0]:
            return "batch"
        return None

    def forward(self, x, *args, **kwargs):
        try:
            w = why(self, x)
        except Exception:  # noqa: BLE001  (a guard that cannot be read = AI Toolkit's own forward)
            w = "guard error"
        if w is None:   # AI Toolkit's operations in its order, without the two multiplications by 1.0
            st["fast"] += 1
            st["modules"].add(id(self))
            org = self.org_forward(x, *args, **kwargs)
            xi = x.dequantize() if isinstance(x, qtensor) else x
            return org + self.lora_up(self.lora_down(xi.to(self.lora_down.weight.dtype))).to(org.dtype)
        if w:
            st["original"][w] = st["original"].get(w, 0) + 1
        else:
            st["at_skip"] += 1
        return orig(self, x, *args, **kwargs)

    NM.ToolkitModuleMixin.forward = forward
    LORA_NOOP_STATS = st
    return st


def lora_noop_stats():
    """G56NF_LORA_NOOP: the counters as the status holds them (None = not installed)."""
    st = LORA_NOOP_STATS
    if st is None:
        return None
    return dict(fast=st["fast"], modules=len(st["modules"]), original=dict(st["original"]), at_skip=st["at_skip"],
                multiplier_reads=st["multiplier_reads"])


def bnb_nosync_install(opt, BO=None, version=None, stream=None):
    """
    
    
    
    
    
    
    
    
    
    """
    st = dict(applied=False, why="", skipped=0, passed=0, tensors=0)
    try:
        import inspect
        opt = getattr(opt, "optimizer", opt)
        st["tensors"] = sum(len(g["params"]) for g in opt.param_groups)
        kind = f"{type(opt).__module__}.{type(opt).__name__}"
        if not type(opt).__module__.startswith("bitsandbytes."):
            st["why"] = f"the optimizer is {kind}, not bitsandbytes"
            return st
        if BO is None:
            import bitsandbytes
            import bitsandbytes.optim.optimizer as BO
            version = getattr(bitsandbytes, "__version__", "?")
        if not isinstance(opt, BO.Optimizer8bit):
            st["why"] = f"{kind} is not a bitsandbytes Optimizer8bit"
        elif getattr(opt, "is_paged", False):
            st["why"] = f"{kind} is paged (its synchronizations order the page transfers)"
        elif version != BNB_NOSYNC_VERSION:
            st["why"] = f"bitsandbytes {version}, not {BNB_NOSYNC_VERSION}"
        elif not callable(getattr(BO, "sync_gpu", None)) or "sync_gpu(p)" not in inspect.getsource(BO.Optimizer8bit.step):
            st["why"] = "Optimizer8bit.step does not call this module's sync_gpu(p)"
        else:
            orig = BO.sync_gpu
            if stream is None:
                import torch

                def stream(dev):
                    return torch._C._cuda_getCurrentRawStream(dev.index if dev.index is not None else torch.cuda.current_device())

            def sync_gpu(t):
                try:
                    skip = t.device.type == "cuda" and stream(t.device) == 0
                except Exception:  # noqa: BLE001  (a stream that cannot be read: bitsandbytes' own synchronization)
                    skip = False
                if skip:
                    st["skipped"] += 1
                    return None
                st["passed"] += 1
                return orig(t)
            BO.sync_gpu = sync_gpu
            st["applied"], st["why"] = True, f"{kind}, bitsandbytes {version}, not paged"
    except Exception as e:  # noqa: BLE001  (a speed knob never stops the training: bitsandbytes' own code stays)
        st["why"] = f"{type(e).__name__}: {str(e)[:120]}"
    return st


def tail_window(fracs, L):
    """
    
    """
    out = set()
    for f in fracs:
        q = float(f) * L
        n = int(q)
        s = max(1, n + 1 if q - n > 0.5 + 1e-9 else n)
        if s < L:
            out.add(s)
    return sorted(out)


class _G56Stage:
    """G56NF_PROFILE: one named stage of AI Toolkit's timer during the profiled steps - CUDA synchronized at its start and end, its
    seconds added to `stages`, a profiler range of its name; the original timer keeps its own record."""

    def __init__(self, orig, stages, name):
        self.orig, self.stages, self.name, self.rf, self.t = orig, stages, name, None, 0.0

    def __enter__(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.rf = torch.profiler.record_function(f"g56:{self.name}")
        self.rf.__enter__()
        self.orig.start(self.name)
        self.t = time.perf_counter()
        return self

    def __exit__(self, et, ev, tb):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        s, n = self.stages.get(self.name, (0.0, 0))
        self.stages[self.name] = (s + time.perf_counter() - self.t, n + 1)
        self.rf.__exit__(et, ev, tb)
        (self.orig.stop if et is None else self.orig.cancel)(self.name)
        return False


class _G56NullTimer:
    """the original-timer side of a _G56Stage outside AI Toolkit's timer (the VAE's encode)"""

    def start(self, name):
        pass

    def stop(self, name):
        pass

    def cancel(self, name):
        pass


class _G56SyncTimer:
    """G56NF_PROFILE: stands in for AI Toolkit's Timer (toolkit/timer.py) during the profiled steps: `with timer(name)` = a _G56Stage
    (nesting allowed); start / stop / print / reset / anything else -> the original timer."""

    def __init__(self, orig, stages):
        self._orig, self._stages = orig, stages

    def __call__(self, name):
        return _G56Stage(self._orig, self._stages, name)

    def __getattr__(self, name):
        return getattr(self._orig, name)


class _G56Stop(Exception):
    """end_step_hook's stop at the step cap / the deadline: job.run() is left through it."""


def tf32_state():
    """."""
    try:
        import torch
        return dict(matmul=bool(torch.backends.cuda.matmul.allow_tf32), cudnn=bool(torch.backends.cudnn.allow_tf32))
    except Exception as e:  # noqa: BLE001
        return dict(error=f"{type(e).__name__}: {str(e)[:80]}")


def is_cuda_oom(e):
    """."""
    try:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return isinstance(e, RuntimeError) and "CUDA out of memory" in str(e)


def gc_switch_on(model):
    """Gradient checkpointing ON in place with the switch AI Toolkit applies at load when train.gradient_checkpointing is set
    (BaseSDTrainProcess.py:1841-1846); the models read the flag on every forward (flux2 src/model.py:178/203, moss_image
    src/transformer.py:282, diffusers models). -> True when a switch was found."""
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        return True
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
        return True
    return False


PHI_INV = (5 ** 0.5 - 1) / 2   # G56NF_TSTRAT: the additive recurrence of the golden ratio - the lowest discrepancy of its kind
TSTRAT_OWN_TYPES = frozenset({"one_step", "two_step", "four_step", "eight_step", "next_sample"})   # draws that are not the uniform one


SDPA_BACKENDS = {"cudnn": "CUDNN_ATTENTION"}   # G56NF_SDPA_BACKEND's values -> the torch.nn.attention.SDPBackend put first


def sdpa_install(name, F=None):
    """
    
    
    
    
    
    
    """
    st = dict(applied=False, why="", want=name, order=None, calls={}, first=None)
    try:
        import torch
        from torch.nn.attention import SDPBackend
        if F is None:
            import torch.nn.functional as F
        be = int(getattr(SDPBackend, SDPA_BACKENDS[name]))
        names = {int(v): k.lower() for k, v in SDPBackend.__members__.items()}
        own = [int(x) for x in torch._C._get_sdp_priority_order()]
        orig = F.scaled_dot_product_attention
        prev_cudnn = torch.backends.cuda.cudnn_sdp_enabled()
        try:
            if name == "cudnn":
                torch.backends.cuda.enable_cudnn_sdp(True)
            torch._C._set_sdp_priority_order([be] + [x for x in own if x != be])
        except Exception:
            torch.backends.cuda.enable_cudnn_sdp(prev_cudnn)
            torch._C._set_sdp_priority_order(own)
            raise

        def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            try:
                c = names.get(int(torch._fused_sdp_choice(query, key, value, attn_mask, dropout_p, is_causal, scale=scale,
                                                          enable_gqa=enable_gqa)), "unknown")
            except Exception:  # noqa: BLE001  (the count only: the call itself goes on)
                c = "unknown"
            st["calls"][c] = st["calls"].get(c, 0) + 1
            if st["first"] is None:
                st["first"] = dict(shape=list(query.shape), dtype=str(query.dtype), device=str(query.device), backend=c)
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                        enable_gqa=enable_gqa)
        sdpa.g56_orig = orig
        F.scaled_dot_product_attention = sdpa
        st["order"] = [names.get(int(x), str(x)) for x in torch._C._get_sdp_priority_order()]
        st["applied"], st["why"] = True, f"{name} first in torch's priority order, its switch on"
    except Exception as e:
        st["why"] = f"{type(e).__name__}: {str(e)[:120]}"
    return st


def gc_step_tokens(args, kwargs):
    """
    
    
    
    
    
    """
    hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
    ctx = args[2] if len(args) > 2 else kwargs.get("encoder_hidden_states")
    refs = args[5] if len(args) > 5 else kwargs.get("ref_latents")
    if refs is not None and any(len(r) > 0 for r in refs):
        return None
    if not (hasattr(hs, "shape") and hasattr(ctx, "shape") and len(hs.shape) == 4 and len(ctx.shape) == 3):
        return None
    b, _, h, w = (int(x) for x in hs.shape)
    return b * (h * w + int(ctx.shape[1]))


def gc_step_install(model, fence, trainer):
    """
    
    
    
    
    
    
    
    
    """
    st = dict(applied=False, why="", fence=int(fence), off=0, on=0, unknown=0, held=0, n_off_max=None, n_on_min=None, n_max=None,
              hold=None)
    try:
        import torch
        if not isinstance(getattr(model, "gradient_checkpointing", None), bool) or not hasattr(model, "register_forward_pre_hook"):
            st["why"] = f"{type(model).__name__} holds no gradient_checkpointing flag"
            return st

        def hold(why):
            if st["hold"] is None:
                st["hold"] = why
                print(f"g56 gc step fence: checkpointing held ON from here ({why})", flush=True)

        def pre(mod, args, kwargs):
            try:
                if not torch.is_grad_enabled():
                    return None
                if st["hold"] is None and getattr(trainer, "_g56_oom", 0):
                    at = getattr(trainer, "_g56_gc_at", None)
                    hold("an OOM in this child" + (f"; the L1 guard switched checkpointing on at step {at}" if at else ""))
                n = None if st["hold"] is not None else gc_step_tokens(args, kwargs)
                on = st["hold"] is not None or n is None or n > st["fence"]
                mod.gradient_checkpointing = on
                trainer.train_config.gradient_checkpointing = on
                if st["hold"] is not None:
                    st["held"] += 1
                elif n is None:
                    st["unknown"] += 1
                elif on:
                    st["on"] += 1
                    st["n_on_min"] = n if st["n_on_min"] is None else min(st["n_on_min"], n)
                else:
                    st["off"] += 1
                    st["n_off_max"] = n if st["n_off_max"] is None else max(st["n_off_max"], n)
                if n is not None:
                    st["n_max"] = n if st["n_max"] is None else max(st["n_max"], n)
            except Exception as e:  # noqa: BLE001  (never the forward: checkpointing on, held to the end)
                try:
                    mod.gradient_checkpointing = True
                    trainer.train_config.gradient_checkpointing = True
                except Exception:  # noqa: BLE001
                    pass
                st["held"] += 1
                try:
                    hold(f"{type(e).__name__}: {str(e)[:100]}")
                except Exception:  # noqa: BLE001
                    pass
            return None

        model.register_forward_pre_hook(pre, with_kwargs=True)
        st["applied"], st["why"] = True, f"a forward pre-hook on {type(model).__name__}"
    except Exception as e:
        st["why"] = f"{type(e).__name__}: {str(e)[:120]}"
    return st


def freeze_install(model, network, k):
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    st = dict(applied=False, why="", k=int(k), blocks=None, frozen_modules=0, kept_modules=0, frozen_params=0, refrozen=0,
              out_grad=None, calls=None)
    try:
        import torch
        blocks = getattr(model, "transformer_blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList):
            raise ValueError(f"no ModuleList transformer_blocks on {type(model).__name__}")
        st["blocks"] = len(blocks)
        if not 0 < int(k) < len(blocks):
            raise ValueError(f"k {k} is not in 1..{len(blocks) - 1}")
        early = {id(m) for b in blocks[:int(k)] for m in b.modules()}
        frozen, kept = [], []
        for m in network.get_all_modules():
            down, up = getattr(m, "lora_down", None), getattr(m, "lora_up", None)
            if not (hasattr(down, "weight") and hasattr(up, "weight")) or getattr(up, "bias", None) is not None:
                raise ValueError(f"a network module that is not lora_down / lora_up weights only ({type(m).__name__})")
            org = (getattr(m, "org_module", None) or [None])[0]
            (frozen if org is not None and id(org) in early else kept).append(m)
        if not frozen or not kept:
            raise ValueError(f"the split leaves {len(frozen)} frozen and {len(kept)} trained modules")
        params = [w for m in frozen for w in (m.lora_down.weight, m.lora_up.weight)]
        rec = dict(grad=[False] * len(blocks), calls=[0] * len(blocks))

        def pre_for(i):   # a PRE-hook: torch's checkpoint stops a recomputation early, before a block's own post-hooks would run
            def pre(mod, args):
                rec["calls"][i] += 1
            return pre

        def post_for(i):
            def post(mod, args, out):
                outs = out if isinstance(out, (tuple, list)) else (out,)
                rec["grad"][i] = rec["grad"][i] or any(bool(getattr(o, "requires_grad", False)) for o in outs)
            return post
        hooks = [b.register_forward_pre_hook(pre_for(i)) for i, b in enumerate(blocks)]
        hooks += [b.register_forward_hook(post_for(i)) for i, b in enumerate(blocks)]
        for w in params:
            w.requires_grad_(False)
        st.update(applied=True, frozen_modules=len(frozen), kept_modules=len(kept), frozen_params=len(params))
        return st, params, hooks, rec, frozen, kept
    except Exception as e:
        st["why"] = f"{type(e).__name__}: {str(e)[:160]}"
        return st, [], [], None, [], []


def freeze_status(st, rec, frozen, kept):
    """G56NF_FREEZE_BLOCKS: the status entry 'freeze' - freeze_install's dict, the first step's block record (out_grad = the blocks
    whose output carried a gradient, calls = how often each block ran) and the largest absolute lora_up value of the frozen and of
    the trained modules now (the frozen ones must still be 0: their delta is zero)."""
    out = dict(st)
    if out.get("out_grad") is None and rec is not None:   # the first step's record, read while its hooks are still on
        out.update(out_grad=[i for i, g in enumerate(rec["grad"]) if g], calls=list(rec["calls"]))
    try:
        out["frozen_up_max"] = max((float(m.lora_up.weight.detach().abs().max()) for m in frozen), default=None)
        out["kept_up_max"] = max((float(m.lora_up.weight.detach().abs().max()) for m in kept), default=None)
    except Exception as e:  # noqa: BLE001
        out["up_error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


class G56StepRules:
    """
    
    
      
      
    
      
      
      
    
    
      
    
      
      
      
      
    
      
      
      
    
      
      
      
      
      
    
      
      
    
    
    """

    def _g56_setup(self, spec, status):
        self._g56_status = status
        self._g56_deadline = float(spec["deadline"])
        self._g56_every = float(spec.get("snapshot_every_s", 600.0))
        self._g56_t_last = None
        self._g56_t_snap = time.time()
        self._g56_dts = []
        self._g56_save_s = 10.0
        self._g56_n = 0
        self._g56_max_steps = int(spec.get("max_steps") or 0)
        self._g56_first = int(spec.get("first_snapshot_steps") or 0)
        self._g56_oom = 0
        self._g56_oom_step = False
        self._g56_gc_at = None
        self._g56_ballast_gb = float(spec.get("test_ballast_gb") or 0.0)
        self._g56_crash_at = int(spec.get("test_crash_at_step") or 0)
        self._g56_ballast = None
        self._g56_keep = frozenset(int(x) for x in (spec.get("keep_steps") or ()))
        self._g56_members = max(1, int(spec.get("members") or 1))   # K_max; 1 = the single run
        self._g56_seed = int(spec.get("seed", 42))
        self._g56_member = 1            # the member being trained
        self._g56_member_step0 = 0      # step_num at its start
        self._g56_member_t0 = None      # wall time of its start (member 1: the start of the first training step)
        self._g56_t_train0 = None
        self._g56_member_s = []
        self._g56_member_log = []       # one line per finished member (status.json: the driver copies them to its log)
        self._g56_len = self._g56_max_steps   # the length of the member being trained (G56NF_FILL_RHO may shorten the later ones)
        self._g56_fill_setup(spec)
        self._g56_p5_setup(spec)
        self._g56_mask_setup(spec)
        self._g56_cfg_setup(spec)
        self._g56_strata_setup(spec)
        self._g56_tstrat_setup(spec)
        self._g56_profile_setup(spec)
        self._g56_nosync = dict(applied=None) if spec.get("bnb_nosync") else None
        self._g56_tail = sorted(float(f) for f in spec.get("tail_fracs") or ()) or None
        self._g56_gcs_fence = int(spec.get("gc_step_fence") or 0)
        self._g56_gcs = None   # its decisions (gc_step_install, at the first step)
        self._g56_sdpa_want = spec.get("sdpa_backend") or None
        self._g56_tf32 = bool(spec.get("tf32"))
        self._g56_sdpa = None   # its state (sdpa_install, at the first step)
        self._g56_freeze_k = int(spec.get("freeze_blocks") or 0)
        self._g56_freeze = None   # its state (freeze_install, at the first step)

    def _g56_fill_setup(self, spec):
        """
        """
        f = spec.get("fill_rho") or {}
        self._g56_fill = {int(k): float(v) for k, v in f.items()} if f and int(spec.get("members") or 1) > 1 else None
        self._g56_roll, self._g56_roll_worst, self._g56_mem_dt, self._g56_fill_over, self._g56_fill_skip = [], 0.0, 0.0, 0.0, False

    def _g56_fill_step(self, dt):
        """G56NF_FILL_RHO: one trained step's seconds. The step after a member boundary is left out (its time holds the save and the
        reset: the overhead)."""
        if self._g56_fill_skip:
            self._g56_fill_skip = False
            return
        self._g56_mem_dt += dt
        self._g56_roll.append(dt)
        if len(self._g56_roll) > FILL_ROLL:
            self._g56_roll.pop(0)
        if len(self._g56_roll) == FILL_ROLL:
            self._g56_roll_worst = max(self._g56_roll_worst, sum(self._g56_roll) / FILL_ROLL)

    def _g56_fill_decide(self, k, now):
        """G56NF_FILL_RHO: fill_plan at the worst rolling mean seen so far (before FILL_ROLL steps: the mean so far) x FILL_SAFETY and
        the largest overhead of a finished member (its wall time beyond the sum of its step times). -> (start, length, K_fit, why)"""
        roll = self._g56_roll_worst or (sum(self._g56_roll) / len(self._g56_roll) if self._g56_roll else 0.0)
        s_plan = FILL_SAFETY * roll
        self._g56_fill_over = max(self._g56_fill_over, self._g56_member_s[-1] - self._g56_mem_dt)
        self._g56_mem_dt = 0.0
        left = self._g56_deadline - now
        go, length, k_fit, why = fill_plan(k, self._g56_members, self._g56_max_steps, left, s_plan, self._g56_fill_over, self._g56_fill)
        self._g56_fill_skip = go
        return go, length, k_fit, (f"{why}; plan s/step {s_plan:.4f} (worst rolling {FILL_ROLL}-step mean x {FILL_SAFETY:g}), overhead "
                                   f"{self._g56_fill_over:.1f} s, left {left:.0f} s")

    def _g56_p5_setup(self, spec):
        """
        
        """
        p = spec.get("plan5")
        self._g56_p5 = Plan5(p, self._g56_max_steps) if p and self._g56_max_steps else None
        self._g56_save_t = 0.0   # the seconds of the last save (_g56_save)
        self._g56_p5_said = -10 ** 9   # the last length a refit line named
        self._g56_len0 = self._g56_len   # the length the member being trained was planned at (PLAN5's cut and refit are shares of it)
        if self._g56_p5 is not None:
            self._g56_fill = None
            self._g56_members = self._g56_p5.khi
            self._g56_deadline = self._g56_p5.deadline()

    def _g56_p5_line(self, line, phase, step_s):
        print("g56 " + line, flush=True)
        self._g56_member_log.append(line)
        _status(self._g56_status, phase=phase, step=int(self.step_num), step_s=step_s, save_s=self._g56_save_s, **self._g56_extra())

    def _g56_p5_probe(self, now, local, step_s):
        """
        
        """
        p = self._g56_p5
        p.probed = True
        go, length, k_fit, why = p.plan(0, now, self._g56_save_t or self._g56_save_s, credit=local)
        if go:
            self._g56_len = self._g56_len0 = max(int(length), local + 1)
        self._g56_deadline = p.deadline()
        self._g56_p5_line(f"plan5 probe at step {local} of member 1: length {self._g56_len}, K_fit {k_fit}, K planned {p.k_plan} ({why})",
                          "train", step_s)

    def _g56_p5_cut(self, k, local, step_s):
        """PLAN5 'cut': the deadline stops member k at `local` of its planned steps; far enough (>= cut x its length) -> it is saved as
        a member (its own step count in the header) and the job ends; -> False when it is not far enough (no file, as before)."""
        p = self._g56_p5
        if not p.cut > 0 or local < p.cut * self._g56_len0:   # (the length it was planned at, not one a refit gave it)
            return False
        self._g56_save(f"member_{k:02d}.safetensors", "member", step=local, meta={"g56_member": str(k), "g56_len": str(int(self._g56_len))})
        self._g56_member_s.append(time.time() - self._g56_member_t0)
        self._g56_p5_line(f"member {k}/{self._g56_members} cut by the deadline at step {local} of {self._g56_len}: kept, at least "
                          f"{p.cut:g} of its length", "done", step_s)
        return True

    def _g56_profile_setup(self, spec):
        """spec['profile'] = [a, b] (G56NF_PROFILE, a measuring tool; absent = None = no code of this runs): the training steps a..b
        (1-based) are profiled."""
        p = spec.get("profile")
        self._g56_prof = None if not p else dict(a=int(p[0]), b=int(p[1]), on=None, timer=None, vae=None, stages={}, t0=0.0, done=None)

    def _g56_profile_start(self):
        """
        """
        pf = self._g56_prof
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile
            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
            if getattr(self, "timer", None) is not None:
                pf["timer"] = self.timer
                self.timer = _G56SyncTimer(pf["timer"], pf["stages"])
            vae = getattr(getattr(self, "sd", None), "vae", None)
            if vae is not None and callable(getattr(vae, "encode", None)) and "encode" not in vars(vae):
                enc, stages = vae.encode, pf["stages"]

                def encode(*a, **k):
                    with _G56Stage(_G56NullTimer(), stages, "vae_encode"):
                        return enc(*a, **k)
                vae.encode = encode
                pf["vae"] = vae
            pf["on"] = profile(activities=acts)
            pf["on"].__enter__()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pf["t0"] = time.perf_counter()
        except Exception as e:  # noqa: BLE001  (a measuring tool never stops the training)
            self._g56_profile_restore()
            pf["done"] = dict(steps=[pf["a"], pf["b"]], error=f"{type(e).__name__}: {str(e)[:160]}")
            pf["on"] = None
            print(f"g56 profile: not started ({pf['done']['error']})", flush=True)

    def _g56_profile_restore(self):
        pf = self._g56_prof
        if pf.get("timer") is not None:
            self.timer, pf["timer"] = pf["timer"], None
        if pf.get("vae") is not None:
            try:
                del pf["vae"].encode
            except Exception:  # noqa: BLE001
                pass
            pf["vae"] = None

    def _g56_profile_stop(self):
        """G56NF_PROFILE, the end of step b: the profiler off, the timer and the VAE back; the 40 heaviest kernels by device time (on a
        CPU-only machine: CPU operators by their own time), the sums by kernel group, the seconds per stage -> <save_root>/
        profile_s<a>-<b>.json + the status entry 'profile' + one line. Never raises."""
        import torch
        pf = self._g56_prof
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.perf_counter() - pf["t0"]
            pf["on"].__exit__(None, None, None)
            self._g56_profile_restore()
            n = pf["b"] - pf["a"] + 1
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            rows = []
            for evt in pf["on"].key_averages():
                if dev == "cuda":
                    if "cuda" not in str(getattr(evt, "device_type", "")).lower():
                        continue
                    us = float(getattr(evt, "self_device_time_total", 0.0) or getattr(evt, "self_cuda_time_total", 0.0) or 0.0)
                else:
                    if str(evt.key).startswith("g56:"):
                        continue
                    us = float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)
                rows.append((str(evt.key), int(getattr(evt, "count", 0)), us / 1000.0))
            rows.sort(key=lambda r: -r[2])
            total = sum(r[2] for r in rows)
            groups = {}
            for name, _, ms in rows:
                g = profile_group(name)
                groups[g] = groups.get(g, 0.0) + ms
            out = dict(steps=[pf["a"], pf["b"]], device=dev, wall_s=round(wall, 3), wall_s_per_step=round(wall / n, 4),
                       kernel_ms_per_step=round(total / n, 2),
                       stages={k: [round(s, 4), c] for k, (s, c) in sorted(pf["stages"].items(), key=lambda x: -x[1][0])},
                       groups={k: round(v, 2) for k, v in sorted(groups.items(), key=lambda x: -x[1])},
                       top=[[name, c, round(ms, 3)] for name, c, ms in rows[:PROFILE_TOP]])
            path = Path(self.save_root) / f"profile_s{pf['a']}-{pf['b']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=1))
            pf["done"] = dict(out, top=out["top"][:3], file=str(path))
        except Exception as e:  # noqa: BLE001
            self._g56_profile_restore()
            pf["done"] = dict(steps=[pf["a"], pf["b"]], error=f"{type(e).__name__}: {str(e)[:160]}")
        pf["on"] = None
        print(f"g56 profile: {json.dumps(pf['done'])[:600]}", flush=True)

    def _g56_extra(self):
        ex = {"oom": self._g56_oom, "gc_on_at": self._g56_gc_at} if self._g56_oom else {}
        if self._g56_members > 1 or self._g56_p5 is not None:   # every status write of a K-member run carries the members so far (the driver reads them at its polls)
            ex.update(members_done=len(self._g56_member_s), member_log=self._g56_member_log)
        if self._g56_p5 is not None:   # PLAN5: the plan in force, the first training step's time and the deadline (the driver's utilization line)
            ex.update(plan5=dict(self._g56_p5.info, t_train0=self._g56_t_train0, deadline=self._g56_deadline, member=self._g56_member,
                                 length=int(self._g56_len or 0), member_s=[round(x, 1) for x in self._g56_member_s]))
        if self._g56_mask is not None:
            ex.update(mask=self._g56_mask_stats())
        if self._g56_cfg is not None:    # a 'configuration only' knob is on: what was trained on so far
            ex.update(cfg=self._g56_cfg_stats())
        if self._g56_strata is not None:
            ex.update(strata=self._g56_strata_stats())
        if getattr(self, "_g56_tstrat", None) is not None:
            ex.update(tstrat=self._g56_tstrat_stats())
        if self._g56_prof is not None and self._g56_prof.get("done") is not None:   # G56NF_PROFILE (a measuring tool)
            ex.update(profile=self._g56_prof["done"])
        if LORA_NOOP_STATS is not None:
            ex.update(lora_noop=lora_noop_stats())
        if self._g56_nosync is not None:
            ex.update(bnb_nosync=dict(self._g56_nosync, steps=self._g56_n))
        if getattr(self, "_g56_gcs", None) is not None:
            ex.update(gc_step=dict(self._g56_gcs))
        if getattr(self, "_g56_sdpa", None) is not None:
            ex.update(sdpa=dict(self._g56_sdpa, calls=dict(self._g56_sdpa["calls"])))
        if getattr(self, "_g56_tf32", False):
            ex.update(tf32=tf32_state())
        if getattr(self, "_g56_freeze", None) is not None:
            ex.update(freeze=freeze_status(self._g56_freeze, self._g56_freeze_rec, *self._g56_freeze_mods))
        return ex

    def _g56_freeze_step(self):
        """
        
        """
        if self._g56_freeze is None:
            m = getattr(self.sd, "unet_unwrapped", None)
            st, params, hooks, rec, frozen, kept = freeze_install(self.sd.unet if m is None else m, self.network, self._g56_freeze_k)
            self._g56_freeze, self._g56_freeze_params, self._g56_freeze_hooks = st, params, hooks
            self._g56_freeze_rec, self._g56_freeze_mods = rec, (frozen, kept)
            what = st["why"] or f"{st['frozen_modules']} of {st['frozen_modules'] + st['kept_modules']} LoRA modules frozen"
            print(f"g56 freeze blocks 0..{self._g56_freeze_k - 1}: {'applied' if st['applied'] else 'not applied'} ({what})", flush=True)
            return
        n = 0
        for w in self._g56_freeze_params:
            if w.requires_grad:
                w.requires_grad_(False)
                n += 1
        self._g56_freeze["refrozen"] += n
        if self._g56_freeze_hooks:   # the first step ran: its block record is kept, the hooks go
            rec = self._g56_freeze_rec
            self._g56_freeze.update(out_grad=[i for i, g in enumerate(rec["grad"]) if g], calls=list(rec["calls"]))
            for h in self._g56_freeze_hooks:
                h.remove()
            self._g56_freeze_hooks = []

    def _g56_mask_setup(self, spec):
        """."""
        self._g56_mask_dir = spec.get("edit_mask_dir") or None
        self._g56_mask = None if self._g56_mask_dir is None else dict(cache={}, steps=0, attached=0, weighted=0, mass=[], off=None)

    def _g56_mask_off(self, why):
        """The weights are switched off for the rest of the run (the unweighted loss), once, with a line: never the training."""
        self._g56_mask_dir = None
        if self._g56_mask["off"] is None:
            self._g56_mask["off"] = why
            print(f"g56 edit mask: OFF from step {self._g56_mask['steps']} ({why}): the unweighted loss", flush=True)

    def _g56_mask_stats(self):
        m = self._g56_mask
        mass = sorted(m["mass"])
        return dict(steps=m["steps"], attached=m["attached"], weighted=m["weighted"], items=len(m["cache"]),
                    masked=sum(1 for w in m["cache"].values() if w is not None),
                    mass_median=mass[len(mass) // 2] if mass else None, off=m["off"])

    def _g56_attach_masks(self, batch_list):
        """Each item's changed-region loss weights on its batch, as the evaluator attaches them (objective_evaluator.py:111-128): the
        pair's mask scaled and cropped like the item's image, weights = floor + mask x (1 - floor) with the floor of the CROPPED
        mask's coverage, a [1, 1, H, W] tensor in batch.mask_tensor. An item without a mask file (an absent or a global edit) and a
        flipped item stay unweighted, as the evaluator scores them; a batch of several items too (AI Toolkit normalizes the weights
        over the whole batch; the evaluator scores one item per batch and our job trains one)."""
        try:
            import torch
            import g56_editmask as EM
            for batch in (batch_list if isinstance(batch_list, list) else [batch_list]):
                self._g56_mask["steps"] += 1
                items = batch.file_items
                if len(items) != 1:
                    continue
                it = items[0]
                key = (Path(it.path).stem, int(it.scale_to_width), int(it.scale_to_height), int(it.crop_x), int(it.crop_y),
                       int(it.crop_width), int(it.crop_height))
                cache = self._g56_mask["cache"]
                if key not in cache:
                    f = Path(self._g56_mask_dir) / f"{key[0]}.png"
                    cache[key] = None
                    if f.is_file() and not (it.flip_x or it.flip_y):
                        weights, values = EM.item_weights(f, key[1:3], (key[3], key[4], key[3] + key[5], key[4] + key[6]))
                        cache[key] = torch.from_numpy(weights).unsqueeze(0).unsqueeze(0)
                        self._g56_mask["mass"].append(EM.changed_mass(weights, values))
                if cache[key] is not None:
                    batch.mask_tensor = cache[key]
                    self._g56_mask["attached"] += 1
        except Exception as e:
            self._g56_mask_off(f"{type(e).__name__}: {str(e)[:120]}")

    def _g56_mask_at_loss(self, kwargs):
        """The evaluator's own check (objective_evaluator.py:83-87) at our loss: the weights AI Toolkit made of batch.mask_tensor must
        be at the prediction's spatial size. They are -> counted. They are not -> this loss and the rest of the run are unweighted
        (mask_multiplier 1.0), said once: AI Toolkit would swallow the mismatch (SDTrainer.py:1013-1023) or broadcast it wrongly."""
        try:
            batch, pred, mult = kwargs.get("batch"), kwargs.get("noise_pred"), kwargs.get("mask_multiplier")
            if batch is None or getattr(batch, "mask_tensor", None) is None:
                return
            ms, ps = getattr(mult, "shape", None), getattr(pred, "shape", None)
            if ms is not None and ps is not None and len(ms) == len(ps) and tuple(ms[-2:]) == tuple(ps[-2:]) and ms[0] == ps[0]:
                self._g56_mask["weighted"] += 1
                return
            kwargs["mask_multiplier"] = 1.0
            self._g56_mask_off(f"weights {None if ms is None else tuple(ms)} against a prediction {None if ps is None else tuple(ps)}")
        except Exception as e:  # noqa: BLE001
            kwargs["mask_multiplier"] = 1.0
            self._g56_mask_off(f"{type(e).__name__}: {str(e)[:120]}")

    def calculate_loss(self, *args, **kwargs):
        if getattr(self, "_g56_mask_dir", None) is not None:   # G56NF_EDIT_MASK; AI Toolkit passes keywords (SDTrainer.py:2273-2281)
            self._g56_mask_at_loss(kwargs)
        return super().calculate_loss(*args, **kwargs)

    def _g56_cfg_setup(self, spec):
        """
        
        """
        self._g56_cfg = (dict(steps=0, batches=0, items=0, mirrored=0, by_dropout={}, pending=None, held=None, error=None)
                         if spec.get("cfg_watch") else None)

    def _g56_cfg_held(self):
        """What AI Toolkit's own objects hold (each value read on its own: one that cannot be read is None): the optimizer's
        class as the member reset reads it and the weight decay of its param groups, the ranks of the network's modules, the
        gradient clip, the timestep draw, accumulation and batch size of the train config."""
        def get(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                return None
        opt, tc = get(lambda: getattr(self.optimizer, "optimizer", self.optimizer)), self.train_config
        return dict(optimizer=get(lambda: f"{type(opt).__module__.split('.')[0]} {type(opt).__name__}") if opt is not None else None,
                    weight_decay=get(lambda: sorted({float(g["weight_decay"]) for g in opt.param_groups})),
                    rank=get(lambda: sorted({int(m.lora_down.weight.shape[0]) for m in self.network.get_all_modules()})),
                    max_grad_norm=get(lambda: float(tc.max_grad_norm)), content_or_style=get(lambda: str(tc.content_or_style)),
                    gradient_accumulation=get(lambda: int(tc.gradient_accumulation)), batch_size=get(lambda: int(tc.batch_size)))

    def _g56_cfg_count(self, batch_list):
        """This step's batches and items - how many, how many mirrored, how many per caption-dropout entry - kept aside until the
        step is known to have trained (_g56_cfg_commit: an OOM step counts nothing). Never stops the training."""
        c = self._g56_cfg
        try:
            if c["held"] is None:
                c["held"] = self._g56_cfg_held()
            n = [0, 0, 0, {}]
            for batch in (batch_list if isinstance(batch_list, list) else [batch_list]):
                n[0] += 1
                for it in batch.file_items:
                    n[1] += 1
                    n[2] += int(bool(getattr(it, "flip_x", False)))
                    rate = f"{float(getattr(getattr(it, 'dataset_config', None), 'caption_dropout_rate', -1.0)):g}"
                    n[3][rate] = n[3].get(rate, 0) + 1
            c["pending"] = n
        except Exception as e:  # noqa: BLE001
            c["pending"], c["error"] = None, c["error"] or f"{type(e).__name__}: {str(e)[:120]}"

    def _g56_cfg_commit(self, trained):
        """The end of a step: its batches and items join the counts when the step trained."""
        c = self._g56_cfg
        n, c["pending"] = c["pending"], None
        if n and trained:
            c["steps"], c["batches"], c["items"], c["mirrored"] = c["steps"] + 1, c["batches"] + n[0], c["items"] + n[1], c["mirrored"] + n[2]
            for k, v in n[3].items():
                c["by_dropout"][k] = c["by_dropout"].get(k, 0) + v

    def _g56_cfg_stats(self):
        c = self._g56_cfg
        return dict(steps=c["steps"], batches=c["batches"], items=c["items"], mirrored=c["mirrored"], by_dropout=c["by_dropout"],
                    **(c["held"] or {}), **({"error": c["error"]} if c["error"] else {}))

    def _g56_strata_setup(self, spec):
        """
        """
        k = int(spec.get("ts_strata") or 0)
        self._g56_strata = None if k < 2 else dict(k=k, rng=random.Random(f"g56-strata-{self._g56_seed}"), items={}, last=None,
                                                   steps=0, drawn=0, applied=0, counts=[0] * k, lo=None, hi=None, off=None)

    def _g56_tstrat_setup(self, spec):
        """
        """
        self._g56_tstrat = None
        if spec.get("tstrat"):
            self._g56_tstrat = dict(u0=0.0, n=0, last=None, steps=0, drawn=0, applied=0, own=0, lo=None, hi=None, counts=None, off=None)
            self._g56_tstrat_restart(self._g56_seed)

    def _g56_tstrat_restart(self, seed):
        """A member starts its own sequence: u0 from its seed through a private generator (never a global draw), n = 0."""
        ts = getattr(self, "_g56_tstrat", None)
        if ts is not None:
            ts["u0"], ts["n"], ts["last"] = random.Random(f"g56-tstrat-{seed}").random(), 0, None

    def _g56_tstrat_unsupported(self, batch):
        """
        """
        tc = self.train_config
        if str(getattr(tc, "timestep_type", "linear")) in TSTRAT_OWN_TYPES:
            return f"timestep type {tc.timestep_type}"
        reg = getattr(batch, "get_is_reg_list", None)
        if callable(reg) and any(reg()):
            return "a regularization batch"
        if str(getattr(tc, "content_or_style", "balanced")) != "balanced":
            return f"content_or_style {tc.content_or_style}"
        if getattr(getattr(self, "model_config", None), "refiner_name_or_path", None) is not None:
            return "a refiner"
        if getattr(getattr(self, "sd", None), "is_multistage", False):
            return "a multistage model"
        return None

    def _g56_tstrat_index(self, batch):
        """The timestep index this batch trains at under G56NF_TSTRAT -> (index, lo, hi) or None = AI Toolkit's own draw. AI Toolkit
        draws torch.randint(lo, hi) over the band - lo = max(min_denoising_steps, 0), hi = min(max_denoising_steps, T - 1), flowmatch;
        else lo + 1 .. hi - 1 (:1225-1226, :1283-1298) - so its marginal is uniform on [lo, hi); the member's n-th point u_n =
        frac(u0 + n x PHI_INV) goes through that CDF's inverse: lo + floor(u_n x (hi - lo)). A batch of several items, a band of one
        index, or a draw that is not the uniform one -> None (counted as own); a failure of this code -> off for the rest of the run,
        said once."""
        ts = self._g56_tstrat
        try:
            ts["steps"] += 1
            ts["last"] = None
            why = self._g56_tstrat_unsupported(batch)
            if why is not None or len(batch.file_items) != 1:
                ts["own"] += 1
                return None
            if ts["lo"] is None:   # the job's band and its draw's bounds, read once
                tc = self.train_config
                lo = max(int(tc.min_denoising_steps), 0)
                hi = min(int(tc.max_denoising_steps), int(tc.num_train_timesteps) - 1)
                if str(getattr(tc, "noise_scheduler", "flowmatch")) != "flowmatch":
                    lo, hi = lo + 1, hi - 1
                ts["lo"], ts["hi"], ts["counts"] = lo, hi, [0] * max(hi - lo, 0)
            lo, hi = ts["lo"], ts["hi"]
            if hi - lo < 2:
                ts["own"] += 1
                return None
            u = (ts["u0"] + ts["n"] * PHI_INV) % 1.0
            idx = lo + min(hi - lo - 1, int(u * (hi - lo)))
            ts["n"], ts["last"], ts["drawn"] = ts["n"] + 1, idx, ts["drawn"] + 1
            ts["counts"][idx - lo] += 1
            return idx, lo, hi
        except Exception as e:
            ts["off"] = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"g56 tstrat: OFF from step {ts['steps']} ({ts['off']}): AI Toolkit's own draw", flush=True)
            return None

    def _g56_tstrat_stats(self):
        """the status entry: the counts and the drawn indices' Kolmogorov distance to the uniform band (disc)"""
        ts = self._g56_tstrat
        disc = None
        c = ts["counts"] or []
        tot = sum(c)
        if tot:
            run, disc = 0, 0.0
            for j, x in enumerate(c):
                run += x
                disc = max(disc, abs(run / tot - (j + 1) / len(c)))
            disc = round(disc, 6)
        return dict(steps=ts["steps"], drawn=ts["drawn"], applied=ts["applied"], own=ts["own"], lo=ts["lo"], hi=ts["hi"], disc=disc,
                    off=ts["off"])

    def _g56_strata_restart(self, seed):
        """A member starts: its own cycles, from its own seed (member 1 = the run's seed, as _g56_strata_setup seeds it)."""
        if self._g56_strata is not None:
            self._g56_strata["items"].clear()
            self._g56_strata["rng"] = random.Random(f"g56-strata-{seed}")

    def _g56_strata_stats(self):
        st = self._g56_strata
        return dict(k=st["k"], lo=st["lo"], hi=st["hi"], steps=st["steps"], drawn=st["drawn"], applied=st["applied"],
                    items=len(st["items"]), counts=st["counts"], off=st["off"])

    def _g56_strata_index(self, batch):
        """The timestep index this batch trains at under G56NF_TS_STRATA, or None = AI Toolkit's own uniform draw (a batch of
        several items; a band of one index; this code failed once: off for the rest of the run, said once). The job's band - AI
        Toolkit draws the indices lo .. hi - 1 (BaseSDTrainProcess.py:1288-1298) - is cut into K strata of equal width; every
        training item - the image file, mirrored or not, of its dataset entry - takes the K strata in a random order, one per
        visit (a new order every K visits), and a uniform index inside the stratum. Over the draws the index is uniform on the
        band, as without the knob."""
        st = self._g56_strata
        try:
            st["steps"] += 1
            st["last"] = None
            items = batch.file_items
            if len(items) != 1:
                return None
            if st["lo"] is None:   # the job's band, read once
                st["lo"], st["hi"] = int(self.train_config.min_denoising_steps), int(self.train_config.max_denoising_steps)
            lo, width = st["lo"], st["hi"] - st["lo"]
            if width < 2:
                return None
            k, it = min(st["k"], width), items[0]
            key = (str(it.path), bool(getattr(it, "flip_x", False)), bool(getattr(it, "flip_y", False)),
                   float(getattr(getattr(it, "dataset_config", None), "caption_dropout_rate", -1.0)))
            cycle = st["items"].setdefault(key, [])
            if not cycle:
                cycle.extend(range(k))
                st["rng"].shuffle(cycle)
            s = cycle.pop()
            idx = lo + min(width - 1, int((s + st["rng"].random()) * width / k))
            st["last"], st["drawn"] = (key, s), st["drawn"] + 1
            st["counts"][s] += 1
            return idx
        except Exception as e:
            st["off"] = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"g56 ts strata: OFF from step {st['steps']} ({st['off']}): AI Toolkit's own draw", flush=True)
            return None

    def process_general_training_batch(self, batch):
        ts = getattr(self, "_g56_tstrat", None)
        if ts is not None:
            pick = self._g56_tstrat_index(batch) if ts["off"] is None else None
            if pick is None:
                return super().process_general_training_batch(batch)
            import torch
            # AI Toolkit's own draw, made with its arguments and discarded (:1293-1298): the torch RNG streams - this step's noise and
            # every later one - stay where they are without the knob (the randint is the function's first torch draw)
            torch.randint(pick[1], pick[2], (len(batch.file_items),), device=self.device_torch)
            idx, st = pick[0], ts
        else:
            st = getattr(self, "_g56_strata", None)
            idx = self._g56_strata_index(batch) if st is not None and st["off"] is None else None
        if idx is None:
            return super().process_general_training_batch(batch)
        tc = self.train_config
        lo, hi = tc.min_denoising_steps, tc.max_denoising_steps
        tc.min_denoising_steps = tc.max_denoising_steps = idx
        try:
            out = super().process_general_training_batch(batch)
        finally:
            tc.min_denoising_steps, tc.max_denoising_steps = lo, hi
        try:   # the effect, in production form: the timestep AI Toolkit returned is its table's entry at our index
            if abs(float(out[2].flatten()[0]) - float(self.sd.noise_scheduler.timesteps[idx])) < 1e-3:
                st["applied"] += 1
        except Exception:  # noqa: BLE001  (the count stays behind: the driver's line shows it)
            pass
        return out

    def _g56_save(self, name, tag, step=None, meta=None):
        """One atomic save of the LoRA as it is now (multiplier 1, float32, its step in the header) -> <save_root>/<name>."""
        import torch
        root = Path(self.save_root)
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f".tmp_{tag}.safetensors"
        prev = self.network.multiplier
        self.network.multiplier = 1.0
        t = time.time()
        try:
            self.network.save_weights(str(tmp), dtype=torch.float32,
                                      metadata={"g56_step": str(int(self.step_num if step is None else step)), **(meta or {})})
        finally:
            self.network.multiplier = prev
        os.replace(tmp, root / name)
        self._g56_save_t = time.time() - t
        self._g56_save_acc = getattr(self, "_g56_save_acc", 0.0) + self._g56_save_t   # (PLAN5: a save's seconds are overhead, not a step's time)

    def _g56_snapshot(self, tag):
        t = time.time()
        self._g56_save(f"g56_{tag}.safetensors", tag)
        self._g56_save_s = max(self._g56_save_s * 0.5, time.time() - t)
        self._g56_t_snap = time.time()
        self._g56_snap_s = getattr(self, "_g56_snap_s", 0.0) + (self._g56_t_snap - t)   # (PLAN5: member 1's snapshots are not a member's overhead)

    def _g56_keep_step(self):
        """."""
        try:
            self._g56_save(f"keep_{int(self.step_num):07d}.safetensors", "keep")
        except Exception as ke:
            print(f"g56 keep: step {int(self.step_num)} not saved ({type(ke).__name__})", flush=True)

    def _g56_tail_step(self):
        """
        
        """
        k, L = self._g56_member, int(self._g56_len or 0)
        local = int(self.step_num) - self._g56_member_step0
        if not L or local not in tail_window(self._g56_tail, L):
            return False
        t = time.time()
        try:
            self._g56_save(f"tail_{k:02d}_{local:07d}.safetensors", "tail", step=local, meta={"g56_member": str(k), "g56_len": str(L)})
            self._g56_save_s = max(self._g56_save_s * 0.5, time.time() - t)
        except Exception as te:
            print(f"g56 tail: member {k} step {local} not saved ({type(te).__name__})", flush=True)
        return True

    def _g56_reset_member(self, k, t0):
        """
        
        
        
        
        
        """
        import math
        import random
        import numpy as np
        import torch
        loras = list(self.network.get_all_modules())
        weights = []
        for m in loras:
            down, up = getattr(m, "lora_down", None), getattr(m, "lora_up", None)
            if not (hasattr(down, "weight") and hasattr(up, "weight")) or getattr(up, "bias", None) is not None:
                raise RuntimeError(f"a LoRA module that is not lora_down / lora_up weights only ({type(m).__name__})")
            weights += [down.weight, up.weight]
        opt = getattr(self.optimizer, "optimizer", self.optimizer)   # accelerate wraps the optimizer
        trained = {id(p) for g in opt.param_groups for p in g["params"]}
        if not weights or trained != {id(w) for w in weights}:
            raise RuntimeError(f"the optimizer trains {len(trained)} tensors, the LoRA modules hold {len(weights)} factors")
        kind = (type(opt).__module__.split(".")[0], type(opt).__name__)
        if kind not in RESET_OPTIMIZERS:
            raise RuntimeError(f"the optimizer is {kind[0]} {kind[1]}: not one whose cleared state is known to start again")
        sched = getattr(self.train_config, "lr_scheduler", "constant")
        if sched != "constant":
            raise RuntimeError(f"the LR schedule is {sched!r}: its state would have to start again too")
        seed = self._g56_seed + (k - 1)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            for m in loras:
                torch.nn.init.kaiming_uniform_(m.lora_down.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(m.lora_up.weight)
        opt.state.clear()
        opt.zero_grad(set_to_none=True)
        self._g56_member, self._g56_member_step0, self._g56_member_t0 = k, int(self.step_num), t0   # its clock includes this reset
        self._g56_keep = frozenset()   # G56NF_KEEP_STEPS are steps of member 1
        self._g56_strata_restart(seed)   # G56NF_TS_STRATA: member k draws its own cycles
        self._g56_tstrat_restart(seed)   # G56NF_TSTRAT: member k runs its own sequence

    def _g56_member_done(self, k, local, step_s):
        """Member k reached its step cap: saved, timed, and the plan for the next one (member_plan); stops the job when none starts."""
        t = time.time()
        self._g56_save(f"member_{k:02d}.safetensors", "member", step=local,
                       meta={"g56_member": str(k), **({"g56_len": str(int(self._g56_len))} if self._g56_p5 is not None else {})})
        self._g56_save_s = max(self._g56_save_s * 0.5, time.time() - t)
        now = time.time()
        self._g56_member_s.append(now - self._g56_member_t0)
        window = self._g56_deadline - self._g56_t_train0
        length = self._g56_max_steps
        if self._g56_p5 is not None:
            go, length, k_fit, why = self._g56_p5.plan(k, now, self._g56_save_s, member_wall=self._g56_member_s[-1] - getattr(self, "_g56_snap_s", 0.0))
            self._g56_snap_s = 0.0
            self._g56_deadline = self._g56_p5.deadline()
            window = self._g56_deadline - self._g56_t_train0
        elif self._g56_fill is not None:
            go, length, k_fit, why = self._g56_fill_decide(k, now)
        else:
            go, k_fit, why = member_plan(k, self._g56_members, self._g56_member_s, window, self._g56_deadline - now)
        if go:
            try:
                self._g56_reset_member(k + 1, now)
                self._g56_len = self._g56_len0 = length
            except Exception as re_:  # noqa: BLE001  (no reset = no further member; the finished ones are shipped)
                go, why = False, f"reset refused: {type(re_).__name__}: {str(re_)[:120]}"
        line = (f"member {k}/{self._g56_members} done: steps {local}, t {self._g56_member_s[-1]:.1f} s, planner: window {window:.0f} s, "
                f"longest member {max(self._g56_member_s):.1f} s, K_fit {k_fit}, next: {'start' if go else 'stop'} ({why})")
        print("g56 " + line, flush=True)
        self._g56_member_log.append(line)
        _status(self._g56_status, phase="train" if go else "done", step=int(self.step_num), step_s=step_s, save_s=self._g56_save_s,
                **self._g56_extra())
        if not go:
            raise _G56Stop()

    def _g56_member_rules(self, now, step_s, oom):
        """K members: the end-of-step rules (in place of the single run's stop / snapshot rules)."""
        k = self._g56_member
        local = int(self.step_num) - self._g56_member_step0
        p5 = self._g56_p5
        if p5 is not None and not p5.probed and k == 1 and not oom and p5.probe <= local < self._g56_len:
            self._g56_p5_probe(now, local, step_s)   # PLAN5: the first plan, while member 1 trains
        if p5 is not None and p5.probed and not oom:   # PLAN5: while member k's length is not final, the plan is checked against the clock
            first = (tail_window(self._g56_tail, self._g56_len) or [self._g56_len])[0] if self._g56_tail is not None else self._g56_len
            last_call, at_end, beat = local == first - 1, local >= self._g56_len, local < first - 1 and self._g56_n % 25 == 0
            if p5.k_plan > k and (last_call or at_end or beat):   # do the members after it still fit? (also at its end: never a short last member)
                new = p5.rest(k, now, self._g56_save_s, local, self._g56_len, now - self._g56_member_t0 - getattr(self, "_g56_snap_s", 0.0))
                self._g56_deadline = p5.deadline()
                if new is not None:
                    line = (f"plan5: member {k} at step {local}: no member fits after it any more - it is the plan's last, its length "
                            f"{self._g56_len} -> {new} steps")
                    self._g56_len = self._g56_len0 = new
                    self._g56_p5_line(line, "train", step_s)
            elif p5.k_plan <= k and not at_end and (last_call or beat):   # the plan's last member: its length follows what fits now
                new = p5.refit(k, now, self._g56_save_s, local, self._g56_len, p5.keep * self._g56_len0)
                if new is not None:
                    if abs(new - self._g56_p5_said) >= max(10, self._g56_max_steps // 20):   # (one line per real change, not per beat)
                        self._g56_p5_said = new
                        self._g56_p5_line(f"plan5: member {k} at step {local}: the plan's last member, its length {self._g56_len} -> {new} "
                                          f"steps (what fits before the deadline now)", "train", step_s)
                    self._g56_len = new
        if self._g56_max_steps and local >= self._g56_len:
            self._g56_member_done(k, local, step_s)
            return
        if now + 1.5 * step_s + 2.0 * self._g56_save_s > self._g56_deadline:
            if p5 is not None and self._g56_p5_cut(k, local, step_s):   # PLAN5 'cut': far enough, the member is kept
                raise _G56Stop()
            if k == 1:
                self._g56_snapshot("final")   # no member finished: the unfinished member 1 is the file, as in the single run
            else:
                line = f"member {k}/{self._g56_members} cut by the deadline at step {local} of {self._g56_len}: no file"
                print("g56 " + line, flush=True)
                self._g56_member_log.append(line)
            _status(self._g56_status, phase="done", step=int(self.step_num), step_s=step_s, save_s=self._g56_save_s,
                    **self._g56_extra())
            raise _G56Stop()
        if self._g56_crash_at and self._g56_n >= self._g56_crash_at:
            print(f"g56 test crash after {self._g56_n} trained steps", flush=True)
            sys.stdout.flush()
            os._exit(3)
        if k == 1 and not oom and snapshot_due(self._g56_n, now, self._g56_t_snap, self._g56_every, self._g56_first):
            self._g56_snapshot("snap")

    def _g56_guard(self, e):
        try:
            if not is_cuda_oom(e):
                return
            self._g56_oom += 1
            self._g56_oom_step = True
            st = getattr(self, "_g56_strata", None)
            if st is not None and st["last"] is not None:   # G56NF_TS_STRATA: the OOM step trained nothing - its stratum goes back
                (key, s), st["last"] = st["last"], None
                st["items"][key].append(s)
                st["drawn"], st["counts"][s] = st["drawn"] - 1, st["counts"][s] - 1
            ts = getattr(self, "_g56_tstrat", None)
            if ts is not None and ts["last"] is not None:   # G56NF_TSTRAT: the OOM step trained nothing - its point goes back
                ts["n"], ts["drawn"], ts["counts"][ts["last"] - ts["lo"]] = ts["n"] - 1, ts["drawn"] - 1, ts["counts"][ts["last"] - ts["lo"]] - 1
                ts["last"] = None
            step = int(self.step_num) + 1
            what = "already on"
            if not getattr(self.train_config, "gradient_checkpointing", True):
                m = getattr(self.sd, "unet_unwrapped", None)
                what = "ON" if gc_switch_on(self.sd.unet if m is None else m) else "no switch on this model"
                self.train_config.gradient_checkpointing = True
                self._g56_gc_at = step
            print(f"g56 OOM guard: step {step} OOM ({self._g56_oom} so far); gradient checkpointing {what}", flush=True)
            _status(self._g56_status, phase="train", step=int(self.step_num), **self._g56_extra())
        except Exception as ge:  # noqa: BLE001  (the guard never stops the training)
            print(f"g56 OOM guard: inactive ({type(ge).__name__})", flush=True)

    def hook_train_loop(self, batch_list):
        if self._g56_ballast_gb and self._g56_ballast is None:
            try:
                import torch
                gb, leave = self._g56_ballast_gb, ""
                if gb < 0:
                    torch.cuda.empty_cache()
                    gb, leave = torch.cuda.mem_get_info()[0] / 2 ** 30 + gb, f" (leaving {-gb:g} GiB)"
                self._g56_ballast = torch.empty(int(gb * 2 ** 30), dtype=torch.uint8, device="cuda")
                print(f"g56 test ballast: {gb:g} GiB held{leave}", flush=True)
            except Exception as be:  # noqa: BLE001
                self._g56_ballast = False
                print(f"g56 test ballast: not held ({type(be).__name__})", flush=True)
        if self._g56_t_train0 is None:
            self._g56_t_train0 = self._g56_member_t0 = time.time()   # the first training step starts: the members' time window
        if getattr(self, "_g56_mask_dir", None) is not None:
            self._g56_attach_masks(batch_list)
        if getattr(self, "_g56_cfg", None) is not None:        # a 'configuration only' knob is on: this step's batches and items
            self._g56_cfg_count(batch_list)
        if getattr(self, "_g56_strata", None) is not None:
            self._g56_strata["last"] = None
        if getattr(self, "_g56_tstrat", None) is not None:
            self._g56_tstrat["last"] = None
        pf = getattr(self, "_g56_prof", None)
        if pf is not None and pf["on"] is None and pf["done"] is None and int(self.step_num) + 1 == pf["a"]:   # G56NF_PROFILE: step a starts
            self._g56_profile_start()
        if getattr(self, "_g56_sdpa_want", None) and self._g56_sdpa is None:
            self._g56_sdpa = sdpa_install(self._g56_sdpa_want)
            print(f"g56 sdpa backend {self._g56_sdpa_want}: {'applied' if self._g56_sdpa['applied'] else 'not applied'} "
                  f"({self._g56_sdpa['why']}); order {self._g56_sdpa['order']}", flush=True)
        ns = getattr(self, "_g56_nosync", None)
        if ns is not None and ns["applied"] is None:
            self._g56_nosync = bnb_nosync_install(getattr(self, "optimizer", None))
            print(f"g56 bnb nosync: {'applied' if self._g56_nosync['applied'] else 'not applied'} ({self._g56_nosync['why']})", flush=True)
        if getattr(self, "_g56_freeze_k", 0):
            self._g56_freeze_step()
        if getattr(self, "_g56_gcs_fence", 0) and self._g56_gcs is None:
            m = getattr(self.sd, "unet_unwrapped", None)
            self._g56_gcs = gc_step_install(self.sd.unet if m is None else m, self._g56_gcs_fence, self)
            print(f"g56 gc step fence {self._g56_gcs_fence}: {'applied' if self._g56_gcs['applied'] else 'not applied'} "
                  f"({self._g56_gcs['why']})", flush=True)
        try:
            return super().hook_train_loop(batch_list)
        except BaseException as e:
            self._g56_guard(e)
            raise

    def end_step_hook(self):
        super().end_step_hook()
        pf = getattr(self, "_g56_prof", None)
        if pf is not None and pf["on"] is not None and int(self.step_num) >= pf["b"]:   # G56NF_PROFILE: step b ended
            self._g56_profile_stop()
            _status(self._g56_status, phase="train", step=int(self.step_num), **self._g56_extra())
        now = time.time()
        oom, self._g56_oom_step = self._g56_oom_step, False
        if getattr(self, "_g56_cfg", None) is not None:   # a 'configuration only' knob is on: an OOM step trained nothing
            self._g56_cfg_commit(not oom)
        if self._g56_t_last is not None and not oom:
            self._g56_dts.append(now - self._g56_t_last)
            self._g56_dts = self._g56_dts[-20:]
            if self._g56_fill is not None:
                self._g56_fill_step(now - self._g56_t_last)
            if self._g56_p5 is not None:     # PLAN5: the step times of the plan, without the saves written since the last step ended
                self._g56_p5.step(max(0.0, now - self._g56_t_last - getattr(self, "_g56_save_acc", 0.0)))
        self._g56_save_acc = 0.0
        if self._g56_t_last is None and self._g56_p5 is not None and self._g56_t_train0 is not None:
            self._g56_p5.first = now - self._g56_t_train0   # PLAN5: the first training step's seconds (not a step time: it holds the warm-up)
        self._g56_t_last = now
        if not oom:
            self._g56_n += 1
        dts = sorted(self._g56_dts[2:] or self._g56_dts or [0.0])
        step_s = dts[len(dts) // 2]
        if self._g56_n % 25 == 0 and not oom:
            if self._g56_p5 is not None:
                self._g56_deadline = min(self._g56_deadline, self._g56_p5.tick())
            _status(self._g56_status, phase="train", step=int(self.step_num), step_s=step_s, save_s=self._g56_save_s,
                    **self._g56_extra())
        if self._g56_keep and not oom and int(self.step_num) in self._g56_keep:
            self._g56_keep_step()
            now = time.time()   # the stop rule below sees the time the keep save took
        if self._g56_tail is not None and not oom and self._g56_tail_step():
            now = time.time()   # the stop rule below sees the time the save took
        if self._g56_members > 1 or self._g56_p5 is not None:
            self._g56_member_rules(now, step_s, oom)
            return
        if (now + 1.5 * step_s + 2.0 * self._g56_save_s > self._g56_deadline
                or (self._g56_max_steps and int(self.step_num) >= self._g56_max_steps)):
            self._g56_snapshot("final")
            _status(self._g56_status, phase="done", step=int(self.step_num), step_s=step_s, save_s=self._g56_save_s,
                    **self._g56_extra())
            raise _G56Stop()
        if self._g56_crash_at and self._g56_n >= self._g56_crash_at:
            print(f"g56 test crash after {self._g56_n} trained steps", flush=True)
            sys.stdout.flush()
            os._exit(3)
        if not oom and snapshot_due(self._g56_n, now, self._g56_t_snap, self._g56_every, self._g56_first):
            self._g56_snapshot("snap")


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    root = spec["aitk_root"]
    # run.py's environment (AI Toolkit's own entry point), offline
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    os.environ["DISABLE_TELEMETRY"] = "YES"
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"
    os.environ["HF_HOME"] = spec["hf_home"]
    if spec.get("hf_hub_cache"):   # the validator-staged hub cache (read-only /cache): the evaluator's own runtime files
        os.environ["HF_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"] = spec["hf_hub_cache"]
    os.environ["MODELS_PATH"] = spec["models_path"]   # read by toolkit/paths.py at import: set before any AI Toolkit import
    for p in (Path(spec["hf_home"]), Path(spec["models_path"])):
        if not p.is_dir():   # the staged MODELS_PATH exists on a read-only mount: never mkdir there
            p.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    sys.path.insert(0, root)
    status = spec["status"]
    _status(status, phase="import")

    import random
    import numpy as np
    import torch
    seed = int(spec.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if spec.get("tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"g56 tf32: {json.dumps(tf32_state(), sort_keys=True)}", flush=True)
    from toolkit.util.unified_memory import apply_unified_memory_patches
    apply_unified_memory_patches()

    config = spec["config"]
    proc_cfg = config["config"]["process"][0]

    # ---- offline patches
    if spec.get("klein_te_dir"):
        from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein4BModel
        Flux2Klein4BModel.flux2_klein_te_path = spec["klein_te_dir"]
    if spec.get("klein_vae"):
        src, dst = spec["klein_vae"]
        if not Path(dst).is_file() and not spec.get("construct_only"):
            from safetensors.torch import load_file, save_file
            from toolkit.models.v2.vae.flux2_kl import convert_diffusers_state_dict
            sd = load_file(src)
            if any(k.startswith("encoder.down_blocks.") for k in sd):
                sd = convert_diffusers_state_dict(sd)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            save_file({k: v.contiguous() for k, v in sd.items()}, dst)
        proc_cfg["model"]["vae_path"] = dst
    if spec.get("moss_processor_dir"):
        import extensions_built_in.diffusion_models.moss_image.moss_image as moss
        _orig = moss._hub_file_folder
        pdir = spec["moss_processor_dir"]

        def _hub_file_folder(repo_id, filename):
            if filename == "preprocessor_config.json":
                return pdir
            return _orig(repo_id, filename)
        moss._hub_file_folder = _hub_file_folder
    if spec.get("lora_noop"):
        import toolkit.network_mixins as NM
        lora_noop_install(NM)

    # ---- the timed trainer
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    class G56TimedTrainer(G56StepRules, SDTrainer):
        def __init__(self, process_id, job, cfg, **kwargs):
            super().__init__(process_id, job, cfg, **kwargs)
            self._g56_setup(spec, status)

    #
    #
    EJ = importlib.import_module("jobs.ExtensionJob")
    _orig_dict = EJ.get_all_extensions_process_dict

    def _process_dict():
        d = dict(_orig_dict())
        d["g56_timed_trainer"] = G56TimedTrainer
        return d
    EJ.get_all_extensions_process_dict = _process_dict

    from toolkit.job import get_job
    _status(status, phase="load")
    job = get_job(config)
    if spec.get("construct_only"):   # tools/newfam_check.py case F (CPU, real AI Toolkit): config classes + __init__ ran, no model
        print("g56 aitk construct ok:", spec["family"], spec["mode"], json.dumps([type(p).__name__ for p in job.process]),
              flush=True)
        try:
            p0 = job.process[0]
            tc, nc = p0.train_config, p0.network_config
            print("g56 aitk construct config:", json.dumps(dict(
                batch_size=tc.batch_size, gradient_accumulation=tc.gradient_accumulation, content_or_style=tc.content_or_style,
                optimizer=tc.optimizer, optimizer_params=tc.optimizer_params, max_grad_norm=tc.max_grad_norm, linear=nc.linear,
                linear_alpha=nc.linear_alpha,
                datasets=[[bool(d.flip_x), float(d.caption_dropout_rate)] for d in p0.dataset_configs]), sort_keys=True), flush=True)
        except Exception as ce:  # noqa: BLE001
            print(f"g56 aitk construct config: not read ({type(ce).__name__}: {str(ce)[:120]})", flush=True)
        _status(status, phase="constructed")
        os._exit(0)
    try:
        job.run()
        _status(status, phase="finished")
    except _G56Stop:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)   # no teardown of CUDA / dataloader workers (the snapshot is on disk)


def selftest_import(root):
    """
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DISABLE_TELEMETRY"] = "YES"
    os.chdir(root)
    sys.path.insert(0, root)
    EJ = importlib.import_module("jobs.ExtensionJob")
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    class G56Probe(SDTrainer):
        pass
    _orig = EJ.get_all_extensions_process_dict

    def _d():
        d = dict(_orig())
        d["g56_timed_trainer"] = G56Probe
        return d
    EJ.get_all_extensions_process_dict = _d
    d = EJ.get_all_extensions_process_dict()
    assert d.get("g56_timed_trainer") is G56Probe, "process type not registered"
    assert "diffusion_trainer" in d, "AI Toolkit's own diffusion_trainer missing"
    from toolkit.config_modules import ModelConfig
    from toolkit.util.get_model import get_model_class
    got = {a: get_model_class(ModelConfig(name_or_path="x", arch=a)).__name__
           for a in ("flux2_klein_4b", "qwen_image_edit_plus", "hidream_o1", "moss_image:base", "moss_image:base_edit")}
    from toolkit.models.v2.vae.flux2_kl import convert_diffusers_state_dict  # noqa: F401  (the Klein VAE patch)
    import extensions_built_in.diffusion_models.moss_image.moss_image as moss
    assert callable(moss._hub_file_folder)
    print("g56 aitk import smoke ok:", json.dumps(got), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--selftest-import":
        selftest_import(sys.argv[2])
        sys.stdout.flush()
        os._exit(0)
    main(sys.argv[1])
