"""Arm sets of the comparison.

An arm is one sampler configuration run on a cell: a schedule (the string "fixed" or a policy object), a cap T on
the number of steps, a number of runs, and overrides of the sampler's resampling settings. An arm set is a
function of the cell that returns the arms of one study; `ARM_SETS` maps the `--arms` name to it. To add a study,
write one function below and decorate it.

Arm names are part of the results: each arm's random stream is derived from its name (cells.stream_seed), and
analysis scripts select arms by name. Renaming an arm changes its numbers.
"""
from dataclasses import dataclass, field
import torch

from .cells import stream_seed
from .gate import resampling_gate

THETAS = (0.0, 0.003, 0.01, 0.03, 0.1)
BUDGETS = (0.01, 0.03, 0.1, 0.3, 1.0)      # whole-sequence error budgets in nats; the policy gets rate = B / N
THICK_SMALL_BUDGETS = (0.001, 0.0001)    # appended after BUDGETS so earlier partial results stay valid
THICK_WIDTHS = (1, 2, 3, 4, 6)
THICK_SPACINGS = (8, 16, 24, 32, 48, 64)
ORDER_PRIORITIES = ("bound", "random", "confident", "separator", "sum", "weighted", "lex", "least", "least_random", "least_tie")
SEQ_ORDERS = ("sum", "lex", "separator", "random", "bound")
RESAMPLE_KS = (4, 16, 64, 256)
RESAMPLE_MODES = (("off", dict(ess_min=0.0)), ("defensive", dict()), ("plain", dict(defensive=0.0)),
                  ("gated", dict(defensive=0.0, resample_power="gate")),
                  ("systematic", dict(defensive=0.0, resampler="systematic")),
                  ("power 0.5", dict(defensive=0.0, resample_power=0.5)), ("power 0.25", dict(defensive=0.0, resample_power=0.25)),
                  ("power 0.5 systematic", dict(defensive=0.0, resample_power=0.5, resampler="systematic")))
INTERACT_BUDGETS = (0.03, 0.3)
INTERACT_ORDERS = ("separator", "sum", "weighted", "lex", "bound", "random", "least_tie")
FIX_VARIANTS = (("", 0, False), (" +pull", 8, False), (" +pull+guard", 8, True))      # name tag, pull_forced, guard
OFF = dict(ess_min=0.0)                   # no resampling

ARM_SETS = {}


@dataclass
class Arm:
    name: str
    schedule: object                      # "fixed" or an object with .reveal(state, t, T, gen)
    T: int                                # number of steps ("fixed") or the cap on them (policies)
    nrun: int
    smc: dict = field(default_factory=dict)       # overrides of twisted_smc's resampling settings (default: defensive 0.25)
    K: int = None                         # particles per run; None = the cell's --K


class Cell:
    """One (model, constraint, N) with its pilot kernels, fitted on first use and kept."""

    def __init__(self, mname, cname, N, a, den, cl, route, truth):
        self.mname, self.cname, self.N, self.a = mname, cname, N, a
        self.den, self.cl, self.route, self.truth = den, cl, route, truth
        self.nrun = a["nrun"]
        self.gate_info = None
        self._kernels = {}

    def seed(self, *parts):
        return stream_seed(self.a["seed"], self.mname, self.cname, self.N, *parts)

    def kernel(self, kind="two-bin"):
        """"two-bin": screened / unscreened by distance. "graded": binned by the number of revealed positions in
        between, from the same pilot traces. "small": two-bin from --pilot-small traces."""
        if kind not in self._kernels:
            from twistroute.decode import fit_distance_kernel, COUNT_EDGES
            n, tag = (self.a["pilot_small"], "pilot_small") if kind == "small" else (self.a["pilot"], "pilot")
            kw = dict(count_edges=COUNT_EDGES) if kind == "graded" else {}
            self._kernels[kind] = fit_distance_kernel(self.den, self.N, n, torch.Generator().manual_seed(self.seed(tag)), **kw)[0]
        return self._kernels[kind]

    def policy(self, kernel, theta=0.0, priority="bound", B=None, **kw):
        """CouplingPolicy without the mixed term; B is a whole-sequence budget in nats (rate B / N)."""
        from twistroute.decode import CouplingPolicy
        kw.setdefault("mixed", False)
        return CouplingPolicy(kernel, theta, priority=priority, rate=None if B is None else B / self.N, **kw)

    def chain_twist(self, eps=0.05, temp=1.0):
        """The first-order chain twist for this cell's automaton (decode.chain_twist), with a pair kernel fitted on
        --pilot model samples. Automaton constraints only. `temp` < 1 is the calibrated version (same pair kernel)."""
        from twistroute.decode import ChainTwist, fit_pair_kernel
        if "pair_kernel" not in self._kernels:
            if not hasattr(self.truth, "wdfa"):
                raise ValueError(f"no chain twist for the constraint {self.cname!r}: it is not an automaton")
            self._kernels["pair_kernel"] = fit_pair_kernel(self.den, self.cl, self.a["pilot"], torch.Generator().manual_seed(self.seed("pair_kernel")))
        key = ("twist", eps, temp)
        if key not in self._kernels:
            self._kernels[key] = ChainTwist(self.truth.wdfa, self._kernels["pair_kernel"], eps=eps, temp=temp,
                                            active=getattr(self.truth, "active", None))
        return self._kernels[key]

    def control(self):
        """One reveal per step in random order without resampling: samples the target itself and cannot hide mass."""
        return Arm("fixed 1/step, no resampling", "fixed", self.N, self.nrun, smc=dict(ess_min=0.0))


def arm_set(name):
    def register(f):
        ARM_SETS[name] = f
        return f
    return register


def budget_tag(B):
    return "theta=0.0" if B is None else f"B={B}"


def _standard(c, core):
    N, n, kern = c.N, c.nrun, c.kernel()
    arms = [Arm(f"fixed {N // T}/step", "fixed", T, n) for T in ((N, N // 8, N // 16) if core else (N, N // 2, N // 4, N // 8))]
    arms += [c.control()]
    arms += [Arm(f"policy theta={th}", c.policy(kern, th), N, n) for th in THETAS]
    if not core:
        arms += [Arm(f"small-pilot theta={th}", c.policy(c.kernel("small"), th), N, n) for th in (0.01, 0.03)]
        arms += [Arm(f"{pr} theta={th}", c.policy(kern, th, pr), N, n) for pr in ("confident", "random") for th in (0.01, 0.03)]
    arms += [Arm("random theta=0.0", c.policy(kern, 0.0, "random"), N, n)]            # same sets rule, order not chosen
    if c.a["nrun_mixed"] > 0 and not core:   # the mixed-path term costs N * A extra route passes per particle per step
        arms += [Arm(f"mixed theta={th}", c.policy(kern, th, mixed=True), N, c.a["nrun_mixed"]) for th in (0.0, 0.01, 0.03)]
        arms += [Arm(f"policy-ref theta={th}", c.policy(kern, th), N, c.a["nrun_mixed"]) for th in (0.0, 0.01, 0.03)]
    return arms


@arm_set("full")
def full(c):
    """Fixed schedules at 1, 2, 4, 8 per step, the threshold sweep, a small pilot, two other priorities, the mixed term."""
    return _standard(c, core=False)


@arm_set("core")
def core(c):
    """For long sequences: the controls, two naive parallel schedules, the threshold sweep."""
    return _standard(c, core=True)


@arm_set("order")
def order(c):
    """Which position next: every priority score at one threshold (--order-theta)."""
    N, n, th = c.N, c.nrun, c.a["order_theta"]
    arms = [Arm("fixed 1/step", "fixed", N, n), c.control()]
    return arms + [Arm(f"{pr} theta={th}", c.policy(c.kernel(), th, pr), N, n) for pr in ORDER_PRIORITIES]


@arm_set("budget")
def budget(c):
    """Whole-sequence budgets, graded against two-bin kernel, with fixed and dilated baselines."""
    from twistroute.decode import DilatedSchedule
    N, n, kg = c.N, c.nrun, c.kernel("graded")
    arms = [Arm("fixed 8/step", "fixed", N // 8, n)]
    arms += [Arm(f"dilated base={b}", DilatedSchedule(b), N, n) for b in (2, 4)]
    arms += [Arm(f"{pr} theta=0.0 graded", c.policy(kg, 0.0, pr), N, n) for pr in ("separator", "bound")]
    arms += [Arm(f"{pr} B={B} graded", c.policy(kg, 0.0, pr, B=B), N, n) for pr in ("separator", "bound", "random", "least_random") for B in BUDGETS]
    arms += [Arm(f"separator B={B} two-bin", c.policy(c.kernel(), 0.0, "separator", B=B), N, n) for B in BUDGETS]
    return arms


@arm_set("interact")
def interact(c):
    """How the order interacts with the constraint: every score at the exact threshold and two budgets, no resampling."""
    N, n, kg = c.N, c.nrun, c.kernel("graded")
    arms = [Arm(f"{pr} {budget_tag(B)}", c.policy(kg, 0.0, pr, B=B), N, n, smc=OFF) for B in (None,) + INTERACT_BUDGETS for pr in INTERACT_ORDERS]
    return arms + [c.control()]


@arm_set("thick")
def thick(c):
    """Do thick separators first beat the greedy rule's single-token bisection? No resampling: r is the measure."""
    from twistroute.decode import DilatedSchedule, ThickSeparatorSchedule
    N, n, kg = c.N, c.nrun, c.kernel("graded")
    arms = [Arm(f"separator B={B} graded", c.policy(kg, 0.0, "separator", B=B), N, n, smc=OFF) for B in BUDGETS + THICK_SMALL_BUDGETS]
    arms += [Arm("separator theta=0.0 graded", c.policy(kg, 0.0, "separator"), N, n, smc=OFF)]
    arms += [Arm("dilated base=2", DilatedSchedule(2), N, n, smc=OFF)]
    arms += [Arm(f"thick w={w} spacing={sp}", ThickSeparatorSchedule(w, sp), N, n, smc=OFF) for w in THICK_WIDTHS for sp in THICK_SPACINGS if sp >= 4 * w]
    return arms


@arm_set("seqorder")
def seqorder(c):
    """Order or parallelism? Every score one token per step, against its parallel versions. No resampling."""
    N, n, kg = c.N, c.nrun, c.kernel("graded")
    arms = [c.control()]
    for pr in SEQ_ORDERS:
        arms += [Arm(f"{pr} sequential", c.policy(kg, 0.0, pr, max_set=1), N, n, smc=OFF),
                 Arm(f"{pr} theta=0.0", c.policy(kg, 0.0, pr), N, n, smc=OFF),
                 Arm(f"{pr} B=0.03", c.policy(kg, 0.0, pr, B=0.03), N, n, smc=OFF)]
    return arms


@arm_set("fix")
def fix(c):
    """Selection fixes: forced-partner pull-in and the guard, against the plain rule. No resampling."""
    N, n, kg = c.N, c.nrun, c.kernel("graded")
    return [Arm(f"{pr} {budget_tag(B)}{tag}", c.policy(kg, 0.0, pr, B=B, pull_forced=pull, guard=gd), N, n, smc=OFF)
            for pr in ("lex", "sum") for B in (None, 0.03) for tag, pull, gd in FIX_VARIANTS]


@arm_set("resample")
def resample(c):
    """Does resampling hurt only when K is small? Total particles nrun * K held fixed across K; exact schedules
    only, so r = 0 throughout. The gate is fitted per schedule on a pilot without resampling."""
    N, a, kg = c.N, c.a, c.kernel("graded")
    names = ("fixed 1/step", "separator theta=0.0")
    sched = lambda sn: "fixed" if sn.startswith("fixed") else c.policy(kg, 0.0, "separator")
    gates, c.gate_info = {}, {}
    for sn in names:
        gates[sn], c.gate_info[sn] = resampling_gate(c.den, c.cl, c.route, N, sched(sn), a.get("gate_pilot", 4000),
                                                     torch.Generator().manual_seed(c.seed("gate", sn)))
    return [Arm(f"{sn} K={K} resampling {rn}", sched(sn), N, max(a["nrun"] * a["K"] // K, 2),
                smc={k: (gates[sn] if v == "gate" else v) for k, v in smc.items()}, K=K)
            for K in (a.get("resample_ks") or RESAMPLE_KS) for sn in names for rn, smc in RESAMPLE_MODES
            if not a.get("resample_modes") or rn in a["resample_modes"]]


TWIST_RESAMPLING = (("off", dict(ess_min=0.0)), ("defensive", dict()), ("plain", dict(defensive=0.0)))


@arm_set("twist")
def twist(c):
    """Product twist against the first-order chain twist, with and without resampling, on three schedules. Without
    resampling the two twists give the same weights; the question is what resampling does under each."""
    N, n, kg, tw = c.N, c.nrun, c.kernel("graded"), c.chain_twist()
    scheds = (("fixed 1/step", lambda: "fixed"), ("lex sequential", lambda: c.policy(kg, 0.0, "lex", max_set=1)),
              ("lex B=0.03 +pull", lambda: c.policy(kg, 0.0, "lex", B=0.03, pull_forced=8)))
    return [Arm(f"{sn} | {tn} twist | resampling {rn}", mk(), N, n, smc=dict(smc, **({"twist": tw} if tn == "chain" else {})))
            for sn, mk in scheds for tn in ("product", "chain") for rn, smc in TWIST_RESAMPLING]


CALIBRATE_TEMPS = (0.7, 0.5)


@arm_set("calibrate")
def calibrate(c):
    """Does calibration help resampling, under either twist? The calibrated twist flattens the model's marginals at
    masked positions (power tau, renormalised) before they enter the twist: `twist_temp` for the product twist
    (which also changes the proposal), `temp` for the chain twist (potential only). Controls: the product twist
    without resampling, and each twist uncalibrated."""
    N, n, kg = c.N, c.nrun, c.kernel("graded")
    scheds = (("fixed 1/step", lambda: "fixed"), ("lex B=0.03 +pull", lambda: c.policy(kg, 0.0, "lex", B=0.03, pull_forced=8)))
    res = (("defensive", dict()), ("plain", dict(defensive=0.0)))
    arms = []
    for sn, mk in scheds:
        arms.append(Arm(f"{sn} | product twist | resampling off", mk(), N, n, smc=dict(OFF)))
        for tau in (1.0,) + CALIBRATE_TEMPS:
            tag = "" if tau == 1.0 else f" tau={tau}"
            for rn, smc in res:
                arms.append(Arm(f"{sn} | product twist{tag} | resampling {rn}", mk(), N, n,
                                smc=dict(smc, **({} if tau == 1.0 else {"twist_temp": tau}))))
                arms.append(Arm(f"{sn} | chain twist{tag} | resampling {rn}", mk(), N, n,
                                smc=dict(smc, twist=c.chain_twist(temp=tau))))
    return arms


@arm_set("chainprop")
def chainprop(c):
    """The chain twist as the proposal. Three schedules (uniformly random order one token per step, `lex` one token
    per step, `lex` budgeted) by three samplers: product proposal with the product twist; product proposal with the
    chain twist as the intermediate twist (the earlier use); chain proposal with the chain twist as the
    intermediate twist. Each without resampling and with defensive resampling. Automaton constraints only."""
    N, n, kg, tw = c.N, c.nrun, c.kernel("graded"), c.chain_twist()
    scheds = (("fixed 1/step", lambda: "fixed"), ("lex sequential", lambda: c.policy(kg, 0.0, "lex", max_set=1)),
              ("lex B=0.03 +pull", lambda: c.policy(kg, 0.0, "lex", B=0.03, pull_forced=8)))
    samplers = (("product proposal, product twist", dict()), ("product proposal, chain twist", dict(twist=tw)),
                ("chain proposal, chain twist", dict(twist=tw, proposal_twist=tw)))
    res = (("off", dict(ess_min=0.0)), ("defensive", dict()))
    return [Arm(f"{sn} | {pn} | resampling {rn}", mk(), N, n, smc=dict(smc, **kw))
            for sn, mk in scheds for pn, kw in samplers for rn, smc in res]


@arm_set("chainref")
def chainref(c):
    """What should the weights take as the model's law of positions revealed together? Four parallel schedules (`lex` at
    budgets 0.03, 0.3 and 1, and four uniformly random positions per step) by four samplers: product proposal and
    chain proposal, each with the product of marginals and with the chain law as the reference for a co-revealed
    set. The chain twist is the intermediate twist wherever the chain law is used. Scored against the exact p(C)
    and the exact constrained law. Without resampling and with defensive resampling. Automaton constraints only."""
    N, n, kg, tw = c.N, c.nrun, c.kernel("graded"), c.chain_twist()
    scheds = (("lex B=0.03 +pull", lambda: c.policy(kg, 0.0, "lex", B=0.03, pull_forced=8), N), ("lex B=0.3 +pull", lambda: c.policy(kg, 0.0, "lex", B=0.3, pull_forced=8), N),
              ("lex B=1 +pull", lambda: c.policy(kg, 0.0, "lex", B=1.0, pull_forced=8), N), ("fixed 4/step", lambda: "fixed", max(1, N // 4)))
    samplers = (("product proposal, product reference", dict()), ("product proposal, chain reference", dict(reference_twist=tw)),
                ("chain proposal, product reference", dict(twist=tw, proposal_twist=tw)),
                ("chain proposal, chain reference", dict(twist=tw, proposal_twist=tw, reference_twist=tw)))
    res = (("off", dict(ess_min=0.0)), ("defensive", dict()))
    return [Arm(f"{sn} | {pn} | resampling {rn}", mk(), T, n, smc=dict(smc, **kw))
            for sn, mk, T in scheds for pn, kw in samplers for rn, smc in res]


AXIS_ORDERS = ("cheapest", "estimate", "confident", "lex", "sum", "separator", "random")
AXIS_BUDGETS = (0.03, 0.3)


@arm_set("axis")
def axis(c):
    """From cheapest-first to separator-first: seven orders, one token per step and in parallel under two budgets
    (with the partner pull-in), without resampling and with the chain twist plus defensive resampling. `cheapest`
    uses the exact twist (the ideal reference, oracle models only); `estimate` uses the chain twist's disagreement
    with the product twist; `confident` is lowest entropy of the constrained marginal. Automaton constraints only."""
    import twist_error as te
    from .orders import cheapest_order, estimate_order
    N, n, kg, tw = c.N, c.nrun, c.kernel("graded"), c.chain_twist()
    ref = te.Twists(c.den, c.truth.wdfa, getattr(c.truth, "active", None) if getattr(c.truth, "active", None) is not None else [True] * N)
    order = lambda name: {"cheapest": cheapest_order(ref, c.den), "estimate": estimate_order(tw)}.get(name, name)
    modes = [("sequential", dict(max_set=1))] + [(f"B={B} +pull", dict(B=B, pull_forced=8)) for B in AXIS_BUDGETS]
    resampling = (("resampling off", OFF), ("chain twist, defensive resampling", dict(twist=tw)))
    arms = [c.control()]
    arms += [Arm(f"{o} {mn} | {rn}", c.policy(kg, 0.0, order(o), **mk), N, n, smc=smc)
             for o in AXIS_ORDERS for mn, mk in modes for rn, smc in resampling]
    return arms
