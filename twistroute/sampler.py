"""The weighted sampler (notes, section 2.3, eq. 4 to 6).

`unconstrained_step` is the model's own denoising step, eq. (4). `weighted_sampler` runs K
particles through constrained denoising steps, multiplies their weights by eq. (5) after every
step (eq. 6), resamples when the weights have become uneven, and returns the weighted
particles. The twist with its proposal (`twistroute.twists`), the schedule
(`twistroute.schedules`) and the resampling rule (`twistroute.resampling`) are arguments.

Weights start at psi(x_T) for the fully masked sequence, not at 1, so that the mean final
weight estimates P(x_0 in C) under the unconstrained process: `log_Z_hat`. This multiplies all
weights by the same number and does not change which particle is returned.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from .constraints.base import Constraint
from .resampling import Resampling, effective_sample_size, log_mean_exp, resample
from .schedules.base import RandomOrder, Schedule, as_schedule
from .state import State, Step
from .twists.base import NEG_INF, Twist
from .twists.product import ProductTwist

Observer = Callable[[int, State, Step, torch.Tensor], None]


@dataclass
class SamplerResult:
    """What `weighted_sampler` returns; `runs` independent runs of K particles.

    log_Z_hat    (runs,) log of the estimate of P(x_0 in C); unbiased on the natural scale
    tokens       (runs, K, L) the particles
    classes      (runs, K, L) their token classes
    log_weights  (runs, K) final log-weights (-inf: weight zero)
    steps        (runs, K) steps in which the particle's ancestry unmasked something, which is
                 its number of model calls
    resamplings  (runs,) number of resampling events
    ess_min      (runs,) smallest effective sample size seen
    support      counts of support violations (all zero unless arithmetic lost the support)
    """

    log_Z_hat: np.ndarray
    tokens: np.ndarray
    classes: np.ndarray
    log_weights: np.ndarray
    steps: np.ndarray
    resamplings: np.ndarray
    ess_min: np.ndarray
    support: Dict[str, int]

    def returned(self, rng: np.random.Generator) -> np.ndarray:
        """(runs, L) the sampler's output: one particle per run, drawn in proportion to its
        weight (notes, end of the weighted-sampler paragraph). A run whose particles all have
        weight zero has no valid output; its row is -1."""
        n_runs, _, length = self.tokens.shape
        out = np.full((n_runs, length), -1, dtype=self.tokens.dtype)
        for run in range(n_runs):
            finite = np.isfinite(self.log_weights[run])
            if not finite.any():
                continue
            weights = np.where(
                finite, np.exp(self.log_weights[run] - self.log_weights[run][finite].max()), 0.0
            )
            out[run] = self.tokens[run, rng.choice(len(weights), p=weights / weights.sum())]
        return out


def model_prediction(
    model: Any,
    token_classes: Any,
    constraint: Constraint,
    tokens: torch.Tensor,
    classes: torch.Tensor,
) -> State:
    """Call the model at x_t and package its prediction (notes, eq. 1)."""
    log_p = model.logits(tokens)
    return State(
        tokens,
        classes,
        log_p,
        token_classes.class_logw(log_p),
        model.mask_id,
        constraint,
        token_classes,
    )


def commit(
    state: State,
    completion: torch.Tensor,
    unmask: torch.Tensor,
    valid: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
) -> Step:
    """Unmask the positions U with the classes of `completion`: at each, a token is drawn from
    the model's prediction within the class. This is the token-level step; the class at U is
    what the constraint sees."""
    n_rows, n_unmask = unmask.shape
    token_classes = state.token_classes
    drawn = torch.gather(completion, 1, unmask)
    log_p_rows = torch.gather(
        state.log_p, 1, unmask[..., None].expand(-1, -1, state.log_p.shape[-1])
    )
    in_class = token_classes.onehot.T[drawn] > 0
    log_p_rows = torch.where(in_class, log_p_rows, torch.full_like(log_p_rows, NEG_INF))
    probabilities = torch.softmax(log_p_rows, -1).reshape(-1, log_p_rows.shape[-1])
    new_tokens = torch.multinomial(probabilities, 1, generator=generator).reshape(n_rows, n_unmask)
    if valid is not None:  # padded columns keep what is there
        new_tokens = torch.where(valid, new_tokens, torch.gather(state.tokens, 1, unmask))
        drawn = torch.where(valid, drawn, torch.gather(state.classes, 1, unmask))
    tokens = state.tokens.clone()
    tokens.scatter_(1, unmask, new_tokens)
    classes = state.classes.clone()
    classes.scatter_(1, unmask, drawn)
    return Step(unmask, valid, tokens, classes, drawn)


@torch.no_grad()
def unconstrained_step(
    state: State,
    schedule: Schedule,
    step: int,
    n_steps: int,
    generator: Optional[torch.Generator] = None,
) -> Step:
    """One step of the model's own sampler, eq. (4): U from the schedule, then every position
    of U independently from the model's prediction at x_t."""
    unmask, valid = schedule.unmask(state, step, n_steps, generator)
    n_rows, length, n_classes = state.log_q.shape
    probabilities = torch.softmax(state.log_q, -1).reshape(-1, n_classes)
    completion = torch.multinomial(probabilities, 1, generator=generator).reshape(n_rows, length)
    return commit(state, completion, unmask, valid, generator)


@torch.no_grad()
def weighted_sampler(
    model: Any,
    token_classes: Any,
    constraint: Constraint,
    length: int,
    n_steps: int,
    *,
    runs: int,
    particles: int,
    twist: Optional[Twist] = None,
    schedule: Any = None,
    resampling: Resampling = Resampling(),
    generator: Optional[torch.Generator] = None,
    observe: Optional[Observer] = None,
) -> SamplerResult:
    """Run `runs` independent weighted samplers of `particles` particles each.

    model          has .logits(tokens) -> (P, L, V) log p_theta(x_0,i = v | x_t) and .mask_id
    token_classes  the map from tokens to the constraint's classes
    constraint     a `twistroute.constraints.Constraint`
    twist          a `twistroute.twists.Twist` (default: the product twist)
    schedule       a `twistroute.schedules.Schedule` (default: `RandomOrder`)
    resampling     a `twistroute.resampling.Resampling`
    observe        called as observe(step, state, step_taken, log_weights) after every step and
                   before resampling, with log_weights (runs, particles); for diagnostics

    One particle and no weights is the constrained sampler of Dang and Ermon
    (arXiv:2607.07026); the product twist with plain resampling is TWISTER (arXiv:2609.35609).
    """
    twist = ProductTwist() if twist is None else twist
    schedule = RandomOrder() if schedule is None else as_schedule(schedule)
    device = token_classes.onehot.device
    if generator is not None and torch.device(generator.device).type != torch.device(device).type:
        raise ValueError(f"the generator is on {generator.device} but the model is on {device}")
    n_rows = runs * particles
    row_of_run = particles * torch.arange(runs, device=device)[:, None]

    tokens = torch.full((n_rows, length), model.mask_id, dtype=torch.long, device=device)
    classes = torch.zeros(n_rows, length, dtype=torch.long, device=device)
    state = model_prediction(model, token_classes, constraint, tokens, classes)
    state.log_psi = twist.log_psi(state)
    log_weights = state.log_psi  # W_j = psi(x_T)
    log_Z_hat = torch.zeros(runs, dtype=torch.float64, device=device)
    ess_min = torch.full((runs,), float(particles), device=device, dtype=torch.float64)
    resamplings = torch.zeros(runs, device=device)
    steps = torch.zeros(n_rows, dtype=torch.long, device=device)
    support = {"proposal_off_support": 0, "twist_collapse": 0, "nan_weight": 0}

    for step_index in range(n_steps):
        last = step_index == n_steps - 1
        # a constrained denoising step x_t -> x_s ~ M(. | x_t)
        if twist.needs_unmask_set:
            unmask, valid = schedule.unmask(state, step_index, n_steps, generator)
            completion, extra = twist.complete(state, generator, unmask, valid)
        else:
            completion, extra = twist.complete(state, generator)
            unmask, valid = schedule.unmask(state, step_index, n_steps, generator)
        step = commit(state, completion, unmask, valid, generator)
        step.extra = extra

        # psi(x_s): the twist at the new prediction, or 1[x_0 in C] when the sequence is complete
        if last:
            if bool((step.tokens == model.mask_id).any()):
                raise ValueError("the schedule left masked positions after the last step")
            new_state = State(
                step.tokens, step.classes, None, None, model.mask_id, constraint, token_classes
            )
            log_psi_next = new_state.in_constraint()
            new_state.cache["log_product"] = log_psi_next
        else:
            new_state = model_prediction(
                model, token_classes, constraint, step.tokens, step.classes
            )
            log_psi_next = twist.log_psi(new_state)
        new_state.log_psi = log_psi_next
        finished = bool(new_state.complete.all())

        # eq. (6): W_j <- W_j w(x_t, x_s)
        log_w, log_fixed = twist.log_weight(state, step, log_psi_next)
        alive = torch.isfinite(log_weights) & torch.isfinite(state.cache["log_product"])
        support["proposal_off_support"] += int((alive & ~torch.isfinite(log_fixed)).sum())
        if not last:
            collapsed = alive & torch.isfinite(log_fixed) & ~torch.isfinite(log_psi_next)
            support["twist_collapse"] += int(collapsed.sum())
        log_weights = log_weights + log_w
        support["nan_weight"] += int(torch.isnan(log_weights).sum())
        steps = steps + (step.tokens != state.tokens).any(-1).long()
        if last:  # keep the last prediction: nothing follows
            new_state.log_p, new_state.log_q = state.log_p, state.log_q
        if observe is not None:
            observe(step_index, state, step, log_weights.view(runs, particles))
        state = new_state

        by_run = log_weights.view(runs, particles)
        normalised = torch.nan_to_num(torch.softmax(by_run, -1), nan=1.0 / particles)
        ess_min = torch.minimum(
            ess_min, torch.nan_to_num(effective_sample_size(normalised), nan=0.0)
        )
        if finished:
            break
        if last or particles == 1:
            continue
        fired, survivors, by_run, log_mean = resample(by_run, resampling, generator)
        if not fired.any():
            continue
        log_Z_hat = log_Z_hat + log_mean
        rows = (survivors + row_of_run).reshape(-1)
        state.take(rows)
        steps = steps[rows]
        log_weights = by_run.reshape(-1)
        resamplings = resamplings + fired.float()

    by_run = log_weights.view(runs, particles)
    log_Z_hat = log_Z_hat + log_mean_exp(by_run, -1)
    problems = {name: count for name, count in support.items() if count}
    if problems:
        warnings.warn(
            f"support invariants violated: {problems}; the weights are no longer exact",
            RuntimeWarning,
            stacklevel=2,
        )
    shape = (runs, particles, length)
    return SamplerResult(
        log_Z_hat=log_Z_hat.cpu().numpy(),
        tokens=state.tokens.view(shape).cpu().numpy(),
        classes=token_classes.token_class[state.tokens].view(shape).cpu().numpy(),
        log_weights=by_run.cpu().numpy(),
        steps=steps.view(runs, particles).cpu().numpy(),
        resamplings=resamplings.cpu().numpy(),
        ess_min=ess_min.cpu().numpy(),
        support=support,
    )
