"""The chain law r(. | x_t) and its passes over an automaton (notes, eq. 11 to 13).

Notation: mu_i(a) is the model's class marginal at position i (one-hot at unmasked positions),
K the pair table, A the number of classes, S the number of automaton states.

    transitions  eq. (11): for each adjacent pair the joint Q_i(a, b) = u_i(a) K(a, b) v_i(b)
                 with marginals mu_i and mu_{i+1}, returned as the conditional
                 Q_i(a, b) / mu_i(a) of eq. (12).
    log_mass     log r(C | x_t): a forward pass over pairs (class at i, automaton state).
    log_mass_with
                 log r(x_U = c, C | x_t): the same pass with the classes at U fixed. The
                 chain's transitions are not changed by fixing them.
    sample       a draw from r(. | x_t, C), eq. (13): a backward pass stores, for every
                 position, class and state, the chance of ending in an accepting state; a
                 forward sweep then draws the classes one position at a time.

Cost: O(L A |E|) for the forward pass, O(L A^2 S) time and O(L A S) memory for the sampler.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

TINY = 1e-300


class ChainLaw:
    def __init__(
        self,
        automaton: Any,
        pair_table: torch.Tensor,
        *,
        active: Optional[torch.Tensor] = None,
        matching_sweeps: int = 5,
        device: str = "cpu",
    ) -> None:
        """automaton: a `twistroute.compile.WDFA` (edges (state, class, next state), start state
        q0, accepting states F); active: (L,) bool, the positions the automaton reads."""
        edges = torch.tensor(automaton.edges, dtype=torch.long).reshape(-1, 3)
        self.edge_from = edges[:, 0].to(device)
        self.edge_class = edges[:, 1].to(device)
        self.edge_to = edges[:, 2].to(device)
        self.n_states = automaton.n
        self.start = automaton.q0
        self.n_classes = automaton.A
        self.accepting = torch.zeros(self.n_states, dtype=torch.float64, device=device)
        self.accepting[sorted(automaton.F)] = 1.0
        self.pair_table = torch.as_tensor(pair_table, dtype=torch.float64, device=device)
        if self.pair_table.shape != (self.n_classes, self.n_classes) or bool(
            (self.pair_table <= 0).any()
        ):
            raise ValueError("the pair table must be (A, A) with positive entries")
        self.matching_sweeps = int(matching_sweeps)
        self.active = (
            None if active is None else torch.as_tensor(active, dtype=torch.bool, device=device)
        )
        # next_state[s, a]: the state after reading class a in state s; -1 if the automaton rejects
        self.next_state = torch.full(
            (self.n_states, self.n_classes), -1, dtype=torch.long, device=device
        )
        self.next_state[self.edge_from, self.edge_class] = self.edge_to

    # ------------------------------------------------------------------ eq. (11) and (12)
    def transitions(self, mu: torch.Tensor) -> torch.Tensor:
        """(B, L, A) marginals -> (B, L - 1, A, A): the chance of class b at i + 1 given class a
        at i, for every adjacent pair."""
        table = self.pair_table
        if self.matching_sweeps <= 0:  # first-order ratio formula, no matching
            typical = torch.einsum("bia,ac->bic", mu[:, :-1], table).clamp_min(TINY)
            ratio = mu[:, 1:, None, :] * table[None, None] / typical[:, :, None, :]
            return ratio / ratio.sum(-1, keepdim=True).clamp_min(TINY)
        rows = mu[:, :-1]  # mu_i, the marginal over a
        columns = mu[:, 1:]  # mu_{i+1}, the marginal over b
        v = torch.ones_like(columns)
        for _ in range(self.matching_sweeps):  # alternate: fit the rows, then the columns
            u = rows / torch.einsum("ac,bic->bia", table, v).clamp_min(TINY)
            v = columns / torch.einsum("bia,ac->bic", u, table).clamp_min(TINY)
        # Q_i(a, b) / mu_i(a) = K(a, b) v(b) / sum_b' K(a, b') v(b'): rows are fitted exactly
        row_sum = torch.einsum("ac,bic->bia", table, v).clamp_min(TINY)
        conditional = table[None, None] * v[:, :, None, :] / row_sum[..., None]
        return conditional / conditional.sum(-1, keepdim=True).clamp_min(TINY)

    @staticmethod
    def normalised(prediction: torch.Tensor) -> torch.Tensor:
        """(B, L, A) log-weights -> marginals mu."""
        weights = torch.exp(prediction.double())
        return weights / weights.sum(-1, keepdim=True).clamp_min(TINY)

    # ------------------------------------------------------------------ forward pass
    def _read(self, mass: torch.Tensor, position: int) -> torch.Tensor:
        """mass[b, a, s]: mass with class a at `position` and the automaton in state s before
        reading it. Returns the mass by the state after reading a."""
        if self.active is not None and not bool(self.active[position]):
            return mass
        moved = mass[:, self.edge_class, self.edge_from]  # (B, E)
        out = torch.zeros(
            mass.shape[0], self.n_classes * self.n_states, dtype=mass.dtype, device=mass.device
        )
        return out.index_add(1, self.edge_class * self.n_states + self.edge_to, moved).view(
            mass.shape
        )

    def log_mass(
        self, prediction: torch.Tensor, log_tilt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """(B, L, A) prediction -> (B,) float64 log r(C | x_t). `log_tilt` (B, L, A) multiplies
        the mass of class a at position i by exp(log_tilt[b, i, a]) (used to fix classes)."""
        mu = self.normalised(prediction)
        conditional = self.transitions(mu)
        n_rows, length, _ = mu.shape
        first = mu[:, 0] if log_tilt is None else mu[:, 0] * torch.exp(log_tilt[:, 0])
        start = torch.zeros(self.n_states, dtype=torch.float64, device=mu.device)
        start[self.start] = 1.0
        forward = self._read(first[:, :, None] * start[None, None, :], 0)  # (B, A, S)
        log_scale = torch.zeros(n_rows, dtype=torch.float64, device=mu.device)
        for i in range(length - 1):
            total = forward.sum((1, 2))
            log_scale = log_scale + torch.log(total)
            forward = forward / total.clamp_min(TINY)[:, None, None]
            moved = torch.einsum("baq,bac->bcq", forward, conditional[:, i])
            if log_tilt is not None:
                moved = moved * torch.exp(log_tilt[:, i + 1])[:, :, None]
            forward = self._read(moved, i + 1)
        return log_scale + torch.log((forward.sum(1) * self.accepting[None]).sum(-1))

    def log_mass_with(
        self,
        prediction: torch.Tensor,
        positions: torch.Tensor,
        classes: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(B,) log r(x_U = classes, C | x_t) for U = positions (B, n); `valid` marks the
        columns that count."""
        log_tilt = torch.zeros(prediction.shape, dtype=torch.float64, device=prediction.device)
        keep = torch.ones_like(positions, dtype=torch.bool) if valid is None else valid
        row = torch.arange(positions.shape[0], device=positions.device)[:, None].expand_as(
            positions
        )[keep]
        log_tilt[row, positions[keep]] = float("-inf")
        log_tilt[row, positions[keep], classes[keep]] = 0.0
        return self.log_mass(prediction, log_tilt)

    def marginals(self, prediction: torch.Tensor) -> torch.Tensor:
        """(B, L, A) float64: r(x_i = a | x_t, C); zeros where C has no mass."""
        with torch.enable_grad():
            log_tilt = torch.zeros(
                prediction.shape, dtype=torch.float64, device=prediction.device, requires_grad=True
            )
            (grad,) = torch.autograd.grad(
                self.log_mass(prediction.detach(), log_tilt).sum(), log_tilt
            )
        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    # ------------------------------------------------------------------ eq. (13)
    def sample(
        self,
        prediction: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        chunk: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw (B, L) class sequences from r(. | x_t, C); also returns (B,) log r(C | x_t). A row
        where C has no mass returns -inf and an arbitrary sequence."""
        draws = []
        log_masses = []
        for low in range(0, prediction.shape[0], chunk):
            draw, log_mass = self._sample(prediction[low : low + chunk], generator)
            draws.append(draw)
            log_masses.append(log_mass)
        return torch.cat(draws), torch.cat(log_masses)

    def _sample(
        self, prediction: torch.Tensor, generator: Optional[torch.Generator]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.normalised(prediction)
        conditional = self.transitions(mu)
        n_rows, length, n_classes = mu.shape
        n_states = self.n_states
        device = mu.device
        after = self.next_state.T  # (A, S)
        allowed = (after >= 0).to(torch.float64)[None]
        after_index = after.clamp_min(0)[None].expand(n_rows, n_classes, n_states)

        def before_reading(value: torch.Tensor, position: int) -> torch.Tensor:
            """value[b, a, s']: indexed by the state after reading a -> by the state before."""
            if self.active is not None and not bool(self.active[position]):
                return value
            return torch.gather(value, 2, after_index) * allowed

        # backward[b, i, a, s]: chance of acceptance given class a at i and state s before
        # reading it, up to a scale per position
        backward = torch.zeros(
            n_rows, length, n_classes, n_states, dtype=torch.float64, device=device
        )
        accept = self.accepting[None, None, :].expand(n_rows, n_classes, n_states)
        backward[:, length - 1] = before_reading(accept, length - 1)
        log_scale = torch.zeros(n_rows, dtype=torch.float64, device=device)
        for i in range(length - 2, -1, -1):
            value = torch.einsum("bac,bcq->baq", conditional[:, i], backward[:, i + 1])
            scale = value.amax((1, 2))
            log_scale = log_scale + torch.log(scale)
            value = value / scale.clamp_min(TINY)[:, None, None]
            backward[:, i] = before_reading(value, i)
        rows = torch.arange(n_rows, device=device)

        def draw(weights: torch.Tensor) -> torch.Tensor:
            total = weights.sum(-1, keepdim=True)
            weights = torch.where(
                total > 0, weights, torch.ones_like(weights)
            )  # no mass: arbitrary class
            return torch.multinomial(weights, 1, generator=generator).squeeze(-1)

        def move(state: torch.Tensor, cls: torch.Tensor, position: int) -> torch.Tensor:
            if self.active is not None and not bool(self.active[position]):
                return state
            return self.next_state[state, cls].clamp_min(0)

        first = mu[:, 0] * backward[:, 0, :, self.start]
        log_mass = log_scale + torch.log(first.sum(-1))
        sequence = torch.zeros(n_rows, length, dtype=torch.long, device=device)
        state = torch.full((n_rows,), self.start, dtype=torch.long, device=device)
        sequence[:, 0] = draw(first)
        state = move(state, sequence[:, 0], 0)
        for i in range(length - 1):
            sequence[:, i + 1] = draw(
                conditional[rows, i, sequence[:, i]] * backward[rows, i + 1, :, state]
            )
            state = move(state, sequence[:, i + 1], i + 1)
        return sequence, log_mass
