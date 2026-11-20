"""Exact path laws on toy problems (no sampling noise): the Feynman-Kac terminal law equals the Doob law and
Z_hat / h0 = 1 to machine precision for every route family, including an ambiguous grammar, while uncorrected
step-exact decoding is measurably biased."""
import numpy as np
import pytest
from twistroute import compile as C
from twistroute.decode.exact import Decoder, ToyDenoiser, tv

FAMILIES = {
    "regular": (C.dfa_forbidden([(1, 1), (0, 0, 0)], 2), 6, 2),
    "counting": (C.dfa_exactly_k(0, 3, 2), 6, 2),
    "copy factor graph": (C.CopyFG(3, 2), 6, 2),
    "Dyck-2 chart": (C.Inside(*C.DYCK2), 4, 4),
    "ambiguous Dyck-1 chart": (C.Inside(*C.DYCK1_AMBIGUOUS), 6, 2),
    "arborescence": (C.Arborescence(3), 3, 4),
}


@pytest.mark.parametrize("name", list(FAMILIES))
def test_fk_equals_doob(name):
    route, L, A = FAMILIES[name]
    laws = Decoder(route, ToyDenoiser(L, A, eps=1.0, seed=1), L, A).exact_laws()
    assert laws["support_ok"]
    assert tv(laws["fk"], laws["doob"]) < 1e-10
    assert abs(laws["zhat_over_h0"] - 1.0) < 1e-10


def test_step_exact_is_biased_and_frozen_denoiser_is_not():
    route, L, A = FAMILIES["regular"]
    biased = Decoder(route, ToyDenoiser(L, A, eps=1.0, seed=1), L, A).exact_laws()
    frozen = Decoder(route, ToyDenoiser(L, A, eps=0.0, seed=1), L, A).exact_laws()
    assert tv(biased["step"], biased["doob"]) > 0.02
    assert tv(frozen["step"], frozen["doob"]) < 1e-10      # no re-conditioning -> no trajectory bias
