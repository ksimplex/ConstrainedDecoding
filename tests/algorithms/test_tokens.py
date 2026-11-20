"""Token lifting of a character-level constraint and the pair table collapsed from token sequences."""
import itertools, re
import numpy as np
import pytest
from twistroute.compile.tokens import lift_pattern, pair_ratio_table

pytest.importorskip("interegular")
VOCAB = ['{', '}', '"', ':', ',', ' ', 'a', 'b', 'ab', '":', '", "', '{"', '"}', '1', '2', '12', ' 1', 'true', 'tr', 'ue', '', '<end>']
PAT = r'\{"(a|b|ab)": (1|12|true)(, "(a|b|ab)": (1|12|true))?\}'


def test_lifted_automaton_accepts_exactly_the_token_sequences_that_spell_a_match():
    strings = [s if s != '<end>' else '' for s in VOCAB]; end = VOCAB.index('<end>')
    cls, d, info = lift_pattern(PAT, strings, end_ids=[end])
    assert info["classes"] < len(VOCAB) and cls[VOCAB.index('')] != cls[end]                 # the other empty token is dead, not an end token
    rng = np.random.default_rng(0); hits = 0
    for trial in range(4000):
        n = int(rng.integers(3, 9)); ids = rng.integers(0, len(VOCAB), n).tolist()
        if trial % 4 == 0:                                                                   # make matches common: tokenise a valid string at random
            target = rng.choice(['{"a": 1}', '{"ab": true}', '{"b": 12, "a": true}', '{"ab": 1, "ab": 12}']); ids = []
            while target:
                opts = [i for i, s in enumerate(strings) if s and target.startswith(s)]; i = int(rng.choice(opts)); ids.append(i); target = target[len(strings[i]):]
            ids += [end] * int(rng.integers(0, 3))
        k = len(ids)
        while k > 0 and ids[k - 1] == end:
            k -= 1
        text = "".join(strings[i] for i in ids[:k]); clean = all(i != end and strings[i] != '' for i in ids[:k])
        want = clean and re.fullmatch(PAT, text) is not None
        assert d.accepts([int(cls[i]) for i in ids]) == want, (ids, text)
        hits += want
    assert hits > 500


def test_pair_table_is_a_bayesian_update_of_independence():
    rng = np.random.default_rng(1); A = 4; cls = np.arange(A)
    T = np.array([[.7, .1, .1, .1], [.1, .7, .1, .1], [.25, .25, .5, 1e-12], [.25, .25, .25, .25]]); T = T / T.sum(1, keepdims=True)
    xs = []
    for _ in range(300):
        x = [int(rng.integers(A))]
        for _ in range(39):
            x.append(int(rng.choice(A, p=T[x[-1]])))
        xs.append(x)
    K, info = pair_ratio_table(xs, cls, A, lam=None)
    assert pair_ratio_table(xs, cls, A)[1]["lam"] == 1.0
    assert (K > 0).all() and abs(K.sum(1) - 1).max() < 1e-12
    r = K / K[:, [0]]                                                     # ratios within a row: K(a, b) / K(a, 0)
    assert r[2, 3] < 0.05                                                 # expected often, never seen: strongly suppressed, not zero
    assert K[0, 0] / K[0, 1] > 3 * K[3, 0] / K[3, 1]                      # the sticky row shows its dependence
    K2, _ = pair_ratio_table(xs[:1], np.zeros(A, dtype=int) + np.array([0, 0, 0, 1]) * 0 + cls, A + 3)   # classes never seen: neutral rows
    assert np.allclose(K2[A:], 1.0 / (A + 3), atol=0.02) and info["lam"] in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    Kc, _ = pair_ratio_table(xs, np.array([0, 0, 1, 1]), 2, lam=1.0)      # collapsing the token-level corpus to two classes is just recounting
    n = np.zeros((2, 2))
    for x in xs:
        y = np.array([0, 0, 1, 1])[np.array(x)]; np.add.at(n, (y[:-1], y[1:]), 1)
    e = np.outer(n.sum(1), n.sum(0)) / n.sum(); want = (n + 1) / (e + 1); assert np.allclose(Kc, want / want.sum(1, keepdims=True))
