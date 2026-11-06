"""Layer-3 adapter: vocabulary mapping, codon classes and the CDS window.

These run without the checkpoint. `scripts/smoke_mrnautilus.py` is the end-to-end check that needs it.
"""
import torch
import pytest

from twistroute_bio.alphabet import CODONS, GENETIC_CODE
from twistroute_bio.models.mrnautilus import (CLS_ID, CODON_OFFSET, CODON_TO_TOKEN, EOS_ID, MASK_ID,
                                              MODEL_CODONS, MODEL_NUCLEOTIDES, TOKEN_TO_CODON,
                                              VOCAB_SIZE, CDSWindow, codon_classes, codons_to_tokens,
                                              nucleotide_tokens, tokens_to_sequence)


class StubModel:
    """Denoiser whose logits encode the input token id, so slicing errors are visible."""
    mask_id, V = MASK_ID, VOCAB_SIZE

    def logits(self, x):
        lg = torch.full(x.shape + (self.V,), -30.0)
        lg.scatter_(-1, x.unsqueeze(-1), 0.0)
        lg[..., self.mask_id] = float("-inf")
        return torch.log_softmax(lg, -1)


def test_vocabulary_is_the_published_one():
    assert VOCAB_SIZE == 86 and len(MODEL_CODONS) == 64 and len(MODEL_NUCLEOTIDES) == 17
    assert (CLS_ID, MASK_ID, EOS_ID, CODON_OFFSET) == (0, 4, 2, 5)


def test_codon_index_maps_round_trip():
    """The model's codon order differs from ours; the two lookup tables must be exact inverses."""
    assert [MODEL_CODONS[CODON_TO_TOKEN[i] - CODON_OFFSET] for i in range(64)] == CODONS
    assert [CODONS[TOKEN_TO_CODON[j]] for j in range(64)] == MODEL_CODONS
    assert sorted(CODON_TO_TOKEN) == list(range(CODON_OFFSET, CODON_OFFSET + 64))


def test_model_codon_table_agrees_with_ncbi_table_1():
    """Cross-check layer 3's vocabulary against layer 2's genetic code: same codons, same amino acids."""
    assert set(MODEL_CODONS) == set(GENETIC_CODE)


def test_nucleotide_tokens_read_uracil_as_thymine():
    assert nucleotide_tokens("ACGU") == nucleotide_tokens("ACGT")
    assert nucleotide_tokens("ACGT") == [69, 70, 71, 72]
    assert nucleotide_tokens("Z") == [3]                                   # <unk>


def test_codon_classes_are_a_bijection_onto_our_alphabet():
    tc = codon_classes()
    assert tc.A == 64 and set(tc.sizes()) == {1}                           # one drawable token per class
    assert len(tc.excluded) == VOCAB_SIZE - 64
    for i, c in enumerate(CODONS):
        lp = torch.full((1, 1, VOCAB_SIZE), -40.0)
        lp[0, 0, CODON_TO_TOKEN[i]] = 0.0
        lp = torch.log_softmax(lp, -1)
        assert int(tc.class_logw(lp)[0, 0].argmax()) == i, c


def test_excluded_tokens_carry_no_class_mass():
    """Probability on nucleotide or special tokens must not leak into any codon class."""
    tc = codon_classes()
    lp = torch.full((1, 1, VOCAB_SIZE), float("-inf"))
    lp[0, 0, 70] = 0.0                                                     # all mass on 'C'
    assert torch.isinf(tc.class_logsumexp(lp)).all()


@pytest.mark.parametrize("utr5,utr3", [("", ""), ("GGGACC", "AATAAA")])
def test_cds_window_layout_and_slicing(utr5, utr3):
    w = CDSWindow(StubModel(), n_codons=4, utr5=utr5, utr3=utr3)
    assert len(w.template) == 1 + len(utr5) + 4 + len(utr3) + 1
    assert w.template[0] == CLS_ID and w.template[-1] == EOS_ID
    assert w.template[w.slice].tolist() == [MASK_ID] * 4

    x = torch.tensor([[CODON_TO_TOKEN[0], MASK_ID, CODON_TO_TOKEN[63], MASK_ID]])
    full = w.embed(x)
    assert full.shape == (1, len(w.template))
    assert full[0, w.slice].tolist() == x[0].tolist()
    assert full[0, :w.start].tolist() == [CLS_ID] + nucleotide_tokens(utr5)

    # the stub puts all mass on the input id, so the returned columns identify the window exactly
    lg = w.logits(x)
    assert lg.shape == (1, 4, VOCAB_SIZE)
    revealed = [0, 2]
    assert [int(lg[0, i].argmax()) for i in revealed] == [x[0, i].item() for i in revealed]


def test_cds_window_does_not_mutate_its_template():
    w = CDSWindow(StubModel(), n_codons=3)
    before = w.template.clone()
    w.embed(torch.full((2, 3), CODON_TO_TOKEN[7]))
    assert torch.equal(w.template, before)


def test_sequence_round_trip():
    idx = torch.tensor([CODONS.index(c) for c in ["ATG", "AAA", "TGG", "TAA"]])
    assert tokens_to_sequence(codons_to_tokens(idx)) == "ATGAAATGGTAA"
