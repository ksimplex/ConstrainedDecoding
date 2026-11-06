"""
mRNAutilus adapter (layer 3): the published masked discrete diffusion model for full-length mRNA,
wrapped as a denoiser that `twistroute.decode` can drive.

Checkpoint `atombio/mrnautilus` (CC BY-NC 4.0). The network is RiNALMo "mega" (30 blocks, 640-dim,
20 heads, rotary positions, SwiGLU) with a masked-LM head; see `twistroute_bio.models._rinalmo`,
whose transformer block is verified bit-identical to upstream's (docs/mrna_model_validation.md).

Vocabulary (upstream `Alphabet_uni`, V = 86)
    0-4     <cls> <pad> <eos> <unk> <mask>
    5-68    the 64 codons, in the order of upstream's CODON_DICT -- NOT lexicographic, and NOT the
            order of `twistroute_bio.alphabet.CODONS`. `MODEL_CODONS` below records it verbatim.
    69-85   the 17 FASTA nucleotide codes A C G T I R Y K M S W B D H V N -
Input layout: [cls] + 5' UTR as nucleotide tokens + CDS as codon tokens + 3' UTR as nucleotide
tokens + [eos]. Uracil is read as thymine. There is no time conditioning (`time_conditioning` is
False in the model code), so the denoiser takes tokens alone.

Two objects live here:

    MRNAutilus      the whole-transcript denoiser: .mask_id, .V, .logits(x) -> (B, N, 86) log-probs
                    with the mask excluded (the SUBS parameterisation of Sahoo et al.,
                    arXiv:2406.07524, Sec. 3.2).
    CDSWindow       a denoiser *view* over just the CDS codon positions of a fixed transcript. The
                    UTRs, <cls> and <eos> stay clamped to their given values; `logits` embeds the
                    window into the full transcript, runs the net once, and returns the window's
                    columns. Layer 1 then sees an ordinary N-position problem and needs no change.

Together with `codon_classes()` these give the decoder a 64-symbol alphabet indexed exactly like
`twistroute_bio.alphabet.CODONS`, which is what `twistroute_bio.constraints.compile_cds` builds its
routes over.

Conditioning note. `codon_classes()` excludes every non-codon token, so CDS positions can only be
filled by codons. The sampler therefore targets the native law conditioned on *both* "every CDS
position is a codon" and the constraint. That is the intended object -- a CDS that emitted a bare
nucleotide token would not be a coding sequence -- but it is a conditioning, not a free choice, and
h0 is a probability under that joint event.
"""
import torch

from twistroute.models.classes import TokenClasses
from ..alphabet import CODONS
from ._rinalmo import load_rinalmo

# Upstream `src/rinalmo/data/constants.py`: CODON_TOKENS = list(CODON_DICT.keys()), RNA_TOKENS.
MODEL_CODONS = (
    "ATA ATC ATT ATG ACA ACC ACG ACT AAC AAT AAA AAG AGC AGT AGA AGG "
    "CTA CTC CTG CTT CCA CCC CCG CCT CAC CAT CAA CAG CGA CGC CGG CGT "
    "GTA GTC GTG GTT GCA GCC GCG GCT GAC GAT GAA GAG GGA GGC GGG GGT "
    "TCA TCC TCG TCT TTC TTT TTA TTG TAC TAT TAA TAG TGC TGT TGA TGG"
).split()
MODEL_NUCLEOTIDES = list("ACGTI") + list("RYKMSWBDHVN") + ["-"]

CLS_ID, PAD_ID, EOS_ID, UNK_ID, MASK_ID = 0, 1, 2, 3, 4
CODON_OFFSET, NUC_OFFSET = 5, 69
VOCAB_SIZE = 5 + len(MODEL_CODONS) + len(MODEL_NUCLEOTIDES)

assert len(MODEL_CODONS) == 64 and len(set(MODEL_CODONS)) == 64
assert VOCAB_SIZE == 86

#: model token id of each codon, indexed by `twistroute_bio.alphabet.CODONS`
CODON_TO_TOKEN = [CODON_OFFSET + MODEL_CODONS.index(c) for c in CODONS]
#: inverse: canonical codon index of each model codon token, indexed from CODON_OFFSET
TOKEN_TO_CODON = [CODONS.index(c) for c in MODEL_CODONS]


def nucleotide_tokens(seq):
    """Nucleotide string -> model token ids (U read as T, unknown characters -> <unk>)."""
    s = seq.upper().replace("U", "T")
    return [NUC_OFFSET + MODEL_NUCLEOTIDES.index(ch) if ch in MODEL_NUCLEOTIDES else UNK_ID for ch in s]


def codon_classes(device=None):
    """TokenClasses mapping the model vocabulary onto the 64 `CODONS` symbols.

    Every non-codon token is excluded, so it is never drawn and contributes to no class mass.
    """
    cls = torch.zeros(VOCAB_SIZE, dtype=torch.long)
    for j, c in enumerate(MODEL_CODONS):
        cls[CODON_OFFSET + j] = TOKEN_TO_CODON[j]
    excluded = tuple(v for v in range(VOCAB_SIZE) if not CODON_OFFSET <= v < CODON_OFFSET + 64)
    tc = TokenClasses(cls, 64, excluded=excluded)
    return tc.to(device) if device is not None else tc


class MRNAutilus:
    """Whole-transcript denoiser. Interface: `mask_id`, `V`, `logits(x)`."""

    mask_id = MASK_ID
    V = VOCAB_SIZE

    def __init__(self, ckpt_path, device="cpu", dtype=torch.float32, impl=None):
        if impl is None:
            impl = "flash" if str(device).startswith("cuda") else "torch"
        self.net = load_rinalmo(ckpt_path, impl=impl, device=device, dtype=dtype)
        self.device, self.dtype, self.impl = device, dtype, impl

    def info(self):
        return dict(V=self.V, mask_id=self.mask_id, impl=self.impl, device=str(self.device),
                    time_conditioning=False)

    @torch.no_grad()
    def logits(self, x):
        """(B, N) token ids -> (B, N, V) float32 log-probabilities with the mask excluded."""
        with torch.autocast("cuda", dtype=self.dtype, enabled=str(self.device).startswith("cuda")):
            lg = self.net(x.to(self.device))
        lg = lg.float()
        lg[..., self.mask_id] = float("-inf")
        return torch.log_softmax(lg, -1)

    def cds_window(self, n_codons, utr5="", utr3=""):
        """A `CDSWindow` denoiser over `n_codons` CDS positions, with the UTRs held fixed."""
        return CDSWindow(self, n_codons, utr5, utr3)


class CDSWindow:
    """Denoiser view over the CDS codon positions of a transcript whose context is fixed.

    `logits(x)` takes (B, n_codons) *model token ids* for the CDS positions only (masked positions
    hold `mask_id`), places them inside the full transcript, and returns the CDS columns.
    """

    def __init__(self, model, n_codons, utr5="", utr3=""):
        self.model, self.n_codons = model, int(n_codons)
        self.mask_id, self.V = model.mask_id, model.V
        left = [CLS_ID] + nucleotide_tokens(utr5)
        right = nucleotide_tokens(utr3) + [EOS_ID]
        self.start = len(left)
        self.template = torch.tensor(left + [MASK_ID] * self.n_codons + right, dtype=torch.long)
        self.slice = slice(self.start, self.start + self.n_codons)

    def __len__(self):
        return self.n_codons

    def info(self):
        return dict(n_codons=self.n_codons, transcript_len=len(self.template),
                    cds_start=self.start, **self.model.info())

    def embed(self, x):
        """(B, n_codons) CDS tokens -> (B, transcript_len) full transcript tokens."""
        full = self.template.to(x.device).unsqueeze(0).expand(x.shape[0], -1).clone()
        full[:, self.slice] = x
        return full

    def logits(self, x):
        return self.model.logits(self.embed(x))[:, self.slice, :]


def codons_to_tokens(idx):
    """Canonical codon indices (any tensor shape) -> model token ids."""
    lut = torch.tensor(CODON_TO_TOKEN, device=idx.device)
    return lut[idx]


def tokens_to_sequence(tok):
    """A 1-D tensor of model CDS token ids -> the nucleotide string."""
    return "".join(MODEL_CODONS[int(v) - CODON_OFFSET] for v in tok)
