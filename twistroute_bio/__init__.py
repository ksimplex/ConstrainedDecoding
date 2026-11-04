"""
Nucleotide and mRNA constraint layer on top of twistroute (layer 2 of docs/mrna_specs.md).

Everything here builds ordinary twistroute routes over an explicit alphabet; nothing depends on a particular
model. Model adapters live in twistroute_bio.models.
"""
from .alphabet import (NUCLEOTIDES, CODONS, GENETIC_CODE, AMINO_ACIDS, nuc_index, codon_index, codon_words,
                       amino_acid_of, synonymous_mask, translate, lift_to_codons)
