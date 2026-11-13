"""
Fetch the human coding sequences the mRNA experiments run on, from NCBI RefSeq.

Two outputs. `corpus` is a few hundred CDSs used only to estimate the codon frequency table, following
Clarke & Clark's own practice of estimating from the organism's own sequences rather than hardcoding a
table; the estimate was checked against the repository's Kazusa-derived table at Pearson r = 0.995, with
the modal codon agreeing for 17 of 18 multi-codon amino acids. `panel` is the gene set the experiments
iterate over: one CDS per gene symbol, 90-260 codons, ordered by length, preferring genes whose natural
sequence already violates the restriction-site constraint so the constraint is active.

    python experiments/fetch_human_cds.py results/
"""
import collections
import itertools
import json
import re
import sys
import time
import urllib.parse
import urllib.request

E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SITES = ("GAATTC", "GGATCC", "AAGCTT", "CTCGAG", "CATATG", "TCTAGA", "GTCGAC", "GCGGCCGC")
CODONS = ["".join(t) for t in itertools.product("ACGT", repeat=3)]
_TCAG, _AAS = "TCAG", "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODE = {a + b + c: _AAS[16 * i + 4 * j + k]
        for i, a in enumerate(_TCAG) for j, b in enumerate(_TCAG) for k, c in enumerate(_TCAG)}

#: The 16-gene panel, pinned by RefSeq accession. Neither a search nor a gene symbol is a sufficient pin:
#: the search returns a different gene set on a different day, and several of these symbols have transcript
#: isoforms of different lengths. The accessions below are those of the transcripts the 2026-11-07 GPU run
#: used, recovered from that job's own input file; the first six are the genes it completed before its
#: scheduler limit (results/mrna_correction_size.csv).
PANEL = (("CCL4L1", "NM_207007.4"),
         ("PDE6D", "NM_001291018.2"),
         ("NDUFA2", "NM_002488.5"),
         ("CRYAB", "NM_001368246.1"),
         ("IFT43", "NM_001255995.3"),
         ("B2M", "NM_004048.4"),
         ("KISS1", "NM_002256.4"),
         ("HBB", "NM_000518.5"),
         ("BCL11A", "NM_001405736.1"),
         ("NMNAT1", "NM_001297779.2"),
         ("NDUFAF2", "NM_174889.5"),
         ("FTL", "NM_000146.4"),
         ("KRAS", "NM_004985.5"),
         ("CRYBA4", "NM_001886.3"),
         ("CSNK2B", "NM_001282385.2"),
         ("TSFM", "NM_001172697.2"))


def _get(ep, **q):
    return urllib.request.urlopen(E + ep + "?" + urllib.parse.urlencode(q), timeout=180).read().decode()


def _parse(gb, lo=0, hi=10 ** 9):
    out = []
    for r in gb.split("\n//\n"):
        if "LOCUS" not in r:
            continue
        acc = re.search(r"VERSION\s+(\S+)", r)
        gene = re.search(r'/gene="([^"]+)"', r)
        cds = re.search(r"\n     CDS\s+(\d+)\.\.(\d+)\n", r)
        seq = re.search(r"\nORIGIN(.*)$", r, re.S)
        if not (acc and cds and seq):
            continue
        s = "".join(re.findall(r"[acgtn]+", seq.group(1))).upper()
        c = s[int(cds.group(1)) - 1:int(cds.group(2))]
        if len(c) % 3 or set(c) - set("ACGT") or not (lo <= len(c) // 3 <= hi):
            continue
        out.append(dict(acc=acc.group(1), gene=gene.group(1) if gene else "?", cds=c, n_codons=len(c) // 3))
    return out


def fetch(term, retmax, lo=0, hi=10 ** 9):
    ids = json.loads(_get("esearch.fcgi", db="nuccore", retmax=retmax, retmode="json", term=term))
    ids = ids["esearchresult"]["idlist"]
    recs = []
    for i in range(0, len(ids), 100):
        recs.append(_get("efetch.fcgi", db="nuccore", id=",".join(ids[i:i + 100]),
                         rettype="gb", retmode="text"))
        time.sleep(0.4)
    return _parse("\n".join(recs), lo, hi)


def codon_table(cds_list, pseudocount=1.0):
    cnt = collections.Counter()
    for h in cds_list:
        cnt.update(h["cds"][i:i + 3] for i in range(0, len(h["cds"]), 3))
    syn = collections.defaultdict(list)
    for c in CODONS:
        syn[CODE[c]].append(c)
    counts = {c: cnt.get(c, 0) + pseudocount for c in CODONS}
    tot = sum(counts.values())
    return {"abs": {c: 1000.0 * counts[c] / tot for c in CODONS},
            "rel": {c: counts[c] / sum(counts[d] for d in syn[CODE[c]]) for c in CODONS},
            "n_codons": int(sum(cnt.values()))}


def main(outdir, n_panel=16):
    base = 'srcdb_refseq_known[PROP] AND "Homo sapiens"[Organism] AND biomol_mrna[PROP]'
    # The corpus is a search, not a pinned list, so the table it produces is not byte-reproducible: a
    # refetch on 2026-11-13 returned 382 CDSs / 151,274 codons against the committed 381 / 147,948, agreeing
    # at Pearson r = 0.998 on relative usage with a worst-case codon difference of 0.035. The committed
    # `results/codon_table.json` is the authoritative one -- the reported CAI and %MinMax numbers were
    # computed against it -- so this writes alongside it rather than over it.
    corpus = fetch(base + " AND 600:3000[SLEN]", 400, lo=100)
    json.dump(codon_table(corpus), open(f"{outdir}/codon_table_refetched.json", "w"))

    # The panel is pinned by gene symbol. An unpinned esearch returns a different gene set on different
    # days, which would make the reported numbers irreproducible; the first six below are the ones the
    # 2026-11-07 GPU run completed before its scheduler limit, so those must stay fixed.
    pool = []
    for sym, acc in PANEL:
        got = _parse(_get("efetch.fcgi", db="nuccore", id=acc, rettype="gb", retmode="text"))
        if not got:
            print(f"  WARNING: {acc} ({sym}) returned no parseable CDS")
            continue
        pool.append(got[0])
        time.sleep(0.3)
    panel = sorted(pool, key=lambda h: h["n_codons"])
    json.dump(panel, open(f"{outdir}/bias_cds.json", "w"))
    print(f"corpus {len(corpus)} CDSs ({codon_table(corpus)['n_codons']:,} codons); "
          f"panel {len(panel)} genes, {panel[0]['n_codons']}-{panel[-1]['n_codons']} codons")
    print(", ".join(f"{h['gene']}({h['n_codons']})" for h in panel))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
