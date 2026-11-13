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

#: The 16-gene panel, pinned by (gene symbol, CDS length in codons). Symbol alone is not enough: several of
#: these genes have transcript isoforms of different lengths and an unpinned search picks a different one on
#: a different day. The lengths below are those of the transcripts actually used on 2026-11-07; the first six
#: are the genes that run completed (results/mrna_correction_size.csv). The accessions themselves were lost
#: when the workspace was cleared before they were committed, so the length is the tightest available pin --
#: once this script has run, `results/bias_cds.json` records the accessions and should be preferred.
PANEL = (("CCL4L1", 93), ("PDE6D", 94), ("NDUFA2", 100), ("CRYAB", 109), ("IFT43", 114), ("B2M", 120),
         ("KISS1", 139), ("HBB", 148), ("BCL11A", 158), ("NMNAT1", 161), ("NDUFAF2", 170), ("FTL", 176),
         ("KRAS", 189), ("CRYBA4", 197), ("CSNK2B", 213), ("TSFM", 216))


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
    for sym, n_cod in PANEL:
        got = fetch(f'{base} AND {sym}[Gene Name]', 40, lo=60, hi=400)
        exact = [h for h in got if h["gene"].upper() == sym.upper() and h["n_codons"] == n_cod]
        if not exact:
            avail = sorted({h["n_codons"] for h in got if h["gene"].upper() == sym.upper()})
            print(f"  WARNING: {sym} has no {n_cod}-codon transcript; available: {avail}")
            continue
        pool.append(sorted(exact, key=lambda h: h["acc"])[0])
        time.sleep(0.3)
    panel = sorted(pool, key=lambda h: h["n_codons"])
    json.dump(panel, open(f"{outdir}/bias_cds.json", "w"))
    print(f"corpus {len(corpus)} CDSs ({codon_table(corpus)['n_codons']:,} codons); "
          f"panel {len(panel)} genes, {panel[0]['n_codons']}-{panel[-1]['n_codons']} codons")
    print(", ".join(f"{h['gene']}({h['n_codons']})" for h in panel))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
