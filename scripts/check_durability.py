"""
What in this working tree is actually durable, and what only exists here?

The workspace is cleared when a session idles, so a file here is not a saved file. Two stores outlive it:
git (for code, once PUSHED -- a local commit is not a save either) and the artifact store (for results).
This reports which tracked files are in neither.

Run it before ending a session, and -- the lesson of 2026-11-13 -- before claiming anything is LOST. On that
date two result files were declared missing and retrieved over SSH from a cluster scratch directory when both
had been in the artifact store for six days, byte-identical. The store is the authority on what exists; a
recollection of what was saved is not. Checking costs one query.

    python scripts/check_durability.py [--dirs results docs]

Exits non-zero if anything is undurable, so it can gate a close-out.
"""
import argparse
import hashlib
import pathlib
import subprocess
import sys

DEFAULT_DIRS = ("results", "docs")


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    r = subprocess.run(("git",) + args, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def git_state():
    """Which files are committed, and is the local branch ahead of its remote?

    'Committed' is not enough on its own: the 2026-11-07 wipe destroyed two commits that had never been
    pushed, so an unpushed commit is reported as a gap of its own.
    """
    tracked = set((_git("ls-files") or "").splitlines())
    dirty = {l[3:].strip('"') for l in (_git("status", "--porcelain") or "").splitlines()}
    ahead = _git("rev-list", "--count", "@{upstream}..HEAD")
    return tracked, dirty, (int(ahead) if ahead and ahead.isdigit() else None)


def artifact_checksums():
    """Every checksum in the project's artifact store, keyed by filename -> {checksums}.

    Keyed on content, not on version: a file matching ANY stored version is durable, even an older one, and
    the point of the check is to answer "do these bytes exist outside this directory" rather than "is this
    the newest version".
    """
    try:
        from host import query                                     # the repl kernel's accessor
    except ImportError:
        return None
    rows = query("""
        SELECT a.filename, v.checksum FROM artifact_versions v
        JOIN artifacts a ON a.id = v.artifact_id
    """)["rows"]
    out = {}
    for fn, ck in rows:
        out.setdefault(fn, set()).add(ck)
    return out


def main(dirs=DEFAULT_DIRS, store=None):
    tracked, dirty, ahead = git_state()
    store = artifact_checksums() if store is None else store
    files = sorted(p for d in dirs for p in pathlib.Path(d).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)

    gaps = []
    for p in files:
        s = str(p)
        in_git = s in tracked and s not in dirty
        in_store = store is not None and _sha256(p) in store.get(p.name, ())
        if not (in_git or in_store):
            gaps.append((s, "uncommitted" if s in tracked else "untracked"))

    print(f"{len(files)} files under {', '.join(dirs)}")
    if store is None:
        print("  NOTE: artifact store not reachable from this kernel -- run from the repl tool for the "
              "store half of the check; git coverage below is still valid.")
    if ahead:
        print(f"  {ahead} commit(s) committed locally but NOT pushed -- these die with the workspace.")
    for s, why in gaps:
        print(f"  {why:<12} {s}")
    if not gaps and not ahead:
        print("  all durable (committed and pushed, or in the artifact store).")
    return 1 if (gaps or ahead) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=list(DEFAULT_DIRS))
    sys.exit(main(ap.parse_args().dirs))
