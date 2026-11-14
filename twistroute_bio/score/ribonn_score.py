"""Score transcripts with RiboNN (translation efficiency), as a callable rather than their CLI.

Their `make predict` path hardcodes num_workers=4, which fails in a sandbox: the DataLoader's shared-memory
manager cannot start ("no response from torch_shm_manager"). num_workers=0 is the fix and costs nothing at
our scale. Returns one row per transcript with the per-cell-type predictions averaged, matching what their
main.py computes.
"""
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.predict import predict_using_nested_cross_validation_models


def score(input_path, models_csv="models/runs.csv", top_k=5, batch_size=32):
    run_df = pd.read_csv(models_csv)
    pred = predict_using_nested_cross_validation_models(
        input_path, run_df, top_k, batch_size=batch_size, num_workers=0)
    cols = [c for c in pred.columns if c.startswith("predicted_")]
    out = pred.groupby(["tx_id", "utr5_sequence", "cds_sequence", "utr3_sequence"],
                       as_index=False)[cols].agg("mean")
    out["mean_predicted_TE"] = out[cols].mean(axis=1)
    return out, cols


if __name__ == "__main__":
    out, cols = score(sys.argv[1] if len(sys.argv) > 1 else "data/prediction_input1.txt")
    print(f"{len(out)} transcripts scored over {len(cols)} cell types")
    print(out[["tx_id", "mean_predicted_TE"]].to_string(index=False))
    out.drop(columns=["utr5_sequence", "cds_sequence", "utr3_sequence"]).to_csv(
        "results/prediction_output.txt", sep="\t", index=False)
