"""Check the exact pairwise-secant/centered-regression identity on native data.

For n rows, sum_{i<j} ||(z_j-z_i)-(x_j-x_i) B||^2 equals
n * sum_i ||(z_i-z_bar)-(x_i-x_bar) B||^2. Pairwise ridge lambda
therefore equals centered-regression ridge lambda/n. Event routing does not
alter this identity within any fixed cluster. No broad impossibility follows.
"""
import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path

import numpy as np


def main(args):
    source = Path(args.source)
    rows = [json.loads(line) for line in (source / "native_traces.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        if not row["invalid"]:
            grouped[(row["level"], row["seed"], row["arm"], row["first_action_contact"])].append(row)
    records = []
    regularization = .01
    for (level, seed, arm, event), cluster in grouped.items():
        if len(cluster) < 3:
            continue
        x = np.array([r["normalized"] for r in cluster])
        z = np.array([r["closest_error"] for r in cluster])
        n = len(x)
        i, j = np.triu_indices(n, k=1)
        dx, dz = x[j] - x[i], z[j] - z[i]
        xc, zc = x - x.mean(0), z - z.mean(0)
        pair = np.linalg.solve(dx.T @ dx + regularization * np.eye(3), dx.T @ dz)
        centered = np.linalg.solve(xc.T @ xc + regularization / n * np.eye(3), xc.T @ zc)
        # Exactly matched downstream inverse-step construction, not a new query.
        anchor_x, anchor_z = x[-1], z[-1]
        proposals = []
        for fitted in (pair, centered):
            step = np.linalg.solve(fitted @ fitted.T + .1 * np.eye(3), -fitted @ anchor_z)
            proposals.append(np.clip(anchor_x + step, 0, 1))
        records.append(dict(level=level, seed=seed, arm=arm, event=event, rows=n,
                            pairs=len(i), max_coefficient_error=float(np.abs(pair - centered).max()),
                            max_proposal_error=float(np.abs(proposals[0] - proposals[1]).max())))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="completed", source=str(source), clusters=len(records),
                  max_coefficient_error=max(r["max_coefficient_error"] for r in records),
                  max_proposal_error=max(r["max_proposal_error"] for r in records),
                  pairwise_regularization=regularization,
                  centered_regularization="pairwise_regularization / cluster_size",
                  native_queries=0, api_requests=0, gpu_work=False,
                  scope="Exact equivalence for complete unweighted pairwise linear secants with matched event groups, features and regularization. Does not cover arbitrary pair weights, learned nonlinear edit grammars, or establish a general impossibility of transferable experimentation.",
                  finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out / "clusters.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/interphyre_continuous_search_v0")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
