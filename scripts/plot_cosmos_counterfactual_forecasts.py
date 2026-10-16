#!/usr/bin/env python3
"""Plot committed RGB forecasts beside an independently replayed real target."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cosmos_policy_native_inference as native
from evaluate_cosmos_counterfactual_ranking import verify_commit, _validate_utilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--candidate", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    commit_path = Path(result["prediction_commit"])
    commit, prediction = verify_commit(commit_path)
    utility = _validate_utilities(commit_path, commit, prediction, result["utilities"])
    item = utility["candidates"][str(args.candidate)]
    future = item["replays"][0]["candidate_future"]
    with np.load(future["observations_npz"], allow_pickle=False) as archive:
        actual = native.preprocess_observation({key: archive[key].copy() for key in archive.files})
    views = ("wrist", "primary")
    titles = ("Current / persistence", "Actual at t+16", "Clean forecast", "Goal-Swap forecast", "Zero-text diagnostic")
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8), constrained_layout=True)
    for i, view in enumerate(views):
        images = [np.load(commit["persistence_predictions"][view]["path"], allow_pickle=False)[0], actual[i]]
        for condition in ("clean", "goal_swap", "no_lang_zero_embedding"):
            relative = prediction["conditions"][condition]["candidates"][args.candidate]["decoded_future_images"][view]["path"]
            images.append(np.load(Path(commit["ranking_json"]).parent / relative, allow_pickle=False)[0])
        for j, value in enumerate(images):
            axes[i, j].imshow(value)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)
        axes[i, 0].set_ylabel(view.capitalize())
    conditions = {row["condition"]: row for row in result["results"]}
    selections = ", ".join(f"{name}: {conditions[key]['selected_index']}" for name, key in
                            (("Clean", "clean"), ("Swap", "goal_swap"), ("Zero", "no_lang_zero_embedding")))
    fig.suptitle(f"Cosmos Policy | prefix {result['prefix_steps']} | fixed candidate {args.candidate}\n"
                 f"Selected candidate: {selections}. Actual candidate success: {item['success']}.", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
