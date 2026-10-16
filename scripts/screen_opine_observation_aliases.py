"""Look for real observation/action aliases in OPINE's public replay archive.

No environment or learned model is executed. The archive stores post-action
frames, sometimes with intermediate animation frames; only the final frame of
each action is used. Action zero/reset and level-completion transitions are
excluded. Click coordinates remain part of the action identity.

An exact collision with different successors disproves deterministic prediction
from that input alone on the recorded data. It does not prove an attack, that
history resolves the process, or that a new memory-learning method is novel.
"""
from __future__ import annotations

import argparse
import ast
from bisect import bisect_left, bisect_right
from collections import defaultdict, Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/opine-world"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def decode_steps(data):
    current = bytearray(64 * 64)
    groups = {}
    for index, frame in enumerate(data["frames"]):
        if "g" in frame:
            grid = frame["g"]
            assert len(grid) == 64 and all(len(row) == 64 for row in grid)
            current[:] = bytes(value for row in grid for value in row)
        for x, y, value in frame.get("d", []):
            assert 0 <= x < 64 and 0 <= y < 64
            current[y * 64 + x] = value
        group = groups.setdefault(frame["s"], {})
        if "a" in frame:
            assert "action" not in group, "Multiple action labels within one step"
            group["action"] = frame["a"]
        group.update(frame=bytes(current), final_frame_index=index)
    assert sorted(groups) == list(range(len(groups)))
    completion = {data["frames"][index]["s"] for index in data["level_steps"]}
    for step, record in groups.items():
        record["level"] = bisect_left(sorted(completion), step)
    return groups, completion


def bounded_range(*args):
    value = range(*args)
    if len(value) > 4096:
        raise ValueError("Mask comprehension too large")
    return value


def static_mask(tree):
    """Evaluate only a restricted, side-effect-free return expression.

    Never import or execute the downloaded learned world-model program. Missing
    functions and unsupported declarations have distinct statuses.
    """
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "move_counter_mask"), None)
    if fn is None:
        return {"status": "absent", "cells": []}
    env = {"range": bounded_range}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(value, (int, float, tuple, list)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
    body = [node for node in fn.body if not (isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return {"status": "unsupported_body", "cells": []}
    expression = ast.Expression(body[0].value)
    allowed = (ast.Expression, ast.List, ast.Tuple, ast.ListComp, ast.comprehension,
               ast.Name, ast.Load, ast.Store, ast.Constant, ast.Call,
               ast.BinOp, ast.Add, ast.Sub, ast.UnaryOp, ast.USub)
    for node in ast.walk(expression):
        if not isinstance(node, allowed):
            return {"status": "unsupported_expression", "cells": []}
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                and node.func.id == "range" and not node.keywords):
            return {"status": "unsupported_call", "cells": []}
    try:
        cells = sorted({(int(r), int(c)) for r, c in eval(
            compile(ast.fix_missing_locations(expression), "<mask-expression>", "eval"),
            {"__builtins__": {}, **env})})
        assert all(0 <= r < 64 and 0 <= c < 64 for r, c in cells)
        if cells:
            rows, cols = {r for r, _ in cells}, {c for _, c in cells}
            rspan, cspan = max(rows)-min(rows)+1, max(cols)-min(cols)+1
            long_indices = cols if cspan >= rspan else rows
            assert min(rspan, cspan) <= 2
            assert len(cells) <= 2 * max(rspan, cspan)
            assert max(long_indices)-min(long_indices)+1 == len(long_indices)
        return {"status": "parsed", "cells": cells}
    except Exception as error:
        return {"status": "unresolved", "cells": [], "error": type(error).__name__}


def decode_models(data):
    lines, models = [], []
    for engine in data["engines"]:
        if "t" in engine:
            # Upstream patches use split("\n"); splitlines() loses trailing
            # empty lines and corrupts later program versions.
            lines = engine["t"].split("\n")
        else:
            for start, count, added in reversed(engine["o"]):
                lines[start:start+count] = added
        code = "\n".join(lines)
        tree = ast.parse(code)
        models.append({"sha256": hashlib.sha256(code.encode()).hexdigest(),
                       "mask": static_mask(tree)})
    return models


def scan_game(path):
    data = json.loads(gzip.decompress(path.read_bytes()))
    steps, completion = decode_steps(data)
    models = decode_models(data)
    events = sorted(data["synth"], key=lambda e: (e["f"], e["round"]))
    event_positions = [e["f"] for e in events]
    groups, usable = defaultdict(list), []
    for step in range(1, len(steps)):
        record, before = steps[step], steps[step-1]
        if step in completion or record.get("action") in (None, "RESET"):
            continue
        action = record["action"]
        assert action.startswith("ACTION") or action.startswith("click("), action
        event_index = bisect_right(event_positions, before["final_frame_index"]) - 1
        mask = None
        if event_index >= 0:
            mask = models[events[event_index]["e"]]["mask"]
        entry = {"step": step, "before": before["frame"], "after": record["frame"],
                 "action": action, "level": record["level"], "mask": mask}
        groups[(record["level"], before["frame"], action)].append(entry)
        usable.append(entry)
    witnesses, conflicting_groups, minimum_errors = [], 0, 0
    for entries in groups.values():
        outcomes = Counter(entry["after"] for entry in entries)
        minimum_errors += len(entries) - max(outcomes.values())
        if len(outcomes) < 2:
            continue
        conflicting_groups += 1
        first = entries[0]
        second = next(entry for entry in entries if entry["after"] != first["after"])
        differences = [(i//64, i % 64) for i, (a, b) in
                       enumerate(zip(first["after"], second["after"])) if a != b]
        known_masks = all(entry["mask"] is not None and
                          entry["mask"]["status"] in ("parsed", "absent")
                          for entry in (first, second))
        covered = None
        if known_masks:
            covered = all(set(differences) <= {tuple(cell) for cell in entry["mask"]["cells"]}
                          for entry in (first, second))
        witnesses.append({"steps": [first["step"], second["step"]],
                          "level": first["level"], "action": first["action"],
                          "before_sha256": hashlib.sha256(first["before"]).hexdigest(),
                          "different_successor_cells": differences,
                          "difference_covered_by_both_available_masks": covered,
                          "mask_statuses": [e["mask"]["status"] if e["mask"] else "no_model_yet"
                                            for e in (first, second)]})
    # Compare finite-history collision counts on one fixed cohort. This is an
    # empirical consistency check, not a trained predictor or held-out score.
    histories = {}
    cohort = [e for e in usable if e["step"] >= 9 and
              all(steps[t]["level"] == e["level"] and steps[t].get("action") != "RESET"
                  for t in range(e["step"]-8, e["step"]))]
    for history in (0, 1, 2, 4, 8):
        table = defaultdict(Counter)
        for entry in cohort:
            step = entry["step"]
            prefix = tuple((steps[t-1]["frame"], steps[t]["action"])
                           for t in range(step-history, step))
            key = (entry["level"], prefix, entry["before"], entry["action"])
            table[key][entry["after"]] += 1
        histories[str(history)] = {
            "samples": len(cohort), "distinct_inputs": len(table),
            "repeated_input_groups": sum(sum(c.values()) > 1 for c in table.values()),
            "conflicting_groups": sum(len(c) > 1 for c in table.values()),
            "minimum_empirical_errors": sum(sum(c.values())-max(c.values()) for c in table.values())}
    return {"game": data["game_id"], "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "steps": len(steps), "usable_transitions": len(usable), "model_versions": len(models),
            "mask_status_counts": dict(Counter(m["mask"]["status"] for m in models)),
            "repeated_input_groups": sum(len(entries) > 1 for entries in groups.values()),
            "conflicting_groups": conflicting_groups, "minimum_empirical_errors": minimum_errors,
            "witnesses": witnesses, "history_consistency": histories}


def main(args):
    started = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sources = sorted((VENDOR / "docs/replay_data").glob("*.json.gz"))
    save(out / "plan.json", {
        "scope": "Read-only empirical consistency screen of published author trajectories; not a reproduction or an attack experiment",
        "source_repo": "https://github.com/david-courtis/opine-world",
        "upstream_commit": subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
        "games": [path.name for path in sources], "fixed_history_lengths": [0, 1, 2, 4, 8],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "api_requests": 0, "gpu_work": False})
    results = []
    for path in sources:
        result = scan_game(path)
        save(out / f"{result['game']}.json", result)
        results.append(result)
    summary = {
        "status": "completed", "elapsed_s": time.monotonic()-started,
        "games": len(results), "model_versions": sum(r["model_versions"] for r in results),
        "usable_transitions": sum(r["usable_transitions"] for r in results),
        "games_with_exact_observation_action_conflicts": sum(r["conflicting_groups"] > 0 for r in results),
        "conflicting_groups": sum(r["conflicting_groups"] for r in results),
        "minimum_empirical_stateless_errors": sum(r["minimum_empirical_errors"] for r in results),
        "witnesses_covered_by_both_masks": sum(w["difference_covered_by_both_available_masks"] is True
                                              for r in results for w in r["witnesses"]),
        "witnesses_not_covered_by_both_masks": sum(w["difference_covered_by_both_available_masks"] is False
                                                  for r in results for w in r["witnesses"]),
        "witnesses_with_unavailable_mask": sum(w["difference_covered_by_both_available_masks"] is None
                                              for r in results for w in r["witnesses"]),
        "results": [{key: value for key, value in r.items() if key != "witnesses"} for r in results],
        "api_requests": 0, "gpu_work": False,
        "limits": "Exact contradictions concern the observed input representation, not latent physics. History can split inputs into singletons, so fewer collisions is not a generalization result. Mask coverage classifies one witness pair per conflicting group, not all pairs. No fabricated model error, mask, or transition is introduced."}
    save(out / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
