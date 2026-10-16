"""Tiny CPU probe of TD-MPC2 task-embedding query side effects.

Runs the actual upstream task_emb method on synthetic embedding weights. It
does not instantiate a world model, load a checkpoint, train, or run a task.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/tdmpc2/tdmpc2/common/world_model.py"


def native_task_emb():
    text = SOURCE.read_text()
    tree = ast.parse(text)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "WorldModel")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "task_emb")
    init = next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignment = next(node for node in ast.walk(init)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Attribute)
                              and target.attr == "_task_emb" for target in node.targets))
    constructor = assignment.value
    assert isinstance(constructor, ast.Call)
    assert isinstance(constructor.func, ast.Attribute)
    assert constructor.func.attr == "Embedding"
    max_norm = next(kw.value for kw in constructor.keywords if kw.arg == "max_norm")
    assert ast.literal_eval(max_norm) == 1
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["task_emb"], {
        "path": SOURCE.relative_to(ROOT).as_posix(),
        "constructor_line": assignment.lineno,
        "constructor": ast.get_source_segment(text, assignment),
        "method_line": method.lineno,
        "method": ast.get_source_segment(text, method),
    }


def one_case(task_emb, dtype, victim_norm):
    # Dimensions match the already-recorded official mt30-48M configuration.
    # These are deliberately synthetic weights, not sampled checkpoint rows.
    weights = torch.zeros((30, 64), dtype=dtype, device="cpu")
    weights[0, 0] = 0.25
    weights[1, :2] = torch.tensor([3.0, 4.0], dtype=dtype)
    weights[2, :2] = torch.tensor([0.6, 0.8], dtype=dtype) * victim_norm
    weights[3, :2] = torch.tensor([5.0, 12.0], dtype=dtype)
    x = torch.arange(15, dtype=dtype, device="cpu").reshape(3, 5)
    conditions = {
        "victim_only": [],
        "other_tasks_first": [1, 3, 0, 4],
        "same_task_first": [2],
        "mixed_queries_first": [1, 2, 3, 2, 0, 4],
    }
    records, outputs = {}, {}
    for name, prefix in conditions.items():
        holder = torch.nn.Module()
        # from_pretrained avoids random initialization and still invokes the
        # real nn.Embedding forward, including its in-place max_norm behavior.
        holder._task_emb = torch.nn.Embedding.from_pretrained(
            weights.clone(), freeze=True, max_norm=1.0
        )
        changes = []
        for task in prefix + [2]:
            before = holder._task_emb.weight.detach().clone()
            output = task_emb(holder, x, torch.tensor([task], device="cpu"))
            after = holder._task_emb.weight.detach()
            rows = torch.nonzero((before != after).any(dim=1)).flatten().tolist()
            changes.append({"queried_task": task, "changed_rows": rows,
                            "only_queried_row_changed": all(row == task for row in rows)})
        outputs[name] = output.detach().clone()
        records[name] = {
            "query_changes": changes,
            "victim_norm_after": float(holder._task_emb.weight[2].norm()),
            "original_input_prefix_preserved": torch.equal(output[:, :x.shape[1]], x),
        }
    reference = outputs["victim_only"]
    for name, output in outputs.items():
        records[name]["victim_output_bitwise_equal_to_victim_only"] = torch.equal(output, reference)
        records[name]["victim_output_max_abs_difference"] = float((output-reference).abs().max())
    checks = [record["original_input_prefix_preserved"]
              and record["victim_output_bitwise_equal_to_victim_only"]
              and all(query["only_queried_row_changed"] for query in record["query_changes"])
              for record in records.values()]
    # Positive control: two over-norm other-task rows really are modified.
    source_changes = records["other_tasks_first"]["query_changes"]
    positive_control = source_changes[0]["changed_rows"] == [1] and source_changes[1]["changed_rows"] == [3]
    return {
        "dtype": str(dtype),
        "requested_victim_norm": victim_norm,
        "victim_norm_before": float(weights[2].norm()),
        "over_norm_other_task_rows_really_changed": positive_control,
        "conditions": records,
        "passed": all(checks) and positive_control,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/tdmpc_embedding_query_isolation_20261015.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(1)
    task_emb, source = native_task_emb()
    with torch.no_grad():
        cases = [one_case(task_emb, dtype, norm)
                 for dtype in (torch.float32, torch.float64)
                 for norm in (0.5, 1.0 + 2.0**-20, 5.0, 10000.0)]
    passed = all(case["passed"] for case in cases)
    result = {
        "schema_version": 1,
        "status": "completed_cpu_embedding_mechanism_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "EXCLUDE_THIS_QUERY_NORMALIZATION_CANDIDATE" if passed else "CANDIDATE_NOT_EXCLUDED",
        "all_checks_passed": passed,
        "source_evidence": source,
        "supporting_source_paths": [
            "vendor/tdmpc2/tdmpc2/common/parser.py",
            "vendor/tdmpc2/tdmpc2/tdmpc2.py",
            "runs/tdmpc_cascade_heldout_matched_v1/provenance.json",
            "scripts/probe_tdmpc_embedding_query_isolation.py",
        ],
        "official_semantics_reference": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html",
        "runtime": {"torch": torch.__version__, "device": "cpu", "threads": 1,
                    "primary_dtype": "torch.float32", "supplemental_dtype": "torch.float64"},
        "protocol": {
            "embedding_shape": [30, 64], "victim_task": 2,
            "weights": "Fixed synthetic rows with norm below, just above, and far above max_norm=1; no checkpoint tensors.",
            "entrypoint": "Actual upstream WorldModel.task_emb method compiled from its AST, bound to a minimal holder with a real nn.Embedding.",
            "invariance": "Same initial weight bank and fixed victim input, with no interleaved training; vary only prior task-embedding lookups.",
            "comparison": "Exact torch.equal of the complete concatenated victim output; record every row changed by every lookup.",
            "case_count": len(cases), "conditions_per_case": 4,
        },
        "cases": cases,
        "interpretation": (
            "In these CPU cases, prior lookups change only their indexed over-norm rows, and the victim's first lookup normalizes its own row before returning it. Prior other-task, same-task, and mixed lookups give exactly the same victim output. The max_norm write alone is therefore not evidence for cross-task decision contamination."
            if passed else "One or more CPU checks failed; inspect the retained cases before rejecting the candidate."
        ),
        "limits": [
            "This is a synthetic operator-level exclusion probe, not an evaluation of the pretrained checkpoint or native planner.",
            "No claim about all floating-point values, concurrent training, GPU execution, warm-start/RNG isolation, action ranking, or real task utility.",
            "Existing conditional TD-MPC2 task losses remain unchanged and do not gain a new causal explanation from this probe.",
        ],
        "resource_use": {"checkpoints_loaded": 0, "world_models_instantiated": 0,
                         "environment_steps": 0, "training_updates": 0,
                         "gpu_work": False, "api_requests": 0, "file_hashes_computed": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "all_checks_passed": passed, "cases": len(cases),
                      "output": str(args.output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
