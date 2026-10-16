#!/usr/bin/env python3
"""Check cached paper quantities, citations, build freshness and source hashes.

This verifier does not establish novelty, a causal mediation claim, population
generalization, or the validity of an unexecuted prospective experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports/decision_leverage_paper"


def checked(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"Verification failed: {command}\n{result.stdout}\n{result.stderr}")
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Write a new current verification receipt")
    args = parser.parse_args()
    commands = [
        [sys.executable, "scripts/compute_decision_leverage_panel.py", "--verify-only"],
        [str(ROOT / ".venv-jepa-native/bin/python"), "scripts/dino_cached_decomposition.py", "--verify-only"],
        [sys.executable, "scripts/analyze_decision_objective_reweighting.py", "--verify-only"],
        [sys.executable, "scripts/analyze_cosmos_first_query_truth.py", "--verify-only"],
    ]
    receipts = [checked(command) for command in commands]
    tex_paths = [PAPER / "main.tex", PAPER / "theory_related_work.tex"]
    tex = "\n".join(path.read_text() for path in tex_paths)
    references = (PAPER / "references.bib").read_text()
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", references))
    assert citations <= bibkeys, f"Missing bibliography entries: {citations-bibkeys}"
    for name in re.findall(r"\\input\{([^}]+)\}", tex):
        assert (PAPER / name).is_file(), f"Missing TeX input: {name}"
    pdf = PAPER / "main.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 20000
    inputs = [*tex_paths, PAPER / "references.bib", PAPER / "panel_table.tex", PAPER / "decision_diagnostics.pdf"]
    assert pdf.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in inputs), "PDF older than its inputs"
    log = (PAPER / "main.log").read_text()
    assert "undefined references" not in log.lower()
    assert not re.search(r"Citation .+ undefined", log)
    assert "Overfull \\hbox" not in log, "Visible layout overflow requires inspection"
    extracted = checked(["pdftotext", "-layout", str(pdf), "-"])["stdout"]
    # Numeric tripwires bind the displayed result table to recomputed sources.
    # They are not a substitute for reviewing prose, assumptions or scope.
    dino = json.loads((ROOT / "reports/dino_first_query_terminal_truth_decision_20261015.json").read_text())
    expected_numbers = []
    for row in dino["rows"]:
        for key in ["original_forecast_weighted_mse", "canonical_forecast_weighted_mse",
                    "true_future_native_cost", "cumulative_native_reward", "terminal_native_reward"]:
            expected_numbers.append(f"{row[key]:.6f}")
    for value in expected_numbers:
        assert value in extracted, f"Measured table number absent from PDF: {value}"
    cosmos = json.loads((ROOT / "reports/cosmos_first_choice_identifiability_bounds_20261015.json").read_text())
    for value in cosmos["any_first_choice_selector"]["capped_step_total_or_expectation_interval"]:
        assert str(value) in extracted
    assert str(cosmos["relative_to_recorded_native"]["native_capped_steps_total"]) in extracted
    hashes = []
    paths = [p for p in PAPER.iterdir() if p.suffix in {".tex", ".bib", ".pdf", ".csv", ".json", ".txt"}
             and p.name not in {"verification.json", "paper_progress.json"}]
    paths += [ROOT / "scripts" / name for name in [
        "compute_decision_leverage_panel.py", "dino_cached_decomposition.py",
        "analyze_decision_objective_reweighting.py", "plot_decision_leverage_paper.py",
        "verify_decision_leverage_paper.py"]]
    for path in sorted(paths):
        data = path.read_bytes()
        hashes.append({"path": str(path.relative_to(ROOT)), "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()})
    report = {
        "status": "passed_current_cached_paper_checks",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": receipts,
        "cited_bibliography_entries": len(citations),
        "pdf_numeric_tripwires": len(expected_numbers) + 3,
        "pdf_pages": len([page for page in extracted.split("\f") if page.strip()]),
        "pdf_source_freshness": True,
        "unresolved_citations": [], "layout_overflow": False,
        "file_hashes": hashes,
        "scope": "Existing cache recomputation, checked numeric strings in compiled PDF, citation closure and artifact hashes. Not a scientific completion, prospective-experiment validation, or novelty guarantee.",
        "activity": {"model_calls": 0, "environment_steps": 0, "api_calls": 0},
    }
    if args.record:
        (PAPER / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ["status", "cited_bibliography_entries", "pdf_numeric_tripwires", "pdf_pages"]}))


if __name__ == "__main__":
    main()
