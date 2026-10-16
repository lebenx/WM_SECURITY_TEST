"""Reference-only search over measurement programs, followed by native tasks.

This finite acquisition grammar is an experimental candidate, not a novelty
claim, a reproduction of Causal Curiosity/MDA, or unrestricted program induction.
It was designed after prior Reactor Lab development, including source review.
No isolation predicate, correct instrument, reducer, formula, target label, or
seed is passed to the search. Locations come from public observations.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from pilot_reactor_measurement_protocol import ProtocolCollector
from pilot_reactor_program_grammar import induce, design
from pilot_reactor_repair_consensus import INSTRUMENTS, VENDOR, parse_measurement, save
from pilot_reactor_simple_program import tune


REDUCERS = {"min": np.min, "mean": np.mean, "median": np.median, "max": np.max}
COUNTS = (1, 8, 32)


class SearchCollector(ProtocolCollector):
    action_budget = 2800

    def readings(self, row, instrument, site, count):
        uid = row["uuid"]
        meter = next(k for k,v in self.known.items() if v["name"] == instrument)
        self.goto(uid)
        parent = None
        if site["kind"] != "native":
            parent = int(re.search(r"\[uuid: (\d+)\]", self.known[uid]["description"])[1])
            self.act(action="PICKUP", arg1=uid)
            if site["kind"] == "landmark":
                self.act(action="TELEPORT_TO_LOCATION", arg1=site["name"])
            else:
                self.goto(site["uuid"])
        values = []
        for _ in range(count):
            obs = self.act(action="USE", arg1=meter, arg2=uid)
            values.append(parse_measurement(instrument, obs["lastActionMessage"]))
        if parent is not None:
            self.goto(parent)
            self.act(action="PUT", arg1=uid, arg2=parent)
        return values


def candidate_order(candidate):
    p, m = candidate["program"], candidate["model"]
    # Intentionally lexicographic; no claim that this is a calibrated code for
    # joint physical/program complexity. Law simplicity comes before run cost.
    return (m["description_characters"], p["estimated_actions_per_target"],
            m["reference_max_error_hz"], json.dumps(p,sort_keys=True), m["sensor"],m["family"])


def unchanged_candidate(refs):
    m = induce(refs)["selected"]
    assert m is not None
    return {"program":{"site":{"kind":"native"},"count":1,"reducer":"first",
                       "instrument":None,"estimated_actions_per_target":0},"model":m}


def search(collector, refs, ground_sites, validation_reads=0, validation_tolerance=1.0):
    assert all(r["observed_calibration_hz"] is not None for r in refs)
    started, start_actions = time.monotonic(), collector.actions
    # Candidate locations are fixed before measuring reference variability.
    sites = [{"kind":"native"}]
    sites += [{"kind":"landmark","name":name} for name in sorted(collector.api.listTeleportLocationsDict())]
    sites += [{"kind":"object","uuid":uid,"public_name":"grass"} for uid in
              sorted(uid for uid,obj in collector.known.items() if obj["name"] == "grass")[:ground_sites]]
    cache, varying = {}, {}
    # Test all public instruments, not a named noisy channel. Four repeats are
    # a fixed screening rule; it may miss noise or systematic nonrandom bias.
    for instrument in INSTRUMENTS:
        cache[(instrument,0)] = [collector.readings(row,instrument,sites[0],4) for row in refs]
        keys = sorted(cache[(instrument,0)][0][0])
        channels = [key for key in keys if any(
            np.ptp([v[key] for v in values]) > 1e-8 for values in cache[(instrument,0)])]
        if channels:
            varying[instrument] = channels
    save(collector.output/"search_candidates.json",{"sites":sites,"varying_channels":varying,
         "counts":COUNTS,"reducers":list(REDUCERS),"reference_ids":[r["uuid"] for r in refs]})
    candidates = [unchanged_candidate(refs)]
    for instrument,channels in varying.items():
        for si,site in enumerate(sites):
            samples = []
            for ri,row in enumerate(refs):
                previous = cache[(instrument,0)][ri] if si == 0 else []
                extra = collector.readings(row,instrument,site,max(COUNTS)+validation_reads-len(previous))
                samples.append(previous+extra)
            with (collector.output/"reference_samples.jsonl").open("a") as handle:
                handle.write(json.dumps({"instrument":instrument,"site":site,"samples":samples})+"\n")
            for count in COUNTS:
                reducers = {"first":lambda x:x[0]} if count == 1 else REDUCERS
                for reducer,aggregate in reducers.items():
                    rows = deepcopy(refs)
                    for row,values in zip(rows,samples):
                        row["readings"] = {key:float(aggregate([v[key] for v in values[:count]])) for key in channels}
                    model = induce(rows)["selected"]
                    if model is None:
                        continue
                    program = {"instrument":instrument,"site":site,"count":count,"reducer":reducer,
                               "estimated_actions_per_target":count+1+(4 if site["kind"]!="native" else 0)}
                    errors=[]
                    for row,values in zip(refs,samples):
                        fresh=values[max(COUNTS):]
                        for offset in range(0,len(fresh)-count+1,count):
                            x=float(aggregate([v[model["sensor"]] for v in fresh[offset:offset+count]]))
                            predicted=float((design([x],model["terms"]) @ np.array(model["coefficients"]))[0])
                            errors.append(abs(predicted-row["observed_calibration_hz"]))
                    candidates.append({"program":program,"model":model,
                                       "validation_max_error_hz":max(errors) if errors else None,
                                       "validation_predictions":len(errors),
                                       "validation_passed":bool(errors) and max(errors)<=validation_tolerance})
    candidates.sort(key=candidate_order)
    unvalidated=candidates[0]
    # The default static program remains a fallback; it is not claimed to have
    # passed an independent test. New acquisition programs must predict fresh
    # reference measurements before being applied to targets.
    eligible=[c for c in candidates if not validation_reads or
              c["program"]["instrument"] is None or c.get("validation_passed")]
    selected=eligible[0]
    save(collector.output/"selection.json",{"selected":selected,"ranked_candidates":candidates,
         "unvalidated_selection_same_training_data":unvalidated,
         "validation_reads_per_reference":validation_reads,"validation_tolerance_hz":validation_tolerance,
         "reference_only":True,"target_outcomes_seen":False,"target_measurements_used":False,
         "search_actions":collector.actions-start_actions,"elapsed_s":time.monotonic()-started})
    return selected


def predict_targets(collector, targets, selected):
    p,m=selected["program"],selected["model"]
    rows=deepcopy(targets)
    for row in rows:
        if p["instrument"] is not None:
            values=collector.readings(row,p["instrument"],p["site"],p["count"])
            xs=[v[m["sensor"]] for v in values]
            row["readings"][m["sensor"]]=float(xs[0] if p["reducer"]=="first" else REDUCERS[p["reducer"]](xs))
    x=[r["readings"][m["sensor"]] for r in rows]
    predictions=(design(x,m["terms"]) @ np.array(m["coefficients"])).tolist()
    save(collector.output/"target_predictions.json",{"observations":rows,"predictions_hz":predictions,
         "model_refit_on_targets":False})
    return predictions


def fixed_calibration(collector, rows):
    # This comparator explicitly retains the previously supplied source-informed
    # isolation/minimum procedure. It is stronger but more privileged.
    calibrated=collector.repeat_measurements(rows,"isolate_min",32)
    refs=[r for r in calibrated if r["observed_calibration_hz"] is not None]
    target=[r for r in calibrated if r["observed_calibration_hz"] is None]
    selected=unchanged_candidate(refs)
    predictions=(design([r["readings"][selected["model"]["sensor"]] for r in target],
                        selected["model"]["terms"]) @ np.array(selected["model"]["coefficients"])).tolist()
    return selected,predictions


def main(args):
    started,out=time.monotonic(),Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cases=[(difficulty,seed) for difficulty in args.difficulties for seed in args.seeds]
    save(out/"plan.json",{"cases":cases,"arms":args.arms,"counts":COUNTS,"reducers":list(REDUCERS),
         "ground_sites":args.ground_sites,"action_budget":SearchCollector.action_budget,"scope":__doc__,
         "validation_reads":args.validation_reads,"validation_tolerance_hz":args.validation_tolerance,
         "variable_instrument_screen":"four repeats per reference for each of five instruments; any channel with nonzero observed range",
         "selection":"reference-only law-description length, then predicted execution actions, then reference fit error and deterministic lexical tie-break",
         "fixed_calibration_privilege":"researcher supplied isolation/minimum protocol from prior source inspection",
         "target_only_baseline":"same grammar, reference-only fit; target observations do not enter selection",
         "fresh_seeds_scope":"New development seeds within the same scenario generator, not out-of-domain transfer",
         "api_requests":0,"gpu_work":False,
         "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    os.environ.setdefault("SDL_AUDIODRIVER","dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
    sys.path.insert(0,str(VENDOR))
    results=[]
    try:
        with (out/"native_stdout.log").open("w",buffering=1) as log,redirect_stdout(log):
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
            for difficulty,seed in cases:
                for arm in args.arms:
                    folder=out/f"{difficulty.lower()}_seed_{seed}_{arm}"
                    folder.mkdir()
                    api=DiscoveryWorldAPI(threadID=9700+len(results))
                    api.FRAME_DIR=str(folder/"frames")
                    assert api.loadScenario("Reactor Lab",difficulty,randomSeed=seed,numUserAgents=1)
                    collector=SearchCollector(api,folder)
                    rows=collector.collect()
                    collection_actions=collector.actions
                    refs=[r for r in rows if r["observed_calibration_hz"] is not None]
                    targets=[r for r in rows if r["observed_calibration_hz"] is None]
                    if arm=="search":
                        selected=search(collector,refs,args.ground_sites,args.validation_reads,args.validation_tolerance)
                        predictions=predict_targets(collector,targets,selected)
                    elif arm=="target_only":
                        selected=unchanged_candidate(refs)
                        predictions=predict_targets(collector,targets,selected)
                    else:
                        selected,predictions=fixed_calibration(collector,rows)
                    save(folder/"frozen_selection.json",selected)
                    valid=all(np.isfinite(p) and 0<=p<=10000 for p in predictions)
                    tuning=[tune(collector,row,p) for row,p in zip(targets,predictions)] if valid else []
                    progress=collector.observation["taskProgress"]
                    result={"arm":arm,"difficulty":difficulty,"seed":seed,"selected":selected,
                            "predictions_hz":predictions,"valid_predictions":bool(valid),
                            "collection_actions":collection_actions,"total_actions":collector.actions,
                            "tuning":tuning,"public_task_progress":progress,
                            "completed_successfully":bool(progress) and all(t["completedSuccessfully"] for t in progress)}
                    save(folder/"result.json",result);results.append(result)
                    print(json.dumps({k:result[k] for k in ["arm","difficulty","seed","completed_successfully","total_actions"]}),file=sys.__stdout__,flush=True)
                    del collector,api
        summary={"status":"completed","episodes":len(results),"unique_worlds":len(cases),"results":results,
                 "elapsed_s":time.monotonic()-started,"api_requests":0,"gpu_work":False,
                 "by_arm":{arm:{"successes":sum(r["completed_successfully"] for r in results if r["arm"]==arm),
                                     "episodes":sum(r["arm"]==arm for r in results),
                                     "actions":sum(r["total_actions"] for r in results if r["arm"]==arm)} for arm in args.arms},
                 "scope":__doc__}
        save(out/"summary.json",summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ["results","scope"]}),flush=True)
    except Exception as error:
        save(out/"status.json",{"status":"failed","error":repr(error),"completed_episodes":len(results)})
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=[8,9])
    parser.add_argument("--difficulties",nargs="+",choices=["Normal","Challenge"],default=["Normal","Challenge"])
    parser.add_argument("--arms",nargs="+",choices=["target_only","fixed_calibration","search"],default=["target_only","fixed_calibration","search"])
    parser.add_argument("--ground-sites",type=int,default=6)
    parser.add_argument("--validation-reads",type=int,choices=[0,32],default=0)
    parser.add_argument("--validation-tolerance",type=float,default=1.0)
    main(parser.parse_args())
