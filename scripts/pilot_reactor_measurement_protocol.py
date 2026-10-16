"""Compare ordinary repeat averaging with a supplied isolation/control protocol.

This is a baseline follow-up on two development failures, not a learned protocol
or new method. The isolation/minimum protocol is researcher-designed after
reading the radiation instrument implementation (nonnegative ambient noise and
nearby-object contributions). Its design provenance must not be hidden.
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

from pilot_reactor_program_grammar import Collector, induce
from pilot_reactor_repair_consensus import ROOT, VENDOR, parse_measurement, save
from pilot_reactor_simple_program import tune


class ProtocolCollector(Collector):
    action_budget = 700

    def act(self, **action):
        assert self.actions < self.action_budget, "Protocol action budget exceeded"
        result = self.api.performAgentAction(0, action)
        self.api.tick()
        obs = self.observe()
        self.actions += 1
        with (self.output / "public_actions.jsonl").open("a") as handle:
            handle.write(json.dumps({"index": self.actions, "action": action,
                                     "result": result, "observation": obs}) + "\n")
        assert result["success"], (action, result, obs["lastActionMessage"])
        self.observation = obs
        return obs

    def isolation_site(self):
        # Only object IDs revealed in the ordinary public UI are available.
        candidates = sorted(uid for uid, obj in self.known.items() if obj["name"] == "grass")
        for uid in candidates[:30]:
            obs = self.goto(uid)
            nearby = [obj for group in obs["nearbyObjects"]["objects"].values() for obj in group]
            if not any(obj["name"].startswith("quantum crystal ") for obj in nearby):
                return uid
        raise RuntimeError("No public isolation site found in 30 candidates")

    def repeat_measurements(self, rows, protocol, repeats):
        result, diagnostics = deepcopy(rows), []
        meter = next(uid for uid,obj in self.known.items() if obj["name"] == "radiation meter")
        site = self.isolation_site() if protocol.startswith("isolate_") else None
        for row in result:
            uid = row["uuid"]
            self.goto(uid)
            parent = None
            if site is not None:
                parent = int(re.search(r"\[uuid: (\d+)\]", self.known[uid]["description"])[1])
                self.act(action="PICKUP", arg1=uid)
                self.goto(site)
            location = self.observation["agentLocation"]
            values = []
            for _ in range(repeats):
                obs = self.act(action="USE", arg1=meter, arg2=uid)
                values.append(parse_measurement("radiation meter", obs["lastActionMessage"])["radiation"])
            if parent is not None:
                self.goto(parent)
                self.act(action="PUT", arg1=uid, arg2=parent)
            row["readings"]["radiation"] = float(min(values) if protocol.endswith("_min") else np.mean(values))
            diagnostics.append({"name":row["name"],"values":values,"mean":float(np.mean(values)),
                                "min":min(values),"max":max(values),"location":location,
                                "aggregate":row["readings"]["radiation"],
                                "returned_to_original_container":parent})
        save(self.output / "protocol_observations.json", result)
        save(self.output / "measurement_diagnostics.json", diagnostics)
        return result


def main(args):
    start, out = time.monotonic(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = [("Normal", 3), ("Challenge", 4)]
    save(out / "plan.json", {"cases": cases, "protocols": args.protocols,
        "repeats_per_crystal":args.repeats, "action_budget":700,
        "induction":"Unchanged finite grammar, fit tolerance, and MDL selector from reactor_program_grammar_v0",
        "scope":__doc__, "source_transfer":False,"api_requests":0,"gpu_work":False,
        "design_source":"vendor/DiscoveryWorld/discoveryworld/objects/ScienceTools.py:265-324 and discoveryworld/ScienceHelpers.py:12-62",
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0,str(VENDOR))
    results=[]
    try:
        with (out/"native_stdout.log").open("w",buffering=1) as log, redirect_stdout(log):
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
            for protocol in args.protocols:
                for difficulty,seed in cases:
                    folder=out/f"{protocol}_{difficulty.lower()}_seed_{seed}"
                    folder.mkdir()
                    api=DiscoveryWorldAPI(threadID=9300+len(results))
                    api.FRAME_DIR=str(folder/"frames")
                    assert api.loadScenario("Reactor Lab",difficulty,randomSeed=seed,numUserAgents=1)
                    collector=ProtocolCollector(api,folder)
                    rows=collector.collect()
                    collection_actions=collector.actions
                    rows=collector.repeat_measurements(rows,protocol,args.repeats)
                    measurement_actions=collector.actions-collection_actions
                    model=induce(rows)
                    save(folder/"program.json",model)
                    targets=[r for r in rows if r["observed_calibration_hz"] is None]
                    selected=model["selected"]
                    tuning=[tune(collector,row,pred) for row,pred in zip(targets,selected["target_predictions_hz"])] if selected else []
                    progress=collector.observation["taskProgress"]
                    result={"protocol":protocol,"difficulty":difficulty,"seed":seed,"selected":selected,
                            "collection_actions":collection_actions,"measurement_actions":measurement_actions,
                            "total_actions":collector.actions,"tuning":tuning,"public_task_progress":progress,
                            "completed_successfully":bool(progress) and all(t["completedSuccessfully"] for t in progress)}
                    save(folder/"result.json",result);results.append(result)
                    print(json.dumps({k:result[k] for k in ["protocol","difficulty","seed","completed_successfully","total_actions"]}),file=sys.__stdout__,flush=True)
                    del collector,api
        summary={"status":"completed","elapsed_s":time.monotonic()-start,"instances":len(results),
                 "tasks_completed":sum(r["completed_successfully"] for r in results),
                 "native_actions":sum(r["total_actions"] for r in results),"results":results,
                 "api_requests":0,"gpu_work":False,"scope":__doc__}
        save(out/"summary.json",summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ["results","scope"]}),flush=True)
    except Exception as error:
        save(out/"status.json",{"status":"failed","error":repr(error),"completed_instances":len(results)})
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--repeats",type=int,default=32)
    parser.add_argument("--protocols",nargs="+",choices=["repeat_mean","repeat_min","isolate_mean","isolate_min"],default=["repeat_mean","isolate_min"])
    main(parser.parse_args())
