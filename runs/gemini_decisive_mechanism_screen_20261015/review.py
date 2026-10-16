"""One bounded methods/idea review; not an experiment or evidence of harm."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wm_security.client import Client

out = Path(__file__).resolve().parent
if (out / 'calls.jsonl').exists():
    raise SystemExit('Retain the previous request; do not repeat this review.')
archive = ROOT / 'reports/research_archive'
entries = json.loads((archive / 'mechanisms.json').read_text())['entries']
models = json.loads((archive / 'models.json').read_text())['models']
payload = {
    'task': 'Select at most ONE decisive next research test for a general world-model-plus-agent security mechanism, after examining the recorded counterevidence. This is a new mechanism/readiness decision, not a repeat of the previous NanoWM protocol review. You may return no candidate. Do not invent outcomes, citations, interfaces or accessible weights.',
    'objective': 'Real pretrained world model, competent native agent, actual harm or privacy loss, causal controls, usable defense and generalization. A coherent ICLR-level contribution is desired; do not assign an unsupported numerical novelty score.',
    'rules': [
        'An explicitly assumed supply-chain threat or ordinary observation threat can be legitimate; do not demand a public deployment exploit or a separately writable Goal branch for all research.',
        'A mechanism need not harm only learned models. However a renamed generic effect with no new system contribution is insufficient.',
        'Do not silently shrink the original full goal to a toy witness, prediction difference, archival work or an easier added planner.',
        'Prefer a small decisive test on existing artifacts, models and real environments. No long training or broad search. Give an executable operation and falsifier, not a research theme.',
        'Distinguish untested ideas, invalid protocols, lack of resources, true negative experiments and lack of novelty. Reopening requires a concrete changed fact or corrected identification argument.',
        'Do not suggest any adaptive WorldAgen cached-metadata or GammaWorld work: those are outside this review. Do not generate attack payloads or operational exploit code.',
    ],
    'current_state': {
        'NanoWM': 'Released checkpoint strictly loads; its stochastic diffusion forecasts feed native CEM. Image Goal only enters objective. Three frozen official heldout reachable tasks have CPU zero-control results1/3. No native Clean episode yet. One16candidateH5 rollout15.94s under shared GPU load; unchanged100candidate/30iteration/20DDIM/two-plan root extrapolates112min, above45min resource decision. Proposed independent-vs-shared initialization noise is only a conditional sensitivity contrast, not yet a fixed security claim. Current expensive Clean is not automatically queued.',
        'resources': 'One shared RTX4090 currently occupied by other work. Existing local checkpoints and API are available. Avoid choosing unrelated experiments just because CPU or API is free.',
        'current_missing': 'A concrete new or legitimately reopened operation with an identifiable security claim. Existing complete claims remain unproven.',
    },
    'archived_mechanisms': [{key: r.get(key) for key in ('id','name','core_hypothesis','attacker_control','native_vs_added_interface','results_status','why_failed_or_limited','what_new_evidence_needed_to_reopen')} for r in entries if r.get('category') != 'sensitive_orchestration'],
    'available_models': [{key: r.get(key) for key in ('id','name','status','native_decision_path','recommended_reuse_status','limitations')} for r in models],
    'required_output': {
        'decision': 'one_candidate or none',
        'one_sentence_candidate_claim': 'string or null; hypothesis, not established finding',
        'actual_permissions_and_protected_inputs': 'string or null',
        'existing_carrier_and_native_path_to_check': 'string or null; no invented interfaces',
        'difference_from_archived_failure_and_prior_art': 'string or null; distinguish factual novelty from uncertainty',
        'minimum_decisive_test': 'string or null; concrete control, measured real utility/privacy and falsifier',
        'what_can_be_done_before_expensive_compute': 'string or null',
        'fatal_gap_or_reason_none': 'string or null',
        'confidence_limits': 'array'
    }
}
(out / 'request_payload.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
client = Client('gemini-3.8-flash', log_dir=out, timeout=60, max_calls=1)
start = time.monotonic()
try:
    result = client.json('You are a critical scientific methods reviewer. Return one JSON object using exactly the requested output keys. Treat all evidence as bounded and negative findings as constraints. Do not manufacture a candidate merely to comply.', payload, tag='one_decisive_next_mechanism', max_tokens=4096)
    (out / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    summary = {'status':'completed_review_only','api_requests':client.calls,'elapsed_seconds':time.monotonic()-start,'model':'gemini-3.8-flash','decision':result.get('decision'),'model_or_environment_experiment':False}
except Exception as error:
    # Client removes configured credentials from errors; only the error type is
    # copied here. The bounded call log retains the diagnostic.
    summary = {'status':'review_unavailable','api_requests':client.calls,'elapsed_seconds':time.monotonic()-start,'error_type':type(error).__name__,'model_or_environment_experiment':False}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
