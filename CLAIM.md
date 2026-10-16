# World Model Security: claim ledger

Status: research paused at the user’s request; archival/publication work only. A cross-system security mechanism is **not established**.

## Required final claim

The paper must identify an attacker-accessible structural condition under which a world-model consumer can lose real task utility, and an absent-condition case in which the same intervention cannot cause that loss. Evidence must cover at least three real pretrained systems, including a meaningful negative control. Prediction divergence and ranking changes are diagnostics, not task harm. Novelty must be argued against the nearest primary work, without numerical self-ratings.

## Security candidates and current diagnostic

**R001 — forecast/action binding:** if an external forecast service can reorder otherwise unchanged responses and a planner associates them with actions only by position, a change of association can select a lower-utility action; preserving candidate identities removes that route, and a policy that does not consume forecasts has no such route.

The bounded cache test completed, but the inspected native paths do not expose the hypothesized external ordering permission. DINO utility declines under the synthetic mapping error, Cosmos aggregate success is unchanged and cost improves, and WorldAgen's native action bypass is unchanged. Restoring IDs recovers the synthetic baseline by construction. **R001 is closed for interface mismatch and lack of a distinct mechanism.** This is not three-system security validation. No security candidate currently satisfies the full claim requirements.

R002 is closed for missing native deadline/partial-result resources. R003's actual native CEM test also closes: returned mean terminal reward3.690524 lies inside its ten elites' range[3.538413,3.887301], rather than below all of them. Nonconvex aggregation has direct prior work and no external inducer has been established here.

**R004 remains a diagnostic asset, not an attack claim:** a fixed correction fitted solely to seed1 prediction residuals lowers mean future MSE0.777127 to0.511728 on the reused30-root bank, while selected cumulative environment return falls109.127186 to103.668100 and terminal reward6.300613 to5.895755. Unlike oracle substitution, inference uses no target future. Unlike independent confirmation, these roots have already informed this project. The single fresh-root native comparison has completed: MSE decreases1.361239 to0.978856 and H15 cumulative return decreases29.943025 to28.036593, but H30 open-loop return **increases**98.235031 to100.029401. Both arms have no success at these horizons.

The same-root full native episode countercheck now resolves the sustained-return claim negatively:99-control return **increases747.432862 to760.171563**, and final success changes false to true; both arms ever succeed. First success is later (72 to97 controls), retained as a secondary outcome, not substituted for the failed primary return condition. This fixed correction/root is closed for harm-seeking expansion. Local early losses do not establish full-task loss, and a single opposite-direction episode does not establish safety. Full-episode forecast error was not measured.

The possible acceptance-policy statement is that prediction-error improvement alone does not guarantee downstream utility improvement. That implication is closely related to predict-then-optimize literature and is not claimed new. A distinct security mechanism, credible provider authority, cross-system evidence and a substantive contribution beyond this observation remain required.

**R005 closes a distinct proposal-dominant explanation on TD-MPC2:** reuse two saved dynamics adaptations and cross their native policy-proposal/evaluation roles. Four complete500-control walker-run episodes give RR612.082555, CC495.216321, CR674.885888, RC553.538964, where R is random-source reference and C cascade-source dynamics. Historical RR/CC trajectories exactly reproduce. C proposals with R evaluation improve return over RR; restoring R evaluation from CC also restores more than the original gap. The registered claim that candidate generation causes material harm despite restored evaluation is therefore unsupported on this case. Proposal-only restoration recovers49.9055%, below the fixed50% threshold, and is not rounded into a pass. Preserve the interaction and allfour outcomes; do not turn the failed gate into a new post-hoc security mechanism. This one reused development case is not an external attack, forecast-accuracy comparison or three-system validation.

## Evidence already available

- GenSafeNav R007: one exact history duplicate inside GST, removed from predictor output before the actor, changes complete-episode raw return30.185066415 to30.087496792 (-0.3232%). Both arms succeed without collision or cost; the duplicate arm reaches the goal in102 rather than119 controls. Pre-GST exact deduplication reproduces the clean physical trace, actor observations and forecasts exactly. The registered material-loss gate fails. This demonstrates a local causal influence under a researcher-added interface, not material harm, a verified external attacker permission or a cross-system mechanism. The fixed operation is closed without dose/source/seed search.

- GenSafeNav R006: reversing complete visible-person forecast/conformity rows gives two successful119-control episodes with no collision and cost0; actual return30.185066415 versus30.185057268. No material task loss is observed. Maximum action difference0.001496 exceeds the precommitted1e-5 tolerance, so exact or all-tolerances numerical invariance is not established. This single-case added-interface control is not validation of a shared security mechanism across systems.

- Cosmos: 30 fixed development roots, two first-action choices per root and 60 real continuations; success can range from 28 to 30 versus native 29, with native capped cost 3477 and worst selectable cost 3646. This is a local finite-bank bound, not a general safety guarantee.
- DINO-WM: two development banks show score/utility mismatch; they are not independent confirmatory replications. A separate frozen 30-root added-ranker confirmation now supports an offline true-future reversal: mean terminal reward 6.300613 to 4.758220, paired difference -1.542393, 95% percentile CI [-2.438788, -0.715561]. Eleven roots lose, eighteen tie and one improves. Fixed candidate0 and candidate2 outperform the original selector on average. This is a diagnostic result, not an attack or native CEM result.
- WorldAgen: corrected three-control replays have zero local reward/success span. Native policy does not consume the world forecast. This does not establish longer-horizon safety or a completed cross-system intervention experiment.

## Unproved requirements

R007 has completed and failed its material-loss gate. The three-source follow-up found no qualified new external operation. R008 is preregistered as a bounded diagnostic of the existing TD-MPC2 loss: separate cumulative reward and terminal-policy/Q dependence while holding C proposals fixed. It is not a new security mechanism; terminal-value extrapolation has close prior art, and Cosmos/DINO are not already validated positive/negative counterparts. More local interface variations would not themselves resolve the cross-system claim.

An attacker-accessible new mechanism; independent real-utility harm under that mechanism; native-versus-added-consumer separation; at least three-system conditional validation; a demonstrated absent-condition counterexample at the claim's scope; substantive differences from closest work; an updated manuscript supporting those claims.

## Closest-work boundaries

| Primary work | What it already establishes | Difference supported here; remaining gap |
|---|---|---|
| [Lambert et al., L4DC2020](https://proceedings.mlr.press/v120/lambert20a.html), [full text, sections4.2–4.3](https://arxiv.org/html/2002.04523v2) | Predictive-model training quality and downstream control need not agree; includes adversarial controller-performance manipulation under a likelihood constraint. | Our fixed residual is fitted without reward, then tested on pretrained visual planning and matched real replays. This implementation/protocol difference does not establish a new security mechanism. |
| [Bhamidipaty et al., 2026, sections1–2.2](https://arxiv.org/html/2605.15960v1) | Formalizes model exploitation as reversal of true/model policy ordering; it can arise from an optimizer without an external adversary. | Our finite-bank measurements and native execution checks are empirical diagnostics, not a new definition or theorem. An external inducing input and a distinct conditional mechanism remain absent. |
| [SWAAP, 2026, appendixE](https://arxiv.org/html/2606.18697v1) | Poisoned adaptation can reduce planning return; explicitly uses TD-MPC2 dynamics for both24 policy-guided proposals and512-candidate evaluation. | R005 crosses these two inference-time dynamics roles with all other weights frozen. The proposal-dominant condition fails; neither poisoning nor proposal dependence itself is new. |

An exact regret decomposition is an accounting identity under its specified intervention; it does not uniquely identify two native causal channels. Nonzero perfect-forecast mismatch is neither necessary for error-induced harm nor sufficient to show that improving forecasts reduces utility. Finite-bank utility bounds require that every possible selected action and its continuation remain within the recorded support; they do not cover adaptive CEM populations or an unrecorded elite mean.

See [IDEAS.md](IDEAS.md), [CLOSED.md](CLOSED.md), [ASSETS.md](ASSETS.md). The diagnostic manuscript under `reports/decision_leverage_paper/` is an asset, not completion of this security objective.
