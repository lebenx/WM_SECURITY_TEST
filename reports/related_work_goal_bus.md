# Goal-bus claim: related-work positioning

> **Historical positioning memo; not the current novelty or evidence assessment.** The original text below is preserved, including its references to a "current claim" and its earlier Goal-invariance premise. Those statements are constrained or superseded by the [current evidence index](research_archive/evidence.json) and [physical-invariance specification](../configs/goal_bus_physical_invariance_spec.json). Consult the [literature registry](research_archive/papers.json) for each source's recorded reading scope and the [research knowledge base](research_archive/index.html) for current evidence and reusable assets. This archive notice neither rechecks the cited papers nor establishes a novelty score.

This memo records the focused literature check used to position the current claim. It is a novelty boundary, not a claim that the mechanism is absent from every unpublished or newly posted paper.

## Closest architectural neighbors

- [WorldVLA](https://arxiv.org/abs/2506.21539) unifies action and image understanding/generation and studies an attention mask that suppresses prior-action information during autoregressive action-chunk generation. Its target is action-generation error propagation. Our intervention asks a different question: with the physical image/state and executed actions fixed, can legal task text still alter the future-world branch through an intermediate action carrier?
- [Causal World Modeling for Robot Control / LingBot-VA](https://arxiv.org/abs/2601.21998) uses a shared latent space for vision and action tokens, closed-loop rollout, and asynchronous inference. This motivates shared world-action pathways, but does not formulate task text changing a fixed-action physical forecast as a security invariant or test a frozen-weight edge split.
- [AffordanceWAM](https://arxiv.org/abs/2609.22332) uses separately parameterized World and Action Experts coupled by masked joint self-attention, with affordance fields as a structured interface. Its directional separation is a relevant design comparator; our evidence targets the failure mode of a shared action carrier in an already released instruction-conditioned checkpoint.
- [When to Trust Imagination](https://arxiv.org/abs/2605.06222) adds a verifier that compares imagined futures with real observations and language to adapt action-chunk execution. It addresses rollout reliability after prediction; it does not isolate or repair a text-to-world causal path inside a frozen denoiser.

## Security distinction

The claim is narrower than “language affects a robot” and broader than a single prompt-leak example. The intervention holds image, proprioception, executed action, rewards, labels, weights, and gradients fixed; the only attacker operation is selecting another legal task instruction. The main invariant is physical: a task string should not change the next-world forecast when the physical state and executed action are fixed. Policy sensitivity is retained as a control, because changing the task can legitimately change the desired action.

The distinctive evidence is the matched causal ladder: a direct world-row text cut is insufficient; removing text from the intermediate action/value carriers restores exact invariance; alternate recorded actions still change the repaired world branch. This distinguishes the proposed bus mechanism from direct language conditioning, generic action-mask design, ordinary model quality comparisons, and parameter-update interference.

## Positioning sentence for the paper

> Existing world-action models study how attention masks and shared representations improve action generation or future prediction; we identify and repair a security-relevant failure mode in which legal task text crosses a shared action-token carrier and changes a fixed-state/fixed-action world forecast, despite a masked direct goal-to-world edge.

The current experiments support this positioning across WorldAgen and AlphaBrain. They do not establish universality across all world-action architectures, and they do not yet establish task-level harm on a successful clean policy.

## 2026 world-action-model positioning check

Recent WAM work makes the boundary more important rather than removing it. [World Guidance](https://arxiv.org/abs/2602.22010) explicitly injects predicted future conditions into action inference, while [JEPA-WAM](https://arxiv.org/abs/2608.09381) couples latent transition prediction and action generation through a shared predictor. These papers study capability and alignment; they do not define a fixed-state/fixed-action Goal-invariance test or a matched decision-invariance test. The present paper should therefore position the bus as an audit primitive for shared world-action interfaces, rather than claim that shared representation itself is novel.

The security contribution is the causal separation: legal task text is allowed to change policy-facing behavior, but its path into a fixed-action physical forecast is treated as an integrity boundary. The paraphrase control and the matched policy-only cut are useful here because they distinguish task-language sensitivity from the specific world-branch intervention.
