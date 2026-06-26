---
type: wc-lessons
updated: 2026-06-26
---

# What the model has learned

> Leak-free over **80** graded matches: outcome accuracy **69%**, Brier 0.4717, log loss 0.8066. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 62% (56 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.23, over 16 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.15, over 14 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.271, over 14 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.472, over 11 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.082, over 11 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.47, over 10 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.19, over 10 wrong matches)
- **availability** (group): model over-rated it (strength 0.053, over 7 wrong matches)
