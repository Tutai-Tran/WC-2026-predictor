---
type: wc-lessons
updated: 2026-06-23
---

# What the model has learned

> Leak-free over **68** graded matches: outcome accuracy **69%**, Brier 0.4608, log loss 0.7868. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 61% (44 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.23, over 15 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.142, over 13 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.267, over 11 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.486, over 9 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.078, over 9 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.116, over 8 wrong matches)
- **availability** (group): model over-rated it (strength 0.053, over 7 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.523, over 7 wrong matches)
