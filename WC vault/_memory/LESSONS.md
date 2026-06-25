---
type: wc-lessons
updated: 2026-06-25
---

# What the model has learned

> Leak-free over **78** graded matches: outcome accuracy **69%**, Brier 0.4703, log loss 0.8038. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 63% (54 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.23, over 16 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.15, over 14 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.272, over 13 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.472, over 11 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.082, over 11 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.501, over 9 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.182, over 9 wrong matches)
- **availability** (group): model over-rated it (strength 0.053, over 7 wrong matches)
