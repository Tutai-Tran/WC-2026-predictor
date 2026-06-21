---
type: wc-lessons
updated: 2026-06-21
---

# What the model has learned

> Leak-free over **61** graded matches: outcome accuracy **69%**, Brier 0.4656, log loss 0.7923. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 60% (37 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.285, over 13 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.119, over 12 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.486, over 9 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.232, over 9 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.116, over 8 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.029, over 7 wrong matches)
- **availability** (group): model under-rated it (strength 0.018, over 6 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.539, over 6 wrong matches)
