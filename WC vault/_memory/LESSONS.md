---
type: wc-lessons
updated: 2026-06-15
---

# What the model has learned

> Leak-free over **36** graded matches: outcome accuracy **72%**, Brier 0.4228, log loss 0.732. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 50% (12 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **goal_volume** (group): model over-rated it (strength 0.011, over 5 wrong matches)
- **motivation** (group): model under-rated it (strength 0.14, over 5 wrong matches)
- **draw** (close_match): model under-rated it (strength 0.156, over 4 wrong matches)
- **draw** (friendly): model under-rated it (strength 0.47, over 4 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.464, over 4 wrong matches)
- **goal_volume** (friendly): model over-rated it (strength 0.093, over 4 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.247, over 4 wrong matches)
- **motivation** (friendly): model over-rated it (strength 0.159, over 4 wrong matches)
