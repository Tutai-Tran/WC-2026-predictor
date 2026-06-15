---
type: wc-lessons
updated: 2026-06-15
---

# What the model has learned

> Leak-free over **34** graded matches: outcome accuracy **74%**, Brier 0.3994, log loss 0.6948. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 50% (10 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **draw** (friendly): model under-rated it (strength 0.47, over 4 wrong matches)
- **goal_volume** (friendly): model over-rated it (strength 0.093, over 4 wrong matches)
- **goal_volume** (group): model under-rated it (strength 0.069, over 4 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.247, over 4 wrong matches)
- **motivation** (friendly): model over-rated it (strength 0.159, over 4 wrong matches)
- **motivation** (group): model under-rated it (strength 0.062, over 4 wrong matches)
- **availability** (friendly): model under-rated it (strength 0.35, over 3 wrong matches)
- **draw** (close_match): model under-rated it (strength 0.13, over 3 wrong matches)
