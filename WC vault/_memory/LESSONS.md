---
type: wc-lessons
updated: 2026-06-15
---

# What the model has learned

> Leak-free over **37** graded matches: outcome accuracy **70%**, Brier 0.4519, log loss 0.7699. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 46% (13 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.219, over 6 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.491, over 5 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.011, over 5 wrong matches)
- **draw** (close_match): model under-rated it (strength 0.156, over 4 wrong matches)
- **draw** (friendly): model under-rated it (strength 0.47, over 4 wrong matches)
- **goal_volume** (friendly): model over-rated it (strength 0.093, over 4 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.062, over 4 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.247, over 4 wrong matches)
