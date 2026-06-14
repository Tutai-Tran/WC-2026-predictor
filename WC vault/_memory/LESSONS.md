---
type: wc-lessons
updated: 2026-06-14
---

# What the model has learned

> Leak-free over **30** graded matches: outcome accuracy **77%**, Brier 0.3893, log loss 0.6776. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 50% (6 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **draw** (friendly): model under-rated it (strength 0.47, over 4 wrong matches)
- **goal_volume** (friendly): model over-rated it (strength 0.093, over 4 wrong matches)
- **motivation** (friendly): model over-rated it (strength 0.159, over 4 wrong matches)
- **availability** (friendly): model under-rated it (strength 0.35, over 3 wrong matches)
- **elo_gap** (friendly): model over-rated it (strength 0.452, over 3 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.442, over 3 wrong matches)
- **home_advantage** (friendly): model over-rated it (strength 0.305, over 3 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.142, over 3 wrong matches)
