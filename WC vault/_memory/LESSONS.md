---
type: wc-lessons
updated: 2026-06-07
---

# What the model has learned

> Leak-free over **12** graded matches: outcome accuracy **83%**, Brier 0.3185, log loss 0.59. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- friendly: 83% (12 matches)

## Systematic biases found (ranked by evidence)
- **draw** (friendly): model under-rated it (strength 0.469, over 2 wrong matches)
- **elo_gap** (friendly): model over-rated it (strength 0.441, over 2 wrong matches)
- **goal_volume** (friendly): model over-rated it (strength 0.074, over 2 wrong matches)
- **home_advantage** (friendly): model over-rated it (strength 0.3, over 2 wrong matches)
- **motivation** (friendly): model over-rated it (strength 0.469, over 2 wrong matches)
- **availability** (friendly): model under-rated it (strength 0.35, over 1 wrong matches)
