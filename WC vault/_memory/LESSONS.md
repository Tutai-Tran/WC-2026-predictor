---
type: wc-lessons
updated: 2026-06-06
---

# What the model has learned

> Leak-free over **4** graded matches: outcome accuracy **75%**, Brier 0.4166, log loss 0.7292. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- friendly: 75% (4 matches)

## Systematic biases found (ranked by evidence)
- **draw** (friendly): model under-rated it (strength 0.45, over 1 wrong matches)
- **elo_gap** (friendly): model over-rated it (strength 0.4, over 1 wrong matches)
- **goal_volume** (friendly): model under-rated it (strength 0.25, over 1 wrong matches)
- **home_advantage** (friendly): model over-rated it (strength 0.3, over 1 wrong matches)
- **motivation** (friendly): model over-rated it (strength 0.35, over 1 wrong matches)
