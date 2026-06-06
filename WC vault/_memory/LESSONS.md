---
type: wc-lessons
updated: 2026-06-06
---

# What the model has learned

> Leak-free over **7** graded matches: outcome accuracy **71%**, Brier 0.4427, log loss 0.7639. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- friendly: 71% (7 matches)

## Systematic biases found (ranked by evidence)
- **draw** (friendly): model under-rated it (strength 0.469, over 2 wrong matches)
- **elo_gap** (friendly): model over-rated it (strength 0.441, over 2 wrong matches)
- **goal_volume** (friendly): model over-rated it (strength 0.074, over 2 wrong matches)
- **home_advantage** (friendly): model over-rated it (strength 0.3, over 2 wrong matches)
- **motivation** (friendly): model over-rated it (strength 0.469, over 2 wrong matches)
- **availability** (friendly): model under-rated it (strength 0.35, over 1 wrong matches)
