---
type: wc-lessons
updated: 2026-06-09
---

# What the model has learned

> Leak-free over **18** graded matches: outcome accuracy **83%**, Brier 0.2951, log loss 0.5602. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- friendly: 83% (18 matches)

## Systematic biases found (ranked by evidence)
- **draw** (friendly): model under-rated it (strength 0.459, over 3 wrong matches)
- **goal_volume** (friendly): model under-rated it (strength 0.068, over 3 wrong matches)
- **home_advantage** (friendly): model over-rated it (strength 0.305, over 3 wrong matches)
- **motivation** (friendly): model over-rated it (strength 0.401, over 3 wrong matches)
- **availability** (friendly): model under-rated it (strength 0.326, over 2 wrong matches)
- **elo_gap** (friendly): model over-rated it (strength 0.441, over 2 wrong matches)
- **elo_gap** (close_match): model over-rated it (strength 0.25, over 1 wrong matches)
- **tactical** (close_match): model under-rated it (strength 0.3, over 1 wrong matches)
