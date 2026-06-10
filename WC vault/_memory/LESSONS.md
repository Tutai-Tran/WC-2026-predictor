---
type: wc-lessons
updated: 2026-06-10
---

# What the model has learned

> Leak-free over **23** graded matches: outcome accuracy **83%**, Brier 0.3147, log loss 0.5819. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- friendly: 83% (23 matches)

## Systematic biases found (ranked by evidence)
- **draw** (friendly): model under-rated it (strength 0.47, over 4 wrong matches)
- **goal_volume** (friendly): model over-rated it (strength 0.093, over 4 wrong matches)
- **motivation** (friendly): model over-rated it (strength 0.159, over 4 wrong matches)
- **availability** (friendly): model under-rated it (strength 0.35, over 3 wrong matches)
- **elo_gap** (friendly): model over-rated it (strength 0.452, over 3 wrong matches)
- **home_advantage** (friendly): model over-rated it (strength 0.305, over 3 wrong matches)
- **elo_gap** (close_match): model over-rated it (strength 0.25, over 1 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.312, over 1 wrong matches)
