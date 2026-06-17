---
type: wc-lessons
updated: 2026-06-17
---

# What the model has learned

> Leak-free over **41** graded matches: outcome accuracy **66%**, Brier 0.4773, log loss 0.8024. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 41% (17 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.275, over 9 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.001, over 8 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.493, over 6 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.204, over 6 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.204, over 6 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.13, over 5 wrong matches)
- **availability** (group): model over-rated it (strength 0.074, over 4 wrong matches)
- **draw** (close_match): model under-rated it (strength 0.156, over 4 wrong matches)
