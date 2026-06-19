---
type: wc-lessons
updated: 2026-06-19
---

# What the model has learned

> Leak-free over **50** graded matches: outcome accuracy **66%**, Brier 0.4864, log loss 0.8205. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 50% (26 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **goal_volume** (group): model over-rated it (strength 0.097, over 11 wrong matches)
- **motivation** (group): model under-rated it (strength 0.256, over 11 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.476, over 8 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.174, over 8 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.029, over 7 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.14, over 7 wrong matches)
- **draw** (close_match): model under-rated it (strength 0.171, over 5 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.529, over 5 wrong matches)
