---
type: wc-lessons
updated: 2026-06-18
---

# What the model has learned

> Leak-free over **48** graded matches: outcome accuracy **67%**, Brier 0.4845, log loss 0.8179. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 50% (24 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **goal_volume** (group): model over-rated it (strength 0.079, over 10 wrong matches)
- **motivation** (group): model under-rated it (strength 0.295, over 10 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.491, over 7 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.029, over 7 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.225, over 7 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.204, over 6 wrong matches)
- **draw** (close_match): model under-rated it (strength 0.171, over 5 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.529, over 5 wrong matches)
