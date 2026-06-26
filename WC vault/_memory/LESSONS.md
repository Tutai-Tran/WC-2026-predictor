---
type: wc-lessons
updated: 2026-06-26
---

# What the model has learned

> Leak-free over **82** graded matches: outcome accuracy **68%**, Brier 0.4734, log loss 0.8083. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 62% (58 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.23, over 16 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.15, over 14 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.271, over 14 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.461, over 12 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.082, over 11 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.156, over 11 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.47, over 10 wrong matches)
- **availability** (group): model over-rated it (strength 0.053, over 7 wrong matches)
