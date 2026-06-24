---
type: wc-lessons
updated: 2026-06-24
---

# What the model has learned

> Leak-free over **72** graded matches: outcome accuracy **69%**, Brier 0.4639, log loss 0.7899. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 62% (48 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.23, over 16 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.142, over 13 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.271, over 12 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.474, over 10 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.101, over 10 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.51, over 8 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.116, over 8 wrong matches)
- **availability** (group): model over-rated it (strength 0.053, over 7 wrong matches)
