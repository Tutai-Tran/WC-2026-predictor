---
type: wc-lessons
updated: 2026-06-27
---

# What the model has learned

> Leak-free over **86** graded matches: outcome accuracy **67%**, Brier 0.4792, log loss 0.8176. The biases below are hypotheses (the audit trail); the model's parameters are where learning is actually applied, only after it validates out-of-sample.

## Accuracy by segment
- group: 61% (62 matches)
- friendly: 83% (24 matches)

## Systematic biases found (ranked by evidence)
- **motivation** (group): model under-rated it (strength 0.23, over 16 wrong matches)
- **goal_volume** (group): model over-rated it (strength 0.098, over 15 wrong matches)
- **tactical** (mismatch): model under-rated it (strength 0.271, over 14 wrong matches)
- **elo_gap** (mismatch): model over-rated it (strength 0.461, over 12 wrong matches)
- **home_advantage** (host): model under-rated it (strength 0.171, over 12 wrong matches)
- **home_advantage** (group): model over-rated it (strength 0.082, over 11 wrong matches)
- **draw** (mismatch): model under-rated it (strength 0.47, over 10 wrong matches)
- **availability** (group): model over-rated it (strength 0.053, over 7 wrong matches)
