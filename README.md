# CosmoBench-Gen

**Infrastructure for evaluating LLM cosmology/physics reasoning: randomized problem generation (leakage-resistant), symbolic ground-truth solving, and an automated grader that diagnoses *which* common mistake a wrong answer matches — not just right/wrong.**

## Motivation

Evaluating an LLM's scientific reasoning well means more than a fixed set of textbook problems a model may have memorized during training. This project builds the infrastructure real evaluation work actually needs: a generator that produces unlimited unique problem instances from physics templates, an exact ground-truth solver for each, and a grader that recognizes specific, named failure patterns (a forgotten term, a sign error, a unit mistake) — the same kind of rubric-based, root-cause-oriented review used in research-level AI model evaluation.

**Honesty note:** this script does not call a live LLM API — no key is wired into the build environment it was developed in. What's demonstrated is the benchmark + grading *infrastructure*, validated against a small, explicitly-labeled set of synthetic test responses (hand-written to exercise each mistake pattern, not real model output). Point the grader at any real model's answers by replacing the synthetic response set.

## Method

Three physics templates, each with randomized parameters and a closed-form or numerically-integrated ground truth:

1. **Age of the universe** in flat ΛCDM (numerical integration of the Friedmann equation, unit conversions derived from first principles — km/Mpc and s/Gyr — not hardcoded magic numbers)
2. **Deceleration parameter** q0 for constant-w dark energy
3. **Low-z luminosity distance** (two-step reasoning: derive q0 from Ωm, then apply the expansion formula)

For each template, a library of "common mistake" distractor functions computes exactly what answer a specific reasoning error would produce (e.g., forgetting the dark-energy contribution, a sign flip, dropping a correction term). The grader checks a candidate answer against the true value *and* every known distractor, tagging the specific error type when it matches one.

## Results

| Metric | Value |
|---|---|
| Problem bank | 45 instances across 3 templates (15/template), fully randomized |
| Grader diagnostic accuracy | **100%** on 12 synthetic validation cases (correct + every distractor type + an unrecognized-error case, per template) |

![Leakage resistance](results/leakage_resistance_demo.png)
![Grader validation](results/grader_validation.png)

The leakage-resistance plot shows the spread of correct answers across 200 random instances per template — every run is numerically distinct, so memorizing one instance's answer provides no advantage on the next. The grader validation matrix is a clean diagonal: every synthetic test case is classified as exactly what it was constructed to be.

## Run it

```bash
pip install -r requirements.txt
python cosmobench_gen.py
```

## Stack

Python · NumPy · SciPy (symbolic/numeric ground truth) · Matplotlib

## License

All rights reserved — see [LICENSE](LICENSE). This repository is shared publicly to demonstrate the work; it is not open source, and no use (including research or academic use) is permitted without written permission.

---
*by Yogesh Bhardwaj — PhD (Applied Mathematics), Delhi Technological University.*
