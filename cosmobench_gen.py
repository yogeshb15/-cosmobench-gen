"""
CosmoBench-Gen
================
Infrastructure for evaluating an LLM's cosmology/physics reasoning: a
benchmark GENERATOR (randomized parameters -> no two runs are identical, so a
model can't have memorized the answer) paired with a symbolic/numeric GROUND
TRUTH solver and an automated GRADER that doesn't just say right/wrong -- it
recognizes *which* common mistake a wrong answer corresponds to.

This is the kind of tooling behind "design original research problems that
expose failure modes in advanced LLMs" style evaluation work: generate large
numbers of unique problem instances, know the exact correct answer for each,
and build a grader that gives partial credit for *diagnosis*, not just a
binary correct/incorrect.

Honesty note: this script does not call a live LLM API (no key is wired into
this environment). What it demonstrates is the benchmark + grading
INFRASTRUCTURE -- the problem generator, the symbolic ground truth, and the
mistake-pattern detector -- validated against a small, explicitly-labelled
set of SYNTHETIC test responses (hand-written to cover each known mistake
pattern, not real model output). Point the grader at any real model's answers
by replacing `synthetic_responses` with actual outputs.

Output
------
- results/leakage_resistance_demo.png  : same template, 3 random instances -> 3 different problems/answers
- results/grader_validation.png        : confusion-style matrix of the grader's self-validation
- Printed: example generated problems, grader diagnosis output, validation accuracy
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import json

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(3)

MPC_KM = 3.0856775814913673e19   # km per megaparsec
GYR_S = 3.15576e16               # seconds per gigayear (Julian year based)
C_KM_S = 299792.458


def h0_per_gyr(H0):
    """Convert H0 [km/s/Mpc] to units of 1/Gyr, from first principles (no magic constants)."""
    H0_per_s = H0 / MPC_KM
    return H0_per_s * GYR_S


# ==============================================================================
# TEMPLATE A: Age of the universe in flat LCDM
# ==============================================================================
def template_A_generate():
    Om = round(rng.uniform(0.24, 0.40), 3)
    H0 = round(rng.uniform(65.0, 74.0), 1)
    text = (f"For a spatially flat LCDM universe with matter density parameter "
            f"Om = {Om} and Hubble constant H0 = {H0} km/s/Mpc, compute the current "
            f"age of the universe in gigayears (Gyr).")
    return {"template": "A_age_of_universe", "params": {"Om": Om, "H0": H0}, "text": text}


def template_A_truth(p):
    Om, H0 = p["Om"], p["H0"]
    integrand = lambda a: 1.0 / (a * np.sqrt(Om / a**3 + (1 - Om)))
    I, _ = quad(integrand, 1e-8, 1.0, limit=200)
    return I / h0_per_gyr(H0)


def template_A_distractors(p):
    Om, H0 = p["Om"], p["H0"]
    d = {}
    # Mistake 1: ignored dark energy entirely (treated as matter-only / Einstein-de Sitter)
    d["ignored_dark_energy"] = (2.0 / 3.0) / h0_per_gyr(H0)
    # Mistake 2: forgot to convert H0 units (used km/s/Mpc directly as if it were 1/Gyr)
    integrand = lambda a: 1.0 / (a * np.sqrt(Om / a**3 + (1 - Om)))
    I, _ = quad(integrand, 1e-8, 1.0, limit=200)
    d["forgot_unit_conversion"] = I / H0
    return d


# ==============================================================================
# TEMPLATE B: Deceleration parameter for constant-w dark energy
# ==============================================================================
def template_B_generate():
    Om = round(rng.uniform(0.25, 0.40), 3)
    w = round(rng.uniform(-1.4, -0.6), 2)
    text = (f"For a flat universe with matter density Om = {Om} and a dark-energy "
            f"component with constant equation-of-state parameter w = {w}, compute "
            f"today's deceleration parameter q0 = Om/2 + (1+3w)(1-Om)/2.")
    return {"template": "B_deceleration_parameter", "params": {"Om": Om, "w": w}, "text": text}


def template_B_truth(p):
    Om, w = p["Om"], p["w"]
    return Om / 2 + (1 + 3 * w) * (1 - Om) / 2


def template_B_distractors(p):
    Om, w = p["Om"], p["w"]
    d = {}
    # Mistake 1: implicitly assumed w=-1 (forgot the (1+3w) factor -> used the LCDM-only formula)
    d["assumed_w_minus1"] = Om / 2 - (1 - Om)
    # Mistake 2: sign error on the w term
    d["sign_error_on_w"] = Om / 2 + (1 - 3 * w) * (1 - Om) / 2
    return d


# ==============================================================================
# TEMPLATE C: Low-z luminosity distance (two-step: derive q0, then apply expansion)
# ==============================================================================
def template_C_generate():
    Om = round(rng.uniform(0.25, 0.40), 3)
    H0 = round(rng.uniform(65.0, 74.0), 1)
    z = round(rng.uniform(0.02, 0.12), 3)
    text = (f"For a flat LCDM universe with Om = {Om} and H0 = {H0} km/s/Mpc, compute the "
            f"luminosity distance D_L (in Mpc) to an object at redshift z = {z}, using the "
            f"low-z expansion D_L = (c*z/H0)*(1 + (1-q0)*z/2), where q0 must first be "
            f"derived from Om for a flat LCDM model.")
    return {"template": "C_luminosity_distance", "params": {"Om": Om, "H0": H0, "z": z}, "text": text}


def template_C_truth(p):
    Om, H0, z = p["Om"], p["H0"], p["z"]
    q0 = 1.5 * Om - 1.0
    return (C_KM_S * z / H0) * (1 + (1 - q0) * z / 2)


def template_C_distractors(p):
    Om, H0, z = p["Om"], p["H0"], p["z"]
    d = {}
    # Mistake 1: naive Hubble law, dropped the deceleration correction term entirely
    d["dropped_correction_term"] = C_KM_S * z / H0
    # Mistake 2: forgot the dark-energy contribution to q0 (used matter-only q0 = Om/2)
    q0_wrong = Om / 2
    d["matter_only_q0"] = (C_KM_S * z / H0) * (1 + (1 - q0_wrong) * z / 2)
    return d


TEMPLATES = {
    "A_age_of_universe": (template_A_generate, template_A_truth, template_A_distractors),
    "B_deceleration_parameter": (template_B_generate, template_B_truth, template_B_distractors),
    "C_luminosity_distance": (template_C_generate, template_C_truth, template_C_distractors),
}

# ------------------------------------------------------------------------------
# Problem bank generation (leakage resistance: every call gives new numbers)
# ------------------------------------------------------------------------------
def generate_bank(n_per_template=15):
    bank = []
    for name, (gen, truth_fn, distractor_fn) in TEMPLATES.items():
        for _ in range(n_per_template):
            prob = gen()
            prob["truth"] = truth_fn(prob["params"])
            prob["distractors"] = distractor_fn(prob["params"])
            bank.append(prob)
    return bank


print("Generating benchmark problem bank...")
bank = generate_bank(n_per_template=15)
print(f"  {len(bank)} problem instances across {len(TEMPLATES)} templates "
      f"({len(bank)//len(TEMPLATES)} instances/template)")

print("\nLeakage-resistance check: same template, three independent calls ->")
for i in range(3):
    p = template_B_generate()
    p["truth"] = template_B_truth(p["params"])
    print(f"  instance {i+1}: Om={p['params']['Om']}, w={p['params']['w']}  "
          f"-> q0 = {p['truth']:.4f}")
print("  (three structurally identical questions, three different numeric answers "
      "and three different correct solutions -- memorizing one instance's answer "
      "does not help on the next)")

# ------------------------------------------------------------------------------
# Grader: classify a candidate numeric answer against truth + known distractors
# ------------------------------------------------------------------------------
def grade_answer(problem, candidate, rel_tol=0.01, abs_tol=1e-6):
    def close(a, b):
        return abs(a - b) <= max(abs_tol, rel_tol * abs(b))

    if close(candidate, problem["truth"]):
        return "correct", None
    for name, val in problem["distractors"].items():
        if close(candidate, val):
            return "incorrect", name
    return "incorrect", "unrecognized_error"


# ------------------------------------------------------------------------------
# Validate the grader against a small, explicitly SYNTHETIC test set.
# These are hand-constructed to exercise each code path (correct / each known
# distractor / a genuinely random wrong answer) -- NOT real model outputs.
# ------------------------------------------------------------------------------
print("\nValidating the grader against synthetic test cases "
      "(hand-built, NOT real model outputs)...")

synthetic_cases = []
for template_name, (gen, truth_fn, distractor_fn) in TEMPLATES.items():
    p = gen()
    p["truth"] = truth_fn(p["params"])
    p["distractors"] = distractor_fn(p["params"])

    synthetic_cases.append({"problem": p, "candidate": p["truth"], "expected_label": "correct"})
    for dname, dval in p["distractors"].items():
        synthetic_cases.append({"problem": p, "candidate": dval, "expected_label": dname})
    synthetic_cases.append({
        "problem": p, "candidate": p["truth"] * rng.uniform(1.5, 3.0),
        "expected_label": "unrecognized_error",
    })

n_correct_diag = 0
rows = []
for case in synthetic_cases:
    verdict, error_type = grade_answer(case["problem"], case["candidate"])
    predicted_label = "correct" if verdict == "correct" else error_type
    match = predicted_label == case["expected_label"]
    n_correct_diag += match
    rows.append((case["problem"]["template"], case["expected_label"], predicted_label, match))

diag_accuracy = n_correct_diag / len(synthetic_cases)
print(f"  {len(synthetic_cases)} synthetic cases, grader diagnostic accuracy: "
      f"{diag_accuracy*100:.1f}%")
print("\n  Example grading output:")
for r in rows[:6]:
    print(f"    template={r[0]:<25s} expected={r[1]:<22s} predicted={r[2]:<22s} "
          f"{'OK' if r[3] else 'MISS'}")

# ------------------------------------------------------------------------------
# Save the full problem bank + grading validation table
# ------------------------------------------------------------------------------
with open("results/problem_bank_sample.json", "w") as f:
    json.dump(bank[:9], f, indent=2, default=float)
print("\nSaved results/problem_bank_sample.json (9 example generated problems)")

# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, template_name in zip(axes, TEMPLATES.keys()):
    gen, truth_fn, _ = TEMPLATES[template_name]
    vals = []
    for _ in range(200):
        p = gen()
        vals.append(truth_fn(p["params"]))
    ax.hist(vals, bins=25, color="#4C72B0", alpha=0.85)
    ax.set_title(template_name)
    ax.set_xlabel("correct answer value")
    ax.set_ylabel("count (200 random instances)")
plt.suptitle("Leakage resistance: answer distribution across randomly generated instances per template")
plt.tight_layout()
plt.savefig("results/leakage_resistance_demo.png", dpi=150)
print("Saved results/leakage_resistance_demo.png")

labels = sorted(set(r[1] for r in rows))
label_idx = {l: i for i, l in enumerate(labels)}
conf = np.zeros((len(labels), len(labels)))
for _, exp, pred, _ in rows:
    conf[label_idx[exp], label_idx.get(pred, label_idx[exp])] += 1

fig2, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(conf, cmap="Blues")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("grader's diagnosis"); ax.set_ylabel("expected diagnosis")
ax.set_title(f"Grader self-validation on {len(synthetic_cases)} synthetic cases\n"
             f"({diag_accuracy*100:.1f}% diagnostic accuracy)")
for i in range(len(labels)):
    for j in range(len(labels)):
        if conf[i, j] > 0:
            ax.text(j, i, int(conf[i, j]), ha="center", va="center",
                     color="white" if conf[i, j] > conf.max() / 2 else "black", fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("results/grader_validation.png", dpi=150)
print("Saved results/grader_validation.png")

with open("results/summary.txt", "w") as f:
    f.write("CosmoBench-Gen results\n")
    f.write("========================\n")
    f.write(f"Templates: {len(TEMPLATES)} (age of universe, deceleration parameter, "
            f"luminosity distance)\n")
    f.write(f"Problem bank: {len(bank)} instances ({len(bank)//len(TEMPLATES)}/template)\n")
    f.write(f"Grader diagnostic accuracy on {len(synthetic_cases)} synthetic "
            f"validation cases: {diag_accuracy*100:.1f}%\n")

print("\nDone.")
