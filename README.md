# 🏭 Tool Wear Prediction from Motor Torque

> **Predicting the invisible from the cheap** — Using motor torque (free in most of cases) to estimate tool wear (expensive to measure)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Forgis/blob/main/Tool_Wear_Forgis.ipynb)

---

## 🎯 The Problem

In CNC milling, **tool wear** directly impacts part quality. A worn tool produces:
- Poor surface finish
- Dimensional errors  
- Eventually, catastrophic failure (broken tool, scrapped part, machine damage)

But here's the catch: **you can't see wear while the tool is cutting.**

### Current Solutions (and why they fail)

| Method | Problem |
|--------|---------|
| Fixed schedules | Wastes 20-30% of tool life (conservative) or risks failure (aggressive) |
| Force sensors | costs much per machine at large scale, requires downtime to install |
| Vision systems | Can't see the cutting edge during operation |

### Our Approach

What if we could predict wear from **data we already have**?

Every CNC machine logs motor torque at 500Hz. This signal is:
- ✅ Free (already recorded)
- ✅ Real-time (no downtime needed)
- ✅ Physics-connected to cutting forces

---

## 📐 The Physics: Why Torque → Wear Works

### Taylor's Tool Life Equation

In 1907, Frederick Taylor discovered that tool life follows a predictable pattern:

```
V × T^n = C
```

Where:
- `V` = cutting speed
- `T` = tool life (time until wear limit)
- `n, C` = constants for the tool/material combination

<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/05fdce32-cf27-4c38-b99e-544d122dba29" />
<img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/4eccef3c-6768-41a5-967b-dba4c7f9f43e" />


This tells us: **wear is deterministic, not random.** If we know the cutting conditions and can track degradation, we can predict remaining life.

### The Torque-Force Connection

When a tool cuts metal, it experiences **cutting forces** (Fx, Fy, Fz). These forces are directly measured by expensive dynamometers.

But the spindle motor must **supply torque** to overcome these forces:

```
τ_spindle ≈ k × F_cutting
```

The relationship isn't perfect (there's friction, inertia, etc.), but it's strong enough. Our analysis shows **r = 0.99** correlation between mean torque and mean cutting force.

### The Wear-Force Connection

As a tool wears:
1. The cutting edge becomes dull
2. More rubbing, less cutting
3. **Higher forces required** for the same material removal
4. Higher torque drawn by the spindle motor

This chain gives us: **Wear ↑ → Force ↑ → Torque ↑**

We validated this with **r = 0.93** correlation between wear and mean torque.

---

## 📊 Dataset: Leibniz Milling Dataset

We use a publicly available dataset from the Leibniz University of Hannover:

| Property | Value |
|----------|-------|
| Total runs | 6,418 |
| Machines | 3 (M1, M2, M3) |
| Tools per machine | 3 |
| Wear range | 3 - 158 μm |
| Torque sampling | 500 Hz |
| Force sampling | 25 kHz |

**Important:** We focus on **Machine 1** (1,856 runs) because:
- M2 has aliasing issues in force data
- M3 has coordinate system misalignment

This isn't a limitation — it's realistic. Real factories have sensor inconsistencies.

---

## 🔬 Phase 1: Exploratory Data Analysis

> **Goal:** Validate that predicting wear from torque is physically possible

### Cell 0-1: Data Loading

```python
# Each HDF5 file = one machining run
# Contains: torque (500Hz), force (25kHz), wear label (μm)
```

We load `filelist.csv` which maps 1,856 M1 files to their metadata (tool ID, run number, wear value).

### Cell 2: Physics Validation (Torque ≈ Force)

**Question:** Does motor torque actually correlate with cutting force?

**Method:**
1. Sample 100 runs across the wear range
2. Extract middle 80% of each signal (stable cutting, no entry/exit transients)
3. Calculate mean |torque| and mean |force| per run
4. Compute Pearson correlation

**Result:** `r = 0.992` ✅

This proves torque is a valid proxy for force — we can skip the expensive sensor.

### Cell 3: Wear Degradation Curve

**Question:** How does wear progress over time?

**Method:** Plot wear vs. cumulated cutting time for all 3 tools

**Result:** All tools show monotonic wear increase:
- **Break-in phase** (0-50 μm): Rapid initial wear as the edge settles
- **Linear phase** (50-110 μm): Steady, predictable wear — best for prediction
- **Failure phase** (>110 μm): Accelerated wear, tool replacement needed

Tool lifetime ≈ 3,600 seconds of contact time.

### Cell 4: Inference Map (Torque → Wear)

**Question:** Can we predict wear from torque statistics?

**Method:** 
1. Sample 300 runs
2. Extract mean |torque| from each
3. Plot against wear, colored by tool

**Result:** `r = 0.93` ✅

Clear positive correlation — as wear increases, torque increases. But notice: **each tool has a different "baseline"**. This will matter later.

---

## 🤖 Phase 2: Model Training & Validation

> **Goal:** Build a model and rigorously test if it actually works

### Cell 5: Baseline Model

**Features extracted per run:**
- `mean_torque` — average cutting effort
- `std_torque` — variation during cut
- `max_torque` — peak effort
- `min_torque` — minimum effort

**Model:** Linear Regression (simple, interpretable)

**Random Split Results:**
```
MAE = 7.4 μm
R²  = 0.88
```

Looks great! But wait...

### Cell 6: Leakage Check (Leave-One-Tool-Out)

**The problem:** Random splits mix data from the same tool in train and test. Since each tool has unique characteristics, the model might just learn "which tool is this?" instead of "what's the wear?"

**Method:** Train on 2 tools, test on the 3rd. Repeat for all combinations.

**Results:**
```
Train T2,T3 → Test T1: R² = 0.75
Train T1,T3 → Test T2: R² = 0.89
Train T1,T2 → Test T3: R² = 0.63

Average R² = 0.76
```

**Key Insight:** The drop from 0.88 → 0.76 reveals partial leakage. The random split was overly optimistic. **Always use grouped cross-validation for tool/machine data.**

### Cell 7: Cross-Machine Generalization

**Question:** Does a model trained on M1 work on M2/M3?

**Results:**
```
M1 → M2: R² = 0.30  (M2 has aliasing issues)
M1 → M3: R² = -1.27 (M3 has coord misalignment — worse than guessing mean!)
```

**Key Insight:** Cross-machine transfer fails completely. This is why **FactoryNet** (Phase 3) is needed — we need diverse training data.

### Cell 8: Per-Tool Normalization

**The insight:** Each tool has a different absolute torque level, but the *relative change* as wear increases is similar.

**Method:** Z-score normalize features within each tool before training.

**Results:**
```
Test T1: R² = 0.74
Test T2: R² = 0.79
Test T3: R² = 0.80

Average R² = 0.78
```

Now performance is **consistent across all tools** — no more variance between T1/T2/T3.

### Cell 9: Production Deployment Strategy

For a **new tool** with no history:

| Approach | Performance | When to Use |
|----------|-------------|-------------|
| Generic model | R² ≈ 0.76, MAE ≈ 11μm | Day 1 (no calibration needed) |
| Calibrated model | R² > 0.90 | After 20-30 cuts with wear measurements |

**Recommendation:** Start generic, then calibrate. The within-tool R² (0.89-0.95) shows the ceiling we can reach.

---

## 🌐 Phase 3: FactoryNet (Future Work)

> **Goal:** Build a model that generalizes across ANY machine

### The Problem We Discovered

- M1 model doesn't work on M2/M3
- Each machine has different sensor characteristics
- Current ML assumes i.i.d. data — factories violate this

### The Solution: Domain-Agnostic Training

**FactoryNet** would be trained on:
1. **Diverse real data** — Many machines, many tools, many conditions
2. **Synthetic data** — Physics-based simulation to fill gaps
3. **Domain adaptation** — Learn features that transfer across machines

### Architecture Concept

```
[Raw Torque Signal]
        ↓
[Feature Extractor] ← Learns machine-agnostic patterns
        ↓
[Domain Classifier] ← Adversarial training to forget machine ID
        ↓
[Wear Predictor] ← Outputs wear estimate
```

The key insight: Force the model to learn **physics** (wear → torque relationship) rather than **machine quirks** (sensor calibration, mounting position, etc.).

---

## 📈 Key Results Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Torque-Force correlation | r = 0.99 | Torque is a valid force proxy |
| Torque-Wear correlation | r = 0.93 | Wear prediction is feasible |
| Random split R² | 0.88 | Overly optimistic (leakage) |
| Leave-One-Tool-Out R² | 0.76 | Honest cross-tool estimate |
| Within-tool R² | 0.89-0.95 | Ceiling with calibration |
| Cross-machine R² | -1.27 to 0.30 | Transfer fails (needs FactoryNet) |

---

## 🧠 Summary Key Insights

1. **Random splits lie.** For grouped data (tools, machines, patients, users), always use grouped cross-validation.

2. **Correlation ≠ Causation, but physics helps.** We didn't just find a correlation — we understand *why* torque relates to wear (Taylor's equation, force chains).

3. **Simple models first.** Linear regression achieved R² = 0.76-0.88. Deep learning won't magically fix the generalization gap — that requires better data or domain adaptation.

4. **Normalization matters.** Per-tool normalization improved cross-tool consistency. This is a form of domain adaptation.

5. **Know your data's limits.** M2/M3 had sensor issues. A robust analysis acknowledges this rather than hiding it.

