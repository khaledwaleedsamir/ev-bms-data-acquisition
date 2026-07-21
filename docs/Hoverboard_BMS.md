# Hoverboard Battery Management AI Project — Design Summary

## System Overview

**Hardware setup:**
- Hoverboard with 2 brushless DC motors
- Two 10s3p battery packs, each with a Daly BMS (100A balance):
  - Pack A: fresh, new cells (training pack)
  - Pack B: degraded, second-life cells (generalization test pack)
- Logger already working: UART/CAN to microcontroller/laptop, **shared/synchronized clock** across BMS and hoverboard MCU streams

**Research objective:** Two-model pipeline, closed-loop:

1. **Model 1 — SOC predictor:** takes battery + hoverboard parameters, outputs current SOC and predicted SOC at a future time horizon, based on current draw dynamics.
2. **Model 2 — Range-optimal speed controller:** takes Model 1's SOC prediction plus pack/hoverboard state, outputs an optimal speed command to maximize distance traveled on the remaining charge. Output is fed back to control the hoverboard — **closed-loop, re-issued every few seconds.**

**Generalization test:** Train both models on the fresh pack only. Verify Model 2 produces sensible optimal-speed behavior when run on the degraded pack, without having trained on it.

---

## Key Design Insight: Degradation Must Be an Explicit Feature

Battery aging changes internal resistance, usable capacity, and the voltage/SOC relationship — nonlinearly, and dependent on C-rate and temperature. If the model never sees these as explicit inputs, it has no mechanism to behave differently on the aged pack; it would just be extrapolating blindly.

**Implication:** Pack health (internal resistance, true capacity) must be measured directly via characterization tests and fed into the models as features — not inferred implicitly from driving data alone.

---

## Data to Collect

### From the Daly BMS (100A balance)
- Pack voltage, per-cell voltages (30 cells — useful for imbalance/degradation signal)
- Pack current (signed)
- Onboard SOC estimate (useful as baseline only — not trusted as ground truth label)
- Remaining capacity (Ah), cycle count
- Temperature sensors (cell + MOSFET)
- Protection/status flags (OV, UV, OT, OC)

### From the hoverboard
- Wheel speed (both motors)
- Motor controller current/duty cycle commands (if accessible)
- Control board voltage and temperature
- IMU/tilt angle if available (proxy for torque demand)

### Derived / engineered features (computed, not raw-logged)
- Internal resistance estimate (ΔV/ΔI over current transients) — **primary degradation-sensitive feature**
- Coulomb-counted SOC (ground truth label, anchored to a real capacity test — not the BMS's own SOC)
- Power draw (V×I), specific power
- Energy per distance (Wh/km), rolling window
- C-rate (current / nominal capacity) — normalizes across pack health better than raw current
- dV/dt, dT/dt trends
- Cell voltage spread (max–min across cells)
- Speed-vs-current slope (rolling resistance/drivetrain efficiency proxy)
- Cumulative Ah throughput / cycle count (explicit age/degradation input)

---

## Data Collection Plan

### Phase 0 — Characterization (both packs, no driving required)
- **Capacity test:** constant low-current discharge (e.g., C/3 for fresh, gentler C/5 for the degraded pack) from full to cutoff — gives true usable Ah and the OCV-SOC curve.
- **HPPC pulse tests:** at ~20/50/80% SOC, apply a known current step and measure instantaneous voltage drop to extract internal resistance.
- **Status:** ✅ Done on fresh pack. ❌ Not yet done on degraded pack — **this is the current blocking gap.**

> Without this on the degraded pack, there's no ground truth to evaluate whether Model 2's speed recommendation is actually good on that pack — this test anchors the entire generalization claim.

### Phase 1 — Driving data collection (fresh pack only, for training)
- Sampling rate: 1–10 Hz
- Vary speed profiles, rider weight (if possible), terrain, ambient temperature
- **Full discharge cycles needed:** ~15–25 cycles, varied driving profiles (for Model 1 — sequence dynamics need full-range SOC coverage, nonlinear at both extremes)
- **Steady-speed segments needed (for the Model 2 efficiency model):** deliberate 30–60s stretches at several constant speeds, repeated at a spread of SOC levels (~90/70/50/30/15%) and ideally a couple of ambient temperatures

### Phase 2 — Validation on degraded pack (test only, not training)
- 3–5 discharge cycles, driving profiles not used in training
- Used purely to evaluate Model 2's output against the now-known true capacity/resistance of the degraded pack

---

## Model 1 — SOC Predictor

- Architecture: sequence model (GRU/LSTM/small TCN) over a window of recent (V, I, T, speed, derived features)
- Outputs: current SOC + SOC at a chosen horizon (e.g., 10–60s ahead; multi-horizon output optional)
- Labels: coulomb-counted SOC anchored to Phase 0 capacity test — **not** the BMS's onboard SOC estimate (drifts with age, would reintroduce the generalization problem)

---

## Model 2 — Range-Optimal Speed Controller

Given the closed-loop, few-seconds-cycle requirement, this is a **feedback control problem**, not a one-shot prediction — errors get corrected on the next cycle rather than compounding.

**Recommended architecture: receding-horizon / greedy controller, not an end-to-end black box.**

1. Model 1 provides current SOC + predicted SOC at horizon h
2. A compact **efficiency model** — Wh/km as a function of (speed, pack resistance/capacity state, temperature) — trained on the steady-speed segments from Phase 1. Can be a small MLP or gradient-boosted tree; takes pack-health features as explicit inputs.
3. At each control cycle: evaluate a small set of candidate speeds through the efficiency model, pick the one minimizing predicted Wh/km subject to safety constraints (predicted SOC stays above a floor, current/temperature within safe bounds)
4. Issue that speed; re-run the whole loop a few seconds later with fresh state

**Why this generalizes better than RL or a single supervised black box:**
- Requires far less data than RL (no simulator/digital twin needed)
- The efficiency model's pack-health inputs are exactly what differs between packs — swapping in the degraded pack's measured resistance/capacity at test time doesn't require the model to have "seen" the degraded pack, since the efficiency model has a learned response surface over resistance/capacity as continuous inputs
- The search-and-pick step downstream is plain arithmetic — generalization risk is concentrated only in the efficiency model, which is the part with the experimental grounding to back it

---

## Open / Next Steps

1. **Run degraded-pack characterization** (capacity test + HPPC) — short (~afternoon), unblocks valid testing
2. Begin Phase 1 fresh-pack driving data collection in parallel (full-discharge cycles + steady-speed segments)
3. Train Model 1 and the Model 2 efficiency model on fresh-pack data
4. Collect Phase 2 validation cycles on the degraded pack
5. Compare controller's recommended speed/range against the degraded pack's now-known true physics to verify near-optimal behavior

**Still to define:**
- Exact SOC prediction horizon for Model 1
- Logging schema (column list, units, derived feature formulas) — offered, not yet built
- Efficiency-model + greedy controller pseudocode/implementation — offered, not yet built
