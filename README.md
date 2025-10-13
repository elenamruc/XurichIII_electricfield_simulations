# XurichIII_electricfield_simulations

# Electric field simulations for Xurich III

This repository contains tools to analyze and visualize electric field simulations obtained from **COMSOL Multiphysics**, using both **2D simulation data** and **3D superposition methods**. It provides Jupyter and Python scripts to post-process simulation outputs, generate diagnostic plots, and compare results across geometries.

## Repository Structure

```
.
├── 2D_analysis_simul.ipynb
├── superposition_3D_analysis.py
├── results_pitch_27mm.csv
├── results_pitch_1mm.csv
└── README.md
```

## Contents

### 1. 2D_analysis_simul.ipynb

A Jupyter notebook for processing **2D COMSOL simulations** of electric fields between gate and cathode electrodes. It includes:

* Data loading and parsing of `data_*.txt` tables exported from COMSOL.
* Conversion between wide and long formats for easier analysis.
* Calculation of:

  * Electric field uniformity (ΔE),
  * Radial distortion (|Er/Ez|),
  * Comparison with theoretical values (E_theory).
* Overlay and comparative plots for multiple geometries and cathode voltages.

**Usage:**

```bash
jupyter notebook 2D_analysis_simul.ipynb
```

Inside the notebook, you can:

* Adjust parameters such as `DRIFT_LENGTH_CM`, `VC_LIST`, or `UNIF_PERCENTILE`.
* Regenerate plots for all geometries using:

```python
for Vc in [-1000, -500, -60, 0, 60, 500, 1000]:
    plot_overlay_vs_Vgate(Vc, y="deltaE")
```

* Compare simulation results with theoretical expectations:

```python
plot_gate27_field_bias_colored(L_cm=3.0)
```

---

### 2. superposition_3D_analysis.py

A Python script for **3D field reconstruction** using the **superposition method**. It allows you to optimize electrode voltages and compute field maps combining pre-simulated components (e.g., gate, anode, cathode, PMTs).

**Main features:**

* Optimization of gate, anode, cathode, and ring voltages for a given target field.
* Discrete PMT grid search (optional) for refining boundary conditions.
* Field uniformity and inhomogeneity metrics (`mean`, `inhEz`).
* Automatic generation of summary plots:

  * `optimal_voltages_vs_E_3D_*.png`
  * `deviation_vs_E_3D_*.png`
  * `deflection_vs_E_3D_*.png`
  * `theta_vs_E_3D_*.png`

**Usage:**

```bash
python superposition_3D_analysis.py
```

This script can be adapted to use your own COMSOL CSV results (see below).

---

### 3. COMSOL Output Files

The `.csv` files (`results_pitch_27mm.csv`, `results_pitch_1mm.csv`) contain the **superposition basis fields** exported from COMSOL for different electrode pitches.

Each file corresponds to:

* A unit potential configuration for each electrode.
* Data used by `superposition_3D_analysis.py` to reconstruct total electric fields.

**Typical columns include:**

* `x`, `y`, `z`: grid coordinates (in cm or mm)
* `Ex`, `Ey`, `Ez`: electric field components (V/cm)
* `V`: potential (V)
