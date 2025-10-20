# 🧭 Training Regimen: PGM + ConvLSTM + Genetic Evolutionary Optimization

This README describes a robust **optimization regime** (training regiment) combining:

- **Probabilistic Graphical Model (PGM)** — for structured dependencies, consistency constraints, and uncertainty quantification.
- **ConvLSTM** — for spatiotemporal sequence modeling on grids (e.g., video frames, radar maps, sensor fields).
- **Genetic / Evolutionary Algorithms (EA)** — for outer-loop optimization of hyperparameters, architectures, and training schedules.

> ✅ Use this template to implement a pipeline that **pretrains** a ConvLSTM, **fits** a PGM (e.g., CRF/HMM) on top of learned features, and **evolves** the full stack for best validation fitness.

---

## Table of Contents

1. #overview  
2. #architecture  
3. #data  
4. #training-schedule  
5. #configuration  
6. #commands--quickstart  
7. #evaluation  
8. #reproducibility  
9. #directory-structure  
10. #pseudocode  
11. #troubleshooting  
12. #roadmap

---

## 1) Overview

This training regimen targets **structured spatiotemporal prediction** problems—e.g., sequence labeling, event forecasting, segmentation over time—where:

- **ConvLSTM** learns the dynamics on spatial grids through time.
- **PGM** (e.g., Conditional Random Field, Hidden Markov Model, Bayesian Network) encodes domain constraints (smoothness, label transitions, topology) and improves **consistency** and **calibration**.
- **Evolutionary Algorithms** search the **hyperparameter space** (learning rates, kernel sizes, model depths) and **architecture decisions** (PGM type, CRF mean-field iterations, ConvLSTM layers) to maximize a metric (e.g., F1, mIoU, RMSE).

---

## 2) Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        Data Sequences                         │
│         (T frames × H × W × C) + optional labels              │
└───────────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────┐
│                         ConvLSTM Stack                        │
│  - Spatial convs + temporal recurrence                        │
│  - Outputs unary potentials / features                        │
└───────────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────┐
│                Probabilistic Graphical Model                  │
│  e.g., CRF (grid edges), HMM (temporal), or hybrid            │
│  - Enforces structure (smoothness, transitions)               │
│  - Inference: mean-field / belief propagation / Viterbi       │
└───────────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────┐
│                       Losses & Metrics                        │
│  - KLDiv (batch mean) / NLL / structured losses                    │
│  - Task metrics: KLDiv                  │
└───────────────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────┐
│              Evolutionary Algorithm (Outer Loop) 
│  -Extensible and implementing
│  - Population of configs, selection, crossover, mutation      │
│  - Fitness = validation metric                                │
│  - Parallel training/evaluation                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 3) Data

**Expected format** (customize as needed):

- `X`: sequences with shape `[N, T, H, W, C]` (or `[N, T, …]` for non-grid features).
- `Y`: labels per time step (e.g., `[N, T, H, W]` for segmentation or `[N, T]` for sequence tags).
- **Splits**: `train/val/test` with consistent normalization.
- **Augmentations**: spatial flips/rotations, temporal jitter, cutout, noise; ensure **label-preserving**.

> Tip: Consider **sliding windows** for long sequences; e.g., `T_window=8` with stride 4.

---

## 4) Training Schedule

### Phase 0 — Data & Feature Prep
- Normalize inputs; build train/val/test windows.
- Optionally precompute static features (edges, motion fields, domain masks).

### Phase 1 — ConvLSTM Pretraining
- Objective: **supervised** training on labels with cross-entropy / MSE.
- Output: **unary potentials** (logits) or **feature maps** per time step.
- Techniques:
  - Cosine LR schedule with warmup.
  - Mixed precision, gradient clipping.
  - Early stopping on **val metric**.

### Phase 2 — PGM Fitting
Choose your PGM:

- **CRF** (spatial or spatiotemporal):
  - Nodes: pixels/cells at each time step.
  - Edges: 4/8-neighbors (spatial), and temporal links.
  - Inference: mean-field (differentiable variants) or loopy BP.

- **HMM** / **Chain CRF** (temporal-only):
  - Strong for sequence tagging with label-transition priors.
  - Inference: forward-backward / Viterbi.

Fit PGM parameters using:
- **MLE**/**MAP** with regularization,
- Or **EM** if latent variables exist.

### Phase 3 — Joint Training (Alternating or End-to-End)
- **Alternating**:
  1. Fix ConvLSTM → fit PGM.
  2. Fix PGM → fine-tune ConvLSTM with PGM-informed loss.
- **End-to-end (if differentiable)**:
  - Insert CRF layer; backprop through mean-field iterations.
  - Loss mixes **unary** + **pairwise** + **regularization**.

Common losses:
- `L = α * CrossEntropy(unary) + β * StructuredNLL(PGM) + γ * Consistency`
- Tune `(α, β, γ)` via EA.

### Phase 4 — Evolutionary Optimization (Outer Loop)
- **Genome** (example):
  - ConvLSTM: `num_layers`, `hidden_sizes`, `kernel_size`, `dropout`, `lr`
  - PGM: `type={CRF,HMM}`, `pairwise_weight`, `num_iterations`, `temporal_links`
  - Training: `batch_size`, `sequence_length`, `loss_weights (α,β,γ)`
- **EA Parameters**:
  - `population_size=24–64`, `elitism=2–4`, `tournament_k=3–5`
  - `mutation_rate=0.1–0.2`, `crossover_rate=0.6–0.8`
  - Parallel islands for diversity; migrate every `G` generations.
- **Fitness**:
  - Primary: val `F1/mIoU` (classification/segmentation) or `RMSE` (regression).
  - Secondary: calibration (ECE), runtime, memory.
- **Budgeting**:
  - Early-stopping + low-fidelity evaluations (smaller T/H/W) in early generations.
  - Promote promising configs to full-fidelity.

### Phase 5 — Calibration & Export
- Temperature scaling / isotonic regression on validation.
- Save: `model.pt`, `pgm_params.pkl`, `config.yaml`, `metrics.json`.

---

## 5) Configuration

Use a single YAML to control the pipeline:

```yaml
# config.yaml
seed: 42
device: "cuda:0"

data:
  root: "./data"
  format: "NTHWC"
  T_window: 8
  stride: 4
  normalize: true
  augmentations:
    spatial_flip: true
    rotate_deg: [0, 90, 180, 270]
    gaussian_noise_std: 0.01

model:
  convlstm:
    num_layers: 2
    hidden_sizes: [64, 64]
    kernel_size: 3
    dropout: 0.1
    bidirectional: false
  pgm:
    type: "crf"   # options: crf, hmm, chain_crf
    pairwise_weight: 0.7
    temporal_links: true
    mean_field_iters: 5

training:
  optimizer: "adamw"
  lr: 2.0e-3
  weight_decay: 1.0e-2
  batch_size: 8
  epochs: 60
  grad_clip: 1.0
  mixed_precision: true
  early_stopping_patience: 8

loss:
  alpha_unary: 1.0
  beta_structured: 0.6
  gamma_consistency: 0.2

evolutionary:
  enabled: true
  population_size: 32
  generations: 20
  elitism: 2
  tournament_k: 3
  mutation_rate: 0.15
  crossover_rate: 0.7
  low_fidelity:
    downsample_factor: 2
    max_epochs: 15
```

---

## 6) Commands & Quickstart

```bash
# 1) Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Preprocess data
python tools/preprocess.py --config config.yaml

# 3) ConvLSTM pretraining
python train_convlstm.py --config config.yaml --save runs/base_convlstm

# 4) PGM fitting (uses saved ConvLSTM logits/features)
python fit_pgm.py --config config.yaml --features runs/base_convlstm/features.pt

# 5) Joint training
python train_joint.py --config config.yaml --resume runs/base_convlstm/ckpt.pt

# 6) Evolutionary search (orchestration)
python evolve.py --config config.yaml --out runs/evo_search --parallel 8

# 7) Evaluate & export
python evaluate.py --config config.yaml --ckpt runs/evo_search/best/ckpt.pt
python export.py   --config config.yaml --ckpt runs/evo_search/best/ckpt.pt --out artifacts/
```

---

## 7) Evaluation

**Primary metrics** (choose per task):

- **Classification/Segmentation**: Accuracy, Precision/Recall, **F1**, **mIoU**, AUROC.
- **Regression**: **RMSE**, MAE, R².
- **Calibration**: **ECE**/**MCE**.
- **Structure**: Boundary F1, temporal consistency score.

**Reporting**:
- Produce `metrics.json` per run with seeds, config hash, and dataset split stats.
- Plot learning curves, confusion matrices, reliability diagrams.

---

## 8) Reproducibility

- Fix seeds: dataset shuffling + dataloader + CUDA determinism (when feasible).
- Log: config YAML, commit hash, data version, environment (CUDA, cuDNN).
- Use **artifact hashing** for checkpoints + PGM params.
- Document **random sources** (EA mutations, crossover) and store **PRNG states**.

---

## 9) Directory Structure

```
project/
├─ README.md
├─ config.yaml
├─ requirements.txt
├─ data/
│  ├─ raw/  ├─ processed/
├─ tools/
│  ├─ preprocess.py   ├─ visualize.py
├─ models/
│  ├─ convlstm.py     ├─ crf.py     ├─ hmm.py
├─ train_convlstm.py
├─ fit_pgm.py
├─ train_joint.py
├─ evolve.py
├─ evaluate.py
├─ export.py
├─ runs/
│  ├─ base_convlstm/  ├─ evo_search/
└─ artifacts/
   ├─ model.pt  ├─ pgm_params.pkl  ├─ metrics.json
```

---

## 10) Pseudocode

### Joint Training with CRF (mean-field)

```python
# High-level pseudocode (framework-agnostic)
```
FUNCTION optimize_convlstm_with_evotorch(num_generations, population_size):
    PRINT "Starting Neuroevolution with EvoTorch and CMA-ES"

    // 1. Setup Model and Data
    DEFINE model_parameters (input_size, hidden_size, num_layers)
    DEFINE data_dimensions (batch_size, sequence_length, height, width)

    CREATE model AS a new ConvLSTM with model_parameters
    CREATE sample_input AS a random tensor with data_dimensions
    CREATE target_output AS a tensor of zeros (representing a uniform distribution)

    CALCULATE num_params = total number of parameters in the model
    PRINT "Optimizing " + num_params + " parameters."

    // 2. Define the Objective Function for the optimizer
    FUNCTION objective_function(params):
        // 'params' is a flat 1D vector of model weights from the optimizer

        // Load the flat parameter vector into the model
        SET model parameters FROM params

        // Perform a forward pass
        CALCULATE model_output = model.forward(sample_input)

        // Reshape outputs for loss calculation
        RESHAPE model_output to (batch_size * sequence_length, hidden_size)
        RESHAPE target_output to (batch_size * sequence_length, hidden_size)

        // Convert model outputs to log-probabilities and targets to probabilities
        CALCULATE log_probs = log_softmax(model_output)
        CALCULATE target_probs = softmax(target_output)

        // Calculate the KL Divergence loss
        CALCULATE loss = kl_divergence(log_probs, target_probs)

        RETURN loss
    END FUNCTION

    // 3. Configure the Optimization Problem for EvoTorch
    CREATE problem with properties:
        objective_sense = "minimize"
        objective_function = objective_function
        solution_length = num_params
        initial_bounds = (-0.1, 0.1)

    // 4. Setup and Run the CMA-ES Algorithm
    CREATE searcher AS a new CMAES instance with:
        problem = problem
        initial_standard_deviation = 0.1
        population_size = population_size

    PRINT "Running optimization for " + num_generations + " generations..."
    searcher.run(num_generations)

    // 5. Get and Display Results
    GET best_params, best_loss FROM searcher status
    PRINT "Optimization Finished"
    PRINT "Best Loss (KL Divergence): " + best_loss

    RETURN best_params, best_loss
END FUNCTION

---

## 11) Troubleshooting

- **Instability during joint training**  
  Reduce `pairwise_weight` or mean-field iterations; warm up with ConvLSTM-only epochs.

- **Over-smoothing (CRF)**  
  Use **class-wise** pairwise weights; add **edge-aware** terms (e.g., guided by gradients).

- **EA stagnation**  
  Increase mutation rate, introduce island model, random restarts, or diversify genomes (kernel sizes / depths).

- **GPU OOM**  
  Lower `T_window`, `H/W` downsampling, use gradient checkpointing, mixed precision.

- **Slow PGM inference**  
  Cache features; limit graph size; consider temporal-only models (HMM/Chain CRF) if spatial grid is large.

---

## 12) Roadmap

- ✅ Baseline ConvLSTM pretraining  
- ✅ CRF/HMM integration  
- ✅ EA-based hyperparameter/architecture search  
- ⏳ Multi-objective EA (accuracy + latency)  
- ⏳ Bayesian EA hybrids (e.g., CMA-ES for fine-tuning)  
- ⏳ Distributed island model with fault tolerance

---

## Notes & Next Steps

If you share a bit about your **data type** (e.g., radar maps, videos, time-series grids) and **target labels** (classification, segmentation, regression), I can tailor the PGM choice (CRF vs HMM), the ConvLSTM shapes, and the EA search space specifically for your problem, Emmanuel.
