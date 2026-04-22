# Cloud GPU Recommendation System for ML Training

A decision-support system that recommends the optimal cloud GPU instance for a 
given ML training workload, minimizing total cost while satisfying VRAM and 
deadline constraints.

## Project Structure

| File | Description |
|---|---|
| `data_generator.py` | Generates a simulation-based benchmark dataset grounded in published GPU throughput specifications |
| `train_and_evaluate.py` | Trains a GradientBoostingRegressor on the benchmark dataset and evaluates MAPE |
| `recommend.py` | CLI tool that takes workload parameters and returns a ranked GPU recommendation table |
| `benchmark_data.csv` | Generated benchmark dataset (4,309 rows across 7 GPU types) |
| `runtime_model.pkl` | Trained model saved to disk |
| `scaler.pkl` | Fitted StandardScaler saved to disk |

## Dependencies

Python 3.8 or higher. Install all dependencies with:

```bash
pip install numpy pandas scikit-learn joblib
```

## How to Run

### Step 1 — Generate the dataset
```bash
python3 data_generator.py
```
This generates `benchmark_data.csv` with 4,309 benchmark rows.

### Step 2 — Train the model
```bash
python3 train_and_evaluate.py
```
This trains the GradientBoostingRegressor and saves `runtime_model.pkl` 
and `scaler.pkl` to disk. It also prints test MAPE and cross-validation results.

### Step 3 — Run the recommendation CLI
```bash
python3 recommend.py --model-params 125 --batch-size 32 --steps 10000 --precision fp16 --deadline 6
```

#### CLI Arguments

| Argument | Description | Example |
|---|---|---|
| `--model-params` | Model size in millions of parameters | `125` for GPT-2 |
| `--batch-size` | Training batch size | `32` |
| `--steps` | Number of training steps | `10000` |
| `--precision` | Numerical precision: `fp32`, `fp16`, or `bf16` | `fp16` |
| `--deadline` | Maximum allowed runtime in hours (optional) | `6` |

#### Example output

Workload: 125M params | batch=32 | steps=10,000 | precision=fp16 | deadline=6.0hr
Est. VRAM needed: 0.2 GB
GPU      Pred Runtime  $/hr(OD)  Est. Cost(OD)  Feasible

A10G     6.11 hr       $0.75     $4.59          YES  * RECOMMENDED
A100-40  3.24 hr       $3.21     $10.40         YES
V100     9.10 hr       $2.48     $22.57         YES
T4       18.94 hr      $0.35     $6.63          NO (>6hr)
K80      30.00 hr      $0.20     $6.00          NO (>6hr)
P100     14.92 hr      $1.46     $21.79         NO (>6hr)




## Code Authorship

| File | Status |
|---|---|
| `data_generator.py` | Written by the authors with LLM assistance in understanding concepts(Claude, Anthropic) |
| `train_and_evaluate.py` | Written by the authors with LLM assistance in understanding concepts(Claude, Anthropic) |
| `recommend.py` | Written by the authors with LLM assistance in understanding concetpts (Claude, Anthropic) |



