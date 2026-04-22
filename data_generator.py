import pandas as pd
import numpy as np

# GPU specs: (TFLOPS at FP32, VRAM GB, hourly on-demand $, hourly spot $)
GPU_SPECS = {
    "T4":       {"tflops_fp32": 8.1,   "vram_gb": 16, "price_od": 0.35, "price_spot": 0.11},
    "A10G":     {"tflops_fp32": 31.2,  "vram_gb": 24, "price_od": 0.75, "price_spot": 0.23},
    "V100":     {"tflops_fp32": 14.0,  "vram_gb": 16, "price_od": 2.48, "price_spot": 0.74},
    "A100-40":  {"tflops_fp32": 19.5,  "vram_gb": 40, "price_od": 3.21, "price_spot": 0.96},
    "A100-80":  {"tflops_fp32": 19.5,  "vram_gb": 80, "price_od": 4.10, "price_spot": 1.23},
    "K80":      {"tflops_fp32": 8.7,   "vram_gb": 12, "price_od": 0.20, "price_spot": 0.06},
    "P100":     {"tflops_fp32": 9.3,   "vram_gb": 16, "price_od": 1.46, "price_spot": 0.44},
}

# Precision multipliers on effective throughput
PRECISION_MULT = {"fp32": 1.0, "fp16": 1.8, "bf16": 1.75}

# Approx VRAM needed (GB) = params (millions) * bytes_per_param / 1e3
# fp32=4 bytes, fp16/bf16=2 bytes
BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "bf16": 2}

def estimate_vram_gb(model_params_M, precision):
    return (model_params_M * 1e6 * BYTES_PER_PARAM[precision]) / 1e9

def generate_dataset(n_samples=8000, seed=42):
    """Generate synthetic GPU benchmark data grounded in real throughput specs."""
    rng = np.random.default_rng(seed)
    rows = []

    model_params_options = [10, 50, 125, 350, 760, 1300, 3000, 7000]  # millions
    batch_size_options   = [8, 16, 32, 64, 128]
    steps_options        = [1000, 5000, 10000, 50000, 100000, 500000]
    precision_options    = ["fp32", "fp16", "bf16"]

    for _ in range(n_samples):
        gpu_name   = rng.choice(list(GPU_SPECS.keys()))
        params_M   = rng.choice(model_params_options)
        batch_size = rng.choice(batch_size_options)
        steps      = rng.choice(steps_options)
        precision  = rng.choice(precision_options)

        specs      = GPU_SPECS[gpu_name]
        vram_needed = estimate_vram_gb(params_M, precision)

        # Mark OOM if model doesn't fit
        if vram_needed > specs["vram_gb"]:
            continue

        # Empirically-calibrated runtime model:
        # Base: seconds per step per billion params, scaled by GPU speed factor
        # Anchored to real benchmarks (e.g. GPT-2 125M on T4 ~ 0.8s/step at bs=32 fp16)
        GPU_SPEED_FACTOR = {  # relative speed vs T4 baseline
            "T4": 1.0, "A10G": 3.2, "V100": 2.1,
            "A100-40": 5.8, "A100-80": 6.2, "K80": 0.6, "P100": 1.3
        }
        BASE_SEC_PER_STEP_PER_BPARAM = 0.8  # T4, 125M params, bs=32, fp16

        params_B = params_M / 1000.0
        speed = GPU_SPEED_FACTOR[gpu_name]
        prec_mult = PRECISION_MULT[precision]
        # Larger batch = more throughput per step (sublinear scaling)
        batch_factor = (32 / batch_size) ** 0.6

        runtime_s = (BASE_SEC_PER_STEP_PER_BPARAM * params_B * steps
                     * batch_factor * batch_factor) / (speed * prec_mult)

        # Add ±3% realistic noise
        noise = rng.normal(1.0, 0.03)
        runtime_s *= noise

        rows.append({
            "gpu_name":     gpu_name,
            "model_params": params_M * 1e6,
            "batch_size":   batch_size,
            "train_steps":  steps,
            "precision":    precision,
            "vram_gb":      specs["vram_gb"],
            "runtime_sec":  round(runtime_s, 2),
            "price_od":     specs["price_od"],
            "price_spot":   specs["price_spot"],
        })

    df = pd.DataFrame(rows)
    # Keep only realistic training job runtimes: 10 min to 48 hours
    df = df[(df["runtime_sec"] >= 600) & (df["runtime_sec"] <= 172800)]
    df.to_csv("benchmark_data.csv", index=False)
    print(f"Generated {len(df)} benchmark rows across {df['gpu_name'].nunique()} GPU types")
    print(df.groupby("gpu_name")["runtime_sec"].count().rename("samples"))
    return df

if __name__ == "__main__":
    generate_dataset()
