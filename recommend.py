import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

GPU_CATALOG = {
    "T4":      {"vram_gb": 16, "price_od": 0.35, "price_spot": 0.11},
    "A10G":    {"vram_gb": 24, "price_od": 0.75, "price_spot": 0.23},
    "V100":    {"vram_gb": 16, "price_od": 2.48, "price_spot": 0.74},
    "A100-40": {"vram_gb": 40, "price_od": 3.21, "price_spot": 0.96},
    "A100-80": {"vram_gb": 80, "price_od": 4.10, "price_spot": 1.23},
    "K80":     {"vram_gb": 12, "price_od": 0.20, "price_spot": 0.06},
    "P100":    {"vram_gb": 16, "price_od": 1.46, "price_spot": 0.44},
}

BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "bf16": 2}
GPU_ID_MAP = {g: i for i, g in enumerate(sorted(GPU_CATALOG.keys()))}
PREC_MAP = {"fp32": 0, "bf16": 1, "fp16": 2}

def estimate_vram_gb(model_params, precision):
    """Estimate minimum VRAM needed to fit model (params in absolute count)."""
    return (model_params * BYTES_PER_PARAM[precision]) / 1e9

def build_feature_row(gpu_name, model_params, batch_size, steps, precision, vram_gb):
    """Build a single feature vector matching the training schema."""
    compute_load = (model_params * steps) / batch_size
    log_compute_load = np.log1p(compute_load)
    gpu_id = GPU_ID_MAP[gpu_name]
    precision_id = PREC_MAP[precision]
    return [gpu_id, log_compute_load, vram_gb, precision_id, batch_size]

def recommend(model_params_M, batch_size, steps, precision,
              deadline_hr=None, pricing="both"):
    """Predict runtime and rank GPUs by total cost for a given workload."""
    model = joblib.load("runtime_model.pkl")
    model_params = model_params_M * 1e6
    vram_needed = estimate_vram_gb(model_params, precision)

    rows, raw_features = [], []
    for gpu_name, specs in GPU_CATALOG.items():
        if vram_needed > specs["vram_gb"]:
            rows.append({"GPU": gpu_name, "VRAM": f"{specs['vram_gb']}GB",
                         "Pred Runtime": "OOM", "$/hr (OD)": specs["price_od"],
                         "Est. Cost (OD)": "—", "Feasible": "✗"})
            continue
        feat = build_feature_row(gpu_name, model_params, batch_size,
                                 steps, precision, specs["vram_gb"])
        raw_features.append(feat)
        rows.append({"GPU": gpu_name, "specs": specs, "_feat_idx": len(raw_features) - 1})

    if raw_features:
        scaler = joblib.load("scaler.pkl")
        feat_cols = ["gpu_id", "log_compute_load", "vram_gb", "precision_id", "batch_size"]
        X_pred = scaler.transform(pd.DataFrame(raw_features, columns=feat_cols))
        pred_log = model.predict(X_pred)
        pred_sec = np.expm1(pred_log)

    results = []
    for row in rows:
        if "specs" not in row:
            results.append(row)
            continue
        specs = row["specs"]
        runtime_sec = pred_sec[row["_feat_idx"]]
        runtime_hr = runtime_sec / 3600

        feasible = True
        if deadline_hr and runtime_hr > deadline_hr:
            feasible = False

        cost_od = runtime_hr * specs["price_od"]
        cost_spot = runtime_hr * specs["price_spot"]

        results.append({
            "GPU":            row["GPU"],
            "VRAM":           f"{specs['vram_gb']}GB",
            "Pred Runtime":   f"{runtime_hr:.2f} hr",
            "$/hr (OD)":      f"${specs['price_od']:.2f}",
            "$/hr (Spot)":    f"${specs['price_spot']:.2f}",
            "Est. Cost (OD)": f"${cost_od:.2f}",
            "Est. Cost (Spot)":f"${cost_spot:.2f}",
            "Feasible":       "✓" if feasible else f"✗ (>{deadline_hr}hr)",
            "_cost_od":       cost_od if feasible else float("inf"),
            "_cost_spot":     cost_spot if feasible else float("inf"),
        })

    results.sort(key=lambda r: r.get("_cost_od", float("inf")))

    return results

def print_table(results):
    cols = ["GPU", "VRAM", "Pred Runtime", "$/hr (OD)", "Est. Cost (OD)",
            "$/hr (Spot)", "Est. Cost (Spot)", "Feasible"]
    widths = {c: max(len(c), max(len(str(r.get(c, "—"))) for r in results))
              for c in cols}
    header = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    print("\n" + header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in results:
        row_str = "  ".join(f"{str(r.get(c,'—')):<{widths[c]}}" for c in cols)
        if r.get("_cost_od", float("inf")) == min(
                x.get("_cost_od", float("inf")) for x in results):
            row_str += "  ★ RECOMMENDED"
        print(row_str)
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Recommend optimal cloud GPU for ML training workload"
    )
    parser.add_argument("--model-params", type=float, required=True,
                        help="Model parameters in millions (e.g. 125 for GPT-2)")
    parser.add_argument("--batch-size",   type=int,   required=True)
    parser.add_argument("--steps",        type=int,   required=True)
    parser.add_argument("--precision",    choices=["fp32","fp16","bf16"], default="fp16")
    parser.add_argument("--deadline",     type=float, default=None,
                        help="Max allowed runtime in hours")
    args = parser.parse_args()

    print(f"\nWorkload: {args.model_params}M params | "
          f"batch={args.batch_size} | steps={args.steps:,} | "
          f"precision={args.precision}"
          + (f" | deadline={args.deadline}hr" if args.deadline else ""))
    print(f"Est. VRAM needed: "
          f"{estimate_vram_gb(args.model_params*1e6, args.precision):.1f} GB")

    results = recommend(args.model_params, args.batch_size, args.steps,
                        args.precision, args.deadline)
    print_table(results)

if __name__ == "__main__":
    main()
