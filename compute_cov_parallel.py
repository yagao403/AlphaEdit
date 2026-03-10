"""
Parallel-friendly script for computing covariance and null-space projection.

Two modes:
  1. --mode compute  : Compute covariance + projection for a SINGLE layer
  2. --mode aggregate : Combine all per-layer projections into the final tensor

Usage (parallel, one job per layer):
    python compute_cov_parallel.py --mode compute --layer 6 \
        --model_name Qwen/Qwen3-32B --hparams_fname Qwen3-32B.json \
        --output_dir ./precomputed --torch_dtype bfloat16

Usage (aggregate after all layer jobs finish):
    python compute_cov_parallel.py --mode aggregate \
        --hparams_fname Qwen3-32B.json --output_dir ./precomputed
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from rome.layer_stats import layer_stats
from util.globals import *
from util.nethook import set_requires_grad


def compute_covariance(model, tok, layer_name, hparams, force_recompute=False):
    print(f"\n{'='*60}")
    print(f"Computing covariance for layer: {layer_name}")
    print(f"{'='*60}")

    stat = layer_stats(
        model,
        tok,
        layer_name,
        STATS_DIR,
        hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=force_recompute,
    )
    cov = stat.mom2.moment().float()
    print(f"  Covariance shape: {cov.shape}")
    return cov


def compute_projection(cov, threshold, gpu_device="cuda:0"):
    n = cov.shape[0]

    for attempt_device in [gpu_device, "cpu"]:
        try:
            if attempt_device != "cpu":
                torch.backends.cuda.preferred_linalg_library("magma")
            print(f"  Running eigendecomposition on ({n}x{n}) on {attempt_device}...")
            cov_dev = cov.to(attempt_device)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_dev)
            break
        except RuntimeError as e:
            print(f"  Failed on {attempt_device}: {e}")
            if attempt_device == "cpu":
                raise
            print(f"  Falling back to CPU...")
            torch.cuda.empty_cache()
            continue

    small_indices = (eigenvalues < threshold).nonzero(as_tuple=True)[0]
    print(f"  Eigenvalues below threshold ({threshold}): "
          f"{len(small_indices)} / {len(eigenvalues)}")

    U_small = eigenvectors[:, small_indices]
    if U_small.device.type == "cuda":
        P = (U_small @ U_small.T).cpu()
        del cov_dev, eigenvalues, eigenvectors, U_small
        torch.cuda.empty_cache()
    else:
        P = U_small @ U_small.T
    print(f"  Projection matrix shape: {P.shape}")
    return P


def run_compute(args, hparams):
    layer = args.layer
    if layer not in hparams.layers:
        print(f"WARNING: layer {layer} is not in hparams.layers {hparams.layers}")

    layer_name = hparams.rewrite_module_tmp.format(layer)
    print(f"Processing single layer {layer} ({layer_name})")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.torch_dtype]

    print(f"\nLoading model: {args.model_name}")
    print(f"  dtype: {model_dtype}, device_map: auto")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=model_dtype,
    )
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    set_requires_grad(False, model)

    if hasattr(model, "hf_device_map"):
        print(f"  Model distributed across devices: {set(model.hf_device_map.values())}")

    cov = compute_covariance(
        model, tok, layer_name, hparams,
        force_recompute=args.force_recompute,
    )

    del model
    torch.cuda.empty_cache()

    P = compute_projection(cov, hparams.nullspace_threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"null_space_project_layer{layer}.pt"
    torch.save(P, output_path)
    print(f"\nSaved projection for layer {layer} to {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.1f} MB")


def run_aggregate(args, hparams):
    output_dir = Path(args.output_dir)
    P_list = []

    for layer in hparams.layers:
        path = output_dir / f"null_space_project_layer{layer}.pt"
        if not path.exists():
            print(f"ERROR: Missing {path}. Run compute for layer {layer} first.")
            return
        print(f"Loading layer {layer} from {path}")
        P_list.append(torch.load(path, map_location="cpu", weights_only=True))

    P = torch.stack(P_list, dim=0)
    print(f"\nFinal projection tensor shape: {P.shape}")

    output_path = output_dir / "null_space_project.pt"
    torch.save(P, output_path)
    print(f"Saved to {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel covariance + projection for AlphaEdit"
    )
    parser.add_argument("--mode", required=True, choices=["compute", "aggregate"])
    parser.add_argument("--layer", type=int, help="Layer index (required for compute mode)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--hparams_fname", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./precomputed")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    hparams_path = HPARAMS_DIR / "AlphaEdit" / args.hparams_fname
    hparams = AlphaEditHyperParams.from_json(hparams_path)
    print(f"Loaded hyperparameters from {hparams_path}")
    print(f"  Target layers: {hparams.layers}")

    if args.mode == "compute":
        if args.layer is None:
            parser.error("--layer is required for compute mode")
        run_compute(args, hparams)
    else:
        run_aggregate(args, hparams)


if __name__ == "__main__":
    main()
