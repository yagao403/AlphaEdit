"""
Standalone script to compute the covariance matrix (K0 K0^T) and
null-space projection matrix P for AlphaEdit.

This should be run BEFORE the main editing experiment, especially
for large models like Qwen3-32B that require multi-GPU.

Usage:
    python compute_cov_and_proj.py \
        --model_name Qwen/Qwen3-32B \
        --hparams_fname Qwen3-32B.json \
        --output_dir ./precomputed

The script will:
  1. Load the model across multiple GPUs via device_map="auto"
  2. Compute the second-moment covariance statistics for each target layer
     (cached automatically under data/stats/)
  3. SVD the covariance to build the null-space projection matrix P
  4. Save P to <output_dir>/null_space_project.pt
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from rome.layer_stats import layer_stats
from util.globals import *
from util.nethook import set_requires_grad


def compute_covariance(model, tok, layer_name, hparams, force_recompute=False):
    """Compute the second-moment covariance for a single layer."""
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
    """
    Compute the null-space projection matrix P from the covariance.

    Steps (from AlphaEdit paper, Section 3.2):
      1. Eigendecomposition of the symmetric PSD covariance K0 K0^T
      2. Keep eigenvectors whose eigenvalues are below the threshold
      3. P = U_small @ U_small^T
    """
    n = cov.shape[0]

    # Try GPU with magma backend first, fall back to CPU
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


def main():
    parser = argparse.ArgumentParser(
        description="Precompute covariance and null-space projection for AlphaEdit"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or local path (e.g. Qwen/Qwen3-32B)",
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        required=True,
        help="Hyperparameters JSON filename under hparams/AlphaEdit/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./precomputed",
        help="Directory to save the projection matrix P",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation even if cached stats exist",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading dtype (float16 recommended for AMD MI250X)",
    )
    args = parser.parse_args()

    # Load hyperparameters
    hparams_path = HPARAMS_DIR / "AlphaEdit" / args.hparams_fname
    hparams = AlphaEditHyperParams.from_json(hparams_path)
    print(f"Loaded hyperparameters from {hparams_path}")
    print(f"  Target layers: {hparams.layers}")
    print(f"  Rewrite module template: {hparams.rewrite_module_tmp}")
    print(f"  Null-space threshold: {hparams.nullspace_threshold}")
    print(f"  mom2_n_samples: {hparams.mom2_n_samples}")

    # Load model with multi-GPU support
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.torch_dtype]

    print(f"\nLoading model: {args.model_name}")
    print(f"  dtype: {model_dtype}")
    print(f"  device_map: auto")

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

    # Compute covariance and projection for each layer
    num_layers = len(hparams.layers)
    P_list = []

    for i, layer in enumerate(hparams.layers):
        layer_name = hparams.rewrite_module_tmp.format(layer)
        print(f"\n[{i+1}/{num_layers}] Processing layer {layer} ({layer_name})")

        cov = compute_covariance(
            model, tok, layer_name, hparams,
            force_recompute=args.force_recompute,
        )

        P_layer = compute_projection(cov, hparams.nullspace_threshold)
        P_list.append(P_layer)

        del cov
        torch.cuda.empty_cache()

    # Stack into [num_layers, dim, dim] tensor
    P = torch.stack(P_list, dim=0)
    print(f"\nFinal projection tensor shape: {P.shape}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "null_space_project.pt"
    torch.save(P, output_path)
    print(f"Saved projection matrix to {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")

    print("\nDone! You can now use this projection matrix in the editing experiment.")
    print(f"  Set the projection_file path in experiments/evaluate.py to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
