import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams
# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def wrap_prompt_with_chat_template(tok, prompt_template: str) -> str:
    """
    Wraps a prompt template (still containing {} for subject substitution)
    with the tokenizer's chat template and appends empty thinking tags
    for non-thinking mode.
    """
    placeholder = "ALPHAEDIT_SUBJECT_PLACEHOLDER"
    filled = prompt_template.replace("{}", placeholder)
    messages = [{"role": "user", "content": filled}]
    templated = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = templated.replace(placeholder, "{}")
    result += "<think>\n\n</think>\n\n"
    return result

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
    use_chat_template: bool = False,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer.
    Supports multi-GPU models loaded with device_map="auto".
    """

    # Update target and print info
    requests = deepcopy(requests)
    if not use_chat_template:
        for i, request in enumerate(requests):
            if request["target_new"]["str"][0] != " ":
                requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    if use_chat_template:
        for i, request in enumerate(requests):
            requests[i]["prompt"] = wrap_prompt_with_chat_template(
                tok, request["prompt"]
            )

    for request in requests[:10]:
        print(
            f"AlphaEdit request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Determine a device for heavy linear algebra (linalg.solve, matmuls)
    compute_device = torch.device("cuda:0")

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    if use_chat_template:
        context_templates = [["{}"]]
    else:
        context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    if use_chat_template and cache_template is not None:
        print(
            "WARNING: use_chat_template is ON with caching enabled. "
            "Cached v* values computed without chat template will be stale. "
            "Consider clearing the cache or disabling --use_cache."
        )

    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None
            and cache_fname.exists()
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(compute_device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                use_chat_template=use_chat_template,
            )

            z_list.append(cur_z.to(compute_device))

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        # Move to common compute device for arithmetic
        cur_zs = cur_zs.to(compute_device)
        zs = zs.to(compute_device)
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        # Move layer_ks to compute device and cast to float32 for numerical stability
        layer_ks = layer_ks.float().to(compute_device)

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets.float() / (len(hparams.layers) - i)

        P_i = P[i,:,:].float().to(compute_device)
        cache_c_i = cache_c[i,:,:].float().to(compute_device)
        lhs = P_i @ (layer_ks @ layer_ks.T + cache_c_i) + hparams.L2 * torch.eye(
            layer_ks.shape[0], dtype=torch.float, device=compute_device
        )
        rhs = P_i @ layer_ks @ resid.T
        upd_matrix = torch.linalg.solve(lhs, rhs)

        # Adjust update matrix shape and move to weight's device/dtype
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        upd_matrix = upd_matrix.to(weights[weight_name].dtype).to(weights[weight_name].device)

        print("orig norm", torch.linalg.norm(weights[weight_name].float()))
        print("upd norm", torch.linalg.norm(upd_matrix.float()))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        del layer_ks, cur_zs, targets, upd_matrix, P_i, cache_c_i, lhs, rhs
        torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        layer_ks_cpu = layer_ks.cpu().float()
        cache_c[i,:,:] += layer_ks_cpu @ layer_ks_cpu.T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    device = torch.device("cuda:0")
    return (
        torch.inverse(COV_CACHE[key].to(device)) if inv else COV_CACHE[key].to(device)
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
