"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def _model_adds_bos(model, tok):
    """Dynamically check if the tokenizer actually prepends a BOS token."""
    test_ids = tok("test")["input_ids"]
    return (
        tok.bos_token_id is not None
        and len(test_ids) > 0
        and test_ids[0] == tok.bos_token_id
    )


def compute_rewrite_quality_mquake(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    has_bos = _model_adds_bos(model, tok)

    rewrite_prompts = record["paraphrase_prompts"]
    target_new = record["new_answer"]
    target_true = record["answer"]
    prob_prompts = [
        rewrite_prompts,
    ]
    target_tok = tok(" " + target_new)["input_ids"]
    if has_bos:
        target_tok = target_tok[1:]
    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = [
        el + tok.decode(target_tok[:i]) if not has_bos or i == 0 else el + ' ' + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    probs = stuff_probs

    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts"
            ]
        )
    }

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    input_device = next(model.parameters()).device
    has_bos = _model_adds_bos(model, tok)

    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(input_device)

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1).to(logits.device)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt")["input_ids"].to(logits.device)
        if has_bos:
            correct_id = correct_id[:, 1].squeeze()
        else:
            correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()
