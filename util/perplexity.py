import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def perplexity(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    max_input_length: int = None,
):
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """
    input_device = next(model.parameters()).device

    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to(input_device)

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    input_ids_on_logits_device = inputs["input_ids"].to(logits.device)
    log_probs = torch.gather(logits[:, :-1, :], 2, input_ids_on_logits_device[:, 1:, None])[0]

    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()
