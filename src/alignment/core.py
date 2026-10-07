
from __future__ import annotations

import torch
import torch.nn.functional as F


def _encode(tokenizer, text):
    """Tokenize text without automatically adding special tokens."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
    )

    if isinstance(encoded, dict):
        ids = encoded["input_ids"]
    else:
        ids = encoded.input_ids

    # Some tokenizers may return [[...]] for one input.
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]

    return list(ids)


def _pad_token_id(tokenizer):
    """Return a usable padding token id."""
    pad_id = getattr(tokenizer, "pad_token_id", None)

    if pad_id is not None:
        return pad_id

    eos_id = getattr(tokenizer, "eos_token_id", None)

    if eos_id is not None:
        return eos_id

    return 0


def build_lm_batch(tokenizer, prompts, responses, max_length):
    """
    Build a causal-LM batch where loss is computed only on response tokens.

    Prompt tokens and padding tokens receive label -100.
    """

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = _pad_token_id(tokenizer)

    rows = []
    label_rows = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = _encode(tokenizer, prompt)
        response_ids = _encode(tokenizer, response)

        # Assignment requirement:
        # append EOS to response BEFORE sequence-length truncation.
        if eos_id is not None:
            response_ids = response_ids + [eos_id]

        input_ids = (prompt_ids + response_ids)[:max_length]

        # Prompt positions do not contribute to the LM objective.
        prompt_length = min(len(prompt_ids), len(input_ids))

        labels = [-100] * prompt_length

        # Remaining positions correspond to the response (possibly truncated).
        labels += input_ids[prompt_length:]

        rows.append(input_ids)
        label_rows.append(labels)

    # Pad to the longest sequence in this batch, capped by max_length.
    batch_length = min(
        max((len(row) for row in rows), default=0),
        max_length,
    )

    input_batch = []
    attention_batch = []
    labels_batch = []

    for input_ids, labels in zip(rows, label_rows):
        input_ids = input_ids[:batch_length]
        labels = labels[:batch_length]

        padding = batch_length - len(input_ids)

        input_batch.append(input_ids + [pad_id] * padding)
        attention_batch.append([1] * len(input_ids) + [0] * padding)
        labels_batch.append(labels + [-100] * padding)

    return {
        "input_ids": torch.tensor(input_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }


def _model_device(model):
    """Find the device on which a model's parameters live."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def response_logprobs_from_batch(model, batch):
    """
    Return the summed log probability of response tokens for every row.

    For a causal LM, logits at position t predict token t+1, so both logits
    and labels are shifted before gathering token log probabilities.
    """

    device = _model_device(model)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    # Causal shift:
    # logits[:, t] predicts input token at t + 1.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    valid_mask = shift_labels != -100

    # gather() cannot use -100 as a vocabulary index.
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    log_probs = F.log_softmax(shift_logits, dim=-1)

    token_logps = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_logps = token_logps * valid_mask

    return token_logps.sum(dim=-1)


def compute_response_logprobs(
    model,
    tokenizer,
    prompts,
    responses,
    max_length,
):
    """Build a response-masked batch and return one summed logp per row."""

    batch = build_lm_batch(
        tokenizer,
        prompts,
        responses,
        max_length,
    )

    return response_logprobs_from_batch(model, batch)


def compute_sft_loss(
    model,
    tokenizer,
    prompts,
    responses,
    max_length,
):
    """
    Compute response-only SFT loss.

    We use the assignment-allowed reduction:
    mean of the per-example summed response negative log probabilities.
    """

    response_logps = compute_response_logprobs(
        model,
        tokenizer,
        prompts,
        responses,
        max_length,
    )

    return -response_logps.mean()


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
):
    """Build separate response-masked chosen and rejected LM batches."""

    if not (
        len(prompts)
        == len(chosen_responses)
        == len(rejected_responses)
    ):
        raise ValueError(
            "prompts, chosen_responses, and rejected_responses "
            "must have the same length"
        )

    chosen_batch = build_lm_batch(
        tokenizer,
        prompts,
        chosen_responses,
        max_length,
    )

    rejected_batch = build_lm_batch(
        tokenizer,
        prompts,
        rejected_responses,
        max_length,
    )

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch,
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Compute the scalar batch-mean DPO objective."""

    policy_gap = (
        policy_chosen_logps
        - policy_rejected_logps
    )

    reference_gap = (
        reference_chosen_logps
        - reference_rejected_logps
    )

    logits = beta * (policy_gap - reference_gap)

    return -F.logsigmoid(logits).mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Compute DPO loss while keeping the reference computation frozen."""

    batch = build_dpo_batch(
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length,
    )

    policy_chosen_logps = response_logprobs_from_batch(
        policy_model,
        batch["chosen"],
    )

    policy_rejected_logps = response_logprobs_from_batch(
        policy_model,
        batch["rejected"],
    )

    # Reference model must never receive gradients.
    with torch.no_grad():
        reference_chosen_logps = response_logprobs_from_batch(
            reference_model,
            batch["chosen"],
        )

        reference_rejected_logps = response_logprobs_from_batch(
            reference_model,
            batch["rejected"],
        )

    return compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )