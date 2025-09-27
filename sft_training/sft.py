import torch


import torch

def tokenize_prompt_and_output(inputs_batch, outputs_batch, tokenizer):
    """
    Tokenizes a batch of prompts and outputs for causal LM training.

    Args:
        inputs_batch: list of lists of prompt strings (batch_size x n_prompts)
        outputs_batch: list of lists of output strings (batch_size x n_outputs)
        tokenizer: Hugging Face tokenizer

    Returns:
        dict with:
            input_ids: tensor(batch_size, max_len)
            labels: tensor(batch_size, max_len)
            response_mask: tensor(batch_size, max_len)
    """
    batch_input_ids = []
    batch_labels = []
    batch_response_masks = []

    flat_inputs = [item for sublist in inputs_batch for item in sublist]
    flat_outputs = [item for sublist in outputs_batch for item in sublist]

    for prompt, response in zip(flat_inputs, flat_outputs):
        input_tokens = tokenizer.encode(prompt)
        output_tokens = tokenizer.encode(response)
        full_tokens = input_tokens + output_tokens

        inputs_t = full_tokens[:-1]
        outputs_t = full_tokens[1:]
        response_mask = [0] * len(input_tokens) + [1] * len(output_tokens)
        response_t = response_mask[1:]

        batch_input_ids.append(inputs_t)
        batch_labels.append(outputs_t)
        batch_response_masks.append(response_t)

    max_len = max(len(seq) for seq in batch_input_ids)

    def pad(sequences, pad_value):
        return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

    input_ids = torch.tensor(pad(batch_input_ids, pad_value=tokenizer.pad_token_id))
    labels = torch.tensor(pad(batch_labels, pad_value=-100))
    response_mask = torch.tensor(pad(batch_response_masks, pad_value=0))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }




def compute_entropy(x):
    log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs)
    entropy = torch.sum(entropy,dim=-1)
    return entropy



def get_response_log_probs(model, inputs, labels, entropy=False):
    model.to('cpu')
    inputs = inputs.to('cpu')
    responses = model(input_ids=inputs).logits
    labels = labels.to('cpu')
    log_probs_full = torch.nn.functional.log_softmax(responses, dim=-1)
    batch_size, seq_len = labels.shape
    log_probs_correct = log_probs_full.gather(dim=-1, index=labels.unsqueeze(-1))
    log_probs_correct = log_probs_correct.squeeze(-1)
    result = {"log_probs": log_probs_correct}
    if entropy:
        token_entropy = compute_entropy(log_probs_full)
        result["token_entropy"] = token_entropy
    return result


def masked_normalize(tensor,mask,normalize_constant,dim):
    mask = mask.float()
    masked_tensor = tensor * mask
    summed = masked_tensor.sum(dim=dim)
    normalized = summed / normalize_constant
    return normalized


def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant):
    nll = -policy_log_probs * response_mask
    loss = masked_normalize(-policy_log_probs, response_mask, normalize_constant, dim=-1)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "nll_sum": nll.sum().detach(),
        "tokens": response_mask.sum().detach(),
        "mean_loss_per_token": masked_normalize(nll, response_mask, response_mask.sum() + 1e-8, dim=None).detach()
    }
    return loss.detach(), metadata



def log_generations(model,tokenizer,prompts: list[str],ground_truths: list[str],reward_fn,device="cuda"):
    model.eval()
    generations_log = []
    for prompt, gt in zip(prompts, ground_truths):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        
        response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        reward = reward_fn(response, gt)   
        with torch.no_grad():
            logits = model(input_ids).logits
            token_entropy = compute_entropy(logits)  
            avg_entropy = token_entropy.mean().item()
        resp_len = len(tokenizer.encode(response))
        generations_log.append({
            "prompt": prompt,
            "ground_truth": gt,
            "response": response,
            "reward": reward,
            "avg_token_entropy": avg_entropy,
            "response_length": resp_len
        })
    all_lengths = [g["response_length"] for g in generations_log]
    correct_lengths = [g["response_length"] for g in generations_log if g["reward"]["answer"] == 1]
    incorrect_lengths = [g["response_length"] for g in generations_log if g["reward"]["answer"] == 0]
    summary = {
        "avg_response_length": sum(all_lengths) / len(all_lengths),
        "avg_length_correct": sum(correct_lengths) / max(len(correct_lengths), 1),
        "avg_length_incorrect": sum(incorrect_lengths) / max(len(incorrect_lengths), 1),
    }
    return generations_log, summary

