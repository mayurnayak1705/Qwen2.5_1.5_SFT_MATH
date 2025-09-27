from sft import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from datasets import load_from_disk
from torch.optim import Adam


dataset_path = "/Users/mithunnayak/Desktop/WORK/ALIGNMENT/assignment5-alignment/math_dataset"
model_path = "/Users/mithunnayak/Desktop/WORK/ALIGNMENT/assignment5-alignment/qwen2.5_1.5_instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,   
    device_map="cuda"          
)

dataset = load_from_disk(dataset_path)
sft_train_data = dataset['train'][101:5101]

input_batch = []
output_batch = []
batch_size = 1
for i in range(0, len(sft_train_data['problem']), batch_size):
    input_batch.append(sft_train_data['problem'][i : i + batch_size])
    output_batch.append(sft_train_data['solution'][i: i+ batch_size])


optimizer = Adam(model.parameters(), lr=0.01)

def train_sft(model, tokenizer, input_batch, output_batch, optimizer, gradient_accumulation_steps=8, normalize_constant=1.0):
    device = "cuda" if torch.backends.mps.is_available() else "cpu"
    model.train()
    data_dict = tokenize_prompt_and_output(input_batch, output_batch, tokenizer)
    input_ids = data_dict['input_ids'].to(device)
    labels = data_dict['labels'].to(device)
    response_mask = data_dict['response_mask'].to(device)
    total_examples = input_ids.size(0)
    optimizer.zero_grad()

    for start in range(0, total_examples, gradient_accumulation_steps):
        end = start + gradient_accumulation_steps
        input_ids_batch = input_ids[start:end]
        labels_batch = labels[start:end]
        response_mask_batch = response_mask[start:end]
        log_probs = get_response_log_probs(model, input_ids_batch, labels_batch, entropy=False)
        loss, metadata = sft_microbatch_train_step(
            log_probs, response_mask_batch, gradient_accumulation_steps, normalize_constant
        )
        
        print(f"Loss: {loss.item()}")

    optimizer.step()
    optimizer.zero_grad()


train_sft(model, tokenizer, input_batch, output_batch, optimizer)

