from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

dataset_path = "/Users/mithunnayak/Desktop/WORK/ALIGNMENT/assignment5-alignment/math_dataset"
model_path = "/Users/mithunnayak/Desktop/WORK/ALIGNMENT/assignment5-alignment/qwen2.5_1.5_instruct"

prompt = '''A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> just provide the answer here </answer>.
User: {question}
Assistant: <think>'''

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("mps")  

import json
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token, tokenizer):
        self.stop_token = stop_token
        self.stop_ids = tokenizer.encode(stop_token, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids

stop_token = "</answer>"
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token, tokenizer)])


from datasets import load_from_disk
from mygrader import *
dataset = load_from_disk(dataset_path)
small_dataset = dataset.select(range(100))
num_of_data = 100

score_list = []

for i in range(num_of_data):
    question = dataset['train'][i]['problem']
    final_prompt = prompt.format(question=question)
    inputs = tokenizer(final_prompt, return_tensors="pt").to("mps")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        stopping_criteria=stopping_criteria,
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gt = dataset['train'][i]['solution']
    gt = extract_answer(gt)
    score = score_model(output, gt, fast=True, tolerant=True, verbose=True)
    score_list.append(score)


    # after processing all outputs/gts
    with open("all_scores.json", "w") as f:
        json.dump(score_list, f, indent=4)

    if score['format_reward'] == 1 and score['answer_reward'] == 0 and score['reward'] == 0:
        with open("case1_format_ok_wrong_answer.txt", "a") as f:
            f.write(f"GT: {dataset['train'][i]['solution']}\nOUT: {output}\n\n")

    # Case 2: format wrong but answer right
    if score['format_reward'] == 0 and score['answer_reward'] == 1 and score['reward'] == 0:
        with open("case2_format_wrong_answer_right.txt", "a") as f:
            f.write(f"GT: {dataset['train'][i]['solution']}\nOUT: {output}\n\n")

    # Case 3: any format failure
    if score['reward'] == 1:
        with open("case3_all_good.txt", "a") as f:
            f.write(f"GT: {dataset['train'][i]['solution']}\nOUT: {output}\n\n")