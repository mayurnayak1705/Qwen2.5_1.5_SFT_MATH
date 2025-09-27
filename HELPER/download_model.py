from transformers import AutoModel, AutoTokenizer


model_name = "Qwen/Qwen2.5-1.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


save_path = "./qwen2.5_1.5_instruct"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
