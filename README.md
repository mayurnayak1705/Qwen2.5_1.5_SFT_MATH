Supervised Finetuning -  Qwen/Qwen2.5-1.5B-Instruct on MATH dataset



Before we perform the SFT on any model we need to analyse the base model score on that task, by performing zero shot prompt to that base model. I used qwedsacf/competition_math dataset for this task where i prompted the base model and evaluated based on custom reward function (Stanford cs336/alignment)sft_training/mygrader.py. This reward function gives scores based on models correct answer to math problem and the proper usage of tags <think></think> and <answer></answer>. The base model was able to give answer for 49/100 questions from the dataset. (folder name: ZERO_SHOT_SCORES)



Once the base line is set we can do the supervised FT on base model. From the competition_math dataset, 1,000 examples were sampled for fine-tuning.and trained just for 1 epoch on H100 gpu.(JarvisLabs.ai)

The model was trained in bfloat16 precision with FlashAttention-2 for faster training and reduced memory usage. I used gradient accumulation=8 so that the mini batch could fit into the memory. Got a final loss of 11.29. 



All the code written from scratch is available in the repo.

The scores of SFT of the same 100 examples are in (SFT_SCORES), with improvement to 51/100 (not much :))



sft_training contains the grader function, sft.py-> main components for SFT and sft_training.py -> training loop.

model (hf): heisenberg-goddamnright/qwen2.5_math_sft

github link:https://github.com/mayurnayak1705/Qwen2.5_1.5_SFT_MATH/
