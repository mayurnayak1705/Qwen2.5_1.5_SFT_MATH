import json

with open('/Users/mithunnayak/Desktop/WORK/ALIGNMENT/assignment5-alignment/SCORES/ZERO_SHOT/all_scores.json', 'r') as f:
    scores = json.load(f)

total_format_reward_score = 0
total_answer_reward_score = 0
total_both_correct_score = 0
for i in range(100):
    total_format_reward_score+=scores[i]['format_reward']
    total_answer_reward_score+=scores[i]['answer_reward']
    total_both_correct_score+=scores[i]['reward']


print(f"score for proper format out of 100 is {total_format_reward_score}")
print(f"score for proper answer out of 100 is {total_answer_reward_score}")
print(f"score for proper format and answer out of 100 is {total_both_correct_score}")