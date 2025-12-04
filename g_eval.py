import os
from openai import OpenAI
import json
from tqdm import tqdm
from copy import deepcopy
import re
from collections import Counter, defaultdict
import argparse 

def construct_messages(n, posts, distribution, n_info, lastest_PL, reasoning, golden_label=None, method = None, eval_type=None):
    if method == "+PL":
        before = "predicted SI labels"
    elif method == "+POST":
        before = "post texts"
    
    if method == "Distr":
        given = "The distribution of predicted SI labels from previous posts"
        info = f"""The distribution of predicted SI labels from previous posts:
{distribution}"""
    else:
        given = f"The distribution of predicted SI labels from previous posts\n- The {before} for the most recent {n} posts"
        info = f"""The distribution of predicted SI labels from previous posts:
{distribution}

The {before} for the most recent {n} posts:
{n_info}"""
    
    golden = ""
    golden_label=""
    
    if eval_type=="Coherence":    
        goal = "Your goal is to evaluate how well the reasoning explains the label prediction by considering both the content of the latest post and the prior label context."
        eval =  f"""Evaluation Criterion:
Coherence: Does the reasoning clearly and logically connect the latest post with the context from prior posts to support the predicted SI label?
- 3: The reasoning is clear, coherent, and logically integrates both the latest post and prior context with no gaps.
- 2: The reasoning mostly makes sense and connects the components, but includes minor gaps or vague parts.
- 1: The reasoning is illogical, disconnected, or unclear; it fails to tie the prediction to post content or context.

Evaluation Steps:
1. Read all the posts written by the user.
2. Read the predicted SI label and the reasoning.
3. Assess how clearly and logically the reasoning connects the content of the latest post and the prior context to the predicted label.
4. Assign a score of 1, 2, or 3."""   
    elif eval_type=="Information Appropriateness":
        goal = "Your goal is to evaluate how well the reasoning explains the label prediction by considering both the content of the latest post and the prior label context."
        eval = f"""Evaluation Criterion:

Information Appropriateness: Does the reasoning appropriately utilize the provided prior information (label distribution and recent labels or post texts) to justify the predicted label?

- 3: The reasoning draws appropriately from both the latest post and the provided prior context (label distributions and/or prior labels/texts), showing sufficient justification for the predicted label.
- 2: The reasoning makes partial use of the provided information, but may miss or underuse important elements from the prior context.
- 1: The reasoning does not make meaningful use of the provided prior information and appears weak or unsupported in justifying the predicted label.

Evaluation Steps:

1. Read all the posts written by the user.
2. Read the predicted SI label and the reasoning.
3. Evaluate whether the reasoning makes adequate use of the provided prior label distribution and recent labels or texts to justify the prediction.
4. Assign a score of 1, 2, or 3."""
    elif eval_type=="Helpfulness":
        if method == "Distr":
            goal = f"Your goal is to evaluate whether the provided prior information — the distribution of predicted SI labels from previous posts — contributed appropriately to correctly inferring the SI label for the latest post."
        else:
            goal = f"Your goal is to evaluate whether the provided prior information — the distribution of predicted SI labels from previous posts and the {before} for the most recent {n} posts — contributed appropriately to correctly inferring the SI label for the latest post."
        golden = "- The gold (ground truth) SI label for the user"
        golden_label = f":\n{golden_label}" if golden_label else ""
        eval= f"""Evaluation Criterion:
Helpfulness: Did the prior information actually help the model produce a more accurate and well-justified prediction?
- 3: The prior information contributed clearly to a correct and well-justified prediction.
- 2: The prediction is close (within one severity level) and prior information is mentioned, but the use is vague, partial, or weakly justified.
- 1: The prediction is incorrect by more than one severity level or the reasoning fails to meaningfully incorporate prior information.

Evaluation Steps:
1. Read all the posts written by the user.
2. Read the predicted SI label and the reasoning.
3. Evaluate whether the reasoning meaningfully uses the prior label distribution and recent labels (or texts).
4. Assign a score of 1, 2, or 3."""
        
    system_prompt = f"""You are evaluating the predicted severity of suicidal ideation (SI) for a user based on a sequence of social media posts written by the user (in chronological order) and the reasoning.
{goal}

You will be given:
- The latest post written by a user
- {given}
- The predicted SI label for the latest post
- The generated reasoning behind the predicted SI label
{golden}
SI Severity Labels (in increasing order of severity):
(A) Supportive
(B) Indicator
(C) Ideation
(D) Behavior
(E) Attempt

{eval}

Please answer using the following format strictly:
Analysis: [your brief analysis]
Rating: [1|2|3]"""

    user_prompt = f"""
User-written posts:
{posts}

{info}

Predicted suicidal ideation severity: 
{lastest_PL}

Reasoning about the predicted suicidal ideation severity:
{reasoning}

{golden}{golden_label}"""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

if __name__ == '__main__':

    api_key = "please write your api key."
    model_type = "gpt-4o-mini-2024-07-18"
    
    cli_parser = argparse.ArgumentParser()
    # Directory
    cli_parser.add_argument("--data_file", type=str, default='./data/CSSRS_suicide_testing.jsonl')
    cli_parser.add_argument("--method", type=str, default='Distr') # 'Distr', "+PL", '+POST'
    cli_parser.add_argument("--n", type=int, default=3)
    cli_parser.add_argument("--model", type=str, default='Llama-2-7b-chat-hf')
    cli_parser.add_argument("--eval_type", type=str, default='Coherence') #'Coherence', 'Information_Appropriateness', 'Helpfulness'
    args = cli_parser.parse_args()
 
    if args.method == "+PL":
        if args.model == "Llama-2-7b-chat-hf": data_file = "Llama-2-7b-chat-hf/output/bs/PreviousStepNum/ZeroShot_4_PreviousStepNum3_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "MentaLLaMA-chat-7B-hf": data_file = "MentaLLaMA-chat-7B-hf/output/bs/PreviousStepNum/ZeroShot_4_PreviousStepNum3_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "Mistral-7B-Instruct-v0.1": data_file = "Mistral-7B-Instruct-v0.1/output/bs1/ZeroShot_PreviousStepNum3_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "Qwen2-7B-Instruct": data_file = "Qwen2-7B-Instruct/output/bs1/ZeroShot_PreviousStepNum3_testing.jsonl".replace("ZeroShot", "separate")
    elif args.method == "+POST":
        if args.model == "Llama-2-7b-chat-hf": data_file = "Llama-2-7b-chat-hf/output/bs/BeforePost/ZeroShot_4_BeforePost3_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "MentaLLaMA-chat-7B-hf": data_file = "MentaLLaMA-chat-7B-hf/output/bs/BeforePost/ZeroShot_4_BeforePost3_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "Mistral-7B-Instruct-v0.1": data_file = "Mistral-7B-Instruct-v0.1/output/bs1/ZeroShot_BeforePost3_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "Qwen2-7B-Instruct": data_file = "Qwen2-7B-Instruct/output/bs1/ZeroShot_BeforePost3_testing.jsonl".replace("ZeroShot", "separate")
    elif args.method == "Distr":
        if args.model == "Llama-2-7b-chat-hf": data_file = "Llama-2-7b-chat-hf/output/bs/PreviousStepNum/ZeroShot_4_PreviousStepNum0_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "MentaLLaMA-chat-7B-hf": data_file = "MentaLLaMA-chat-7B-hf/output/bs/PreviousStepNum/ZeroShot_4_PreviousStepNum0_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "Mistral-7B-Instruct-v0.1": data_file = "Mistral-7B-Instruct-v0.1/output/bs1/ZeroShot_PreviousStepNum0_testing.jsonl".replace("ZeroShot", "separate")
        elif args.model == "Qwen2-7B-Instruct": data_file = "Qwen2-7B-Instruct/output/bs1/ZeroShot_PreviousStepNum0_testing.jsonl".replace("ZeroShot", "separate")
