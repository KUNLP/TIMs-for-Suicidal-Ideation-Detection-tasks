import argparse 
import os 
import logging 
from attrdict import AttrDict 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import numpy as np
import math

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from tqdm import tqdm
from sentence_splitter import split_text_into_sentences

import matplotlib.pyplot as plt
import seaborn as sns

def make_prompt_all_post(post, prompt_type = 1):
    start_post = "These are posts made by a specific user on the social media site. "
    task_info = "You should choose only one of the labels to represent the severity of the user's suicidal ideation based on their posts. "
    label_info = "The labels represent the increasing severity of each user's suicidal ideation in the following order:\n"
    label_info += f"(A) Supportive\n(B) Indicator\n(C) Ideation\n(D) Behavior\n(E) Attempt\n\n"
        
    
    post_info = (f"The posts written by the user: {post}.\n"
                  f"Based on the latest post provided, "
                  f"you should choose one of the labels (A-E) to categorize the severity of the user's suicidal ideation.\n")
    if prompt_type == 1:
        output_info = "The suicide ideation label for the posts is (" 
    elif prompt_type == 2:
        output_info = "("
    elif prompt_type == 3:
        output_info = "Label ("
    instruction = (start_post + task_info + label_info
                   + post_info +
                   "Question: What is the suicide ideation label for the latest post? "
                   "Output: "+ output_info)
    return instruction

def make_common_prompt(is_score=False, is_originprob=False, before_post_info=[], previous_step_num=5):
    start_post = "These are posts made by a specific user on the social media site. "
    
    if is_score:
        task_info = "Your task is to predict the severity of the user's suicidal ideation based on their posts:\n"
    else:
        task_info = "You should choose only one of the labels to represent the severity of the user's suicidal ideation based on their posts. "

    if is_score:label_info = ""
    else:
        label_info = "The labels represent the increasing severity of each user's suicidal ideation in the following order:\n\n"
        label_info += f"(A) Supportive\n(B) Indicator\n(C) Ideation\n(D) Behavior\n(E) Attempt\n\n"
            
    if is_originprob: post_info = ""
    else: 
        post_info = "Consider the labels from the user's previously written posts to inform your decision.\n\n" if before_post_info else ""
        if len(before_post_info) > previous_step_num:
            post_info += "Summary of previous posts' labels:\n"
            label_counts = {label: before_post_info.count(label) for label in set(before_post_info) if label != "N"}
            for label, count in label_counts.items():
                post_info += f"Label {label}: {count} posts\n" 
    instruction = start_post + task_info + label_info + post_info
    return instruction

def make_previous_step_num_prompt(BeforePost=False, before_post_info=[], before_post=[], previous_step_num=5):
    post_info = ""
    if previous_step_num > 0:
        add_idx = (len(before_post_info) - previous_step_num) if len(before_post_info) > previous_step_num else 0
        for b_idx, (before_post_idx, before_post_text) in enumerate(zip(before_post_info[(-1 * previous_step_num):], before_post)):
            b_idx += add_idx
            num = f"{b_idx + 1}{'th' if 4 <= (b_idx + 1) % 10 else {0: 'th', 1: 'st', 2: 'nd', 3: 'rd'}.get((b_idx + 1) % 10)}"
            if BeforePost:
                post_info += f"The {num} post written by the user: {before_post_text}\n" # BeforePost
            else:
                post_info += f"The {num} post written by the user is difficult to label.\n" if before_post_idx == "N" else f"The {num} post written by the user is labeled {before_post_idx}.\n"
    return post_info

def make_score_prompt(post):   

    post_info = (f"Here is the latest post written by the user: : {post}.\n"
                  f"[Post completed]\n\n"
                  f"When the post is complete, the user who wrote the post is labeled with:\n"
                  f"(A) Supportive\n(B) Indicator\n(C) Ideation\n(D) Behavior\n(E) Attempt\n\n"
                  )
    
    label_info = "The labels represent the increasing severity of each user's suicidal ideation in the following order:\n\n"
    label_info += f"(A) Supportive\n(B) Indicator\n(C) Ideation\n(D) Behavior\n(E) Attempt\n\n"
        
    specific_info = (f"Assign a score from 0 to 10 for each label, reflecting the relative intensity of suicidal ideation the user is likely experiencing. Then critique your answer by thinking it through step by step. Finally, give your revised scores.\n\n"
        f"You must output in the following format, including headings (of course, you should give your own scores), with no additional commentary:\n\n"
        f"First pass scores:\n(A) Supportive: <score>\n(B) Indicator: <score>\n(C) Ideation: <score>\n(D) Behavior: <score>\n(E) Attempt: <score>\n\n"
        f"Critique: <your critique here>\n\n"
        f"Revised scores:\n(A) Supportive: <revised score>\n(B) Indicator: <revised score>\n(C) Ideation: <revised score>\n(D) Behavior: <revised score>\n(E) Attempt: <revised score>\n\n"
        f"[End of answer]\n\n"
        f"Remember: zero is a valid score, meaning that the user is likely not experiencing that label. You must score at least one label > 0."
    )
    instruction = post_info + label_info + post_info +specific_info
    return instruction

def make_end_prompt(post, prompt_num=1):
    post_info = (f"The latest post written by the user: {post}.\n"
                  f"Based on the latest post provided, "
                  f"you should choose one of the labels (A-E) to categorize the severity of the user's suicidal ideation.\n")
    
    instruction = post_info + "Question: What is the suicide ideation label for the latest post? Output: "
    if prompt_num==1: instruction +="The suicide ideation label for the latest post is ("
    elif prompt_num==2: instruction +="("
    elif prompt_num==3: instruction +="Label ("
    return instruction

def masking_wo_selected_index(next_token_logits, selected_indices, device, mask_num=-1000.0):
    mask = torch.full(next_token_logits.shape, mask_num).to(device)
    mask[0, 0, selected_indices] = next_token_logits[0, 0, selected_indices]
    next_token_logits.masked_scatter_(next_token_logits == next_token_logits, mask)
    return next_token_logits

def calculate_entropy(probabilities):
    log_probabilities = torch.log2(torch.where(probabilities > 0, probabilities, torch.ones_like(probabilities)))
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    return entropy.item()

def main(cli_args):
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    args.cache_dir = args.model_name_or_path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    args.output_separate_file = os.path.join(args.output_dir, args.output_separate_file)

    print("\n\n"+args.output_file+"\n\n")
    print("all_post: ", args.all_post)
    print("previous_step_num: ", args.previous_step_num)
    print("BeforePost: ", args.BeforePost)
    print("CoT: ", args.CoT)

    #print(args.model_name_or_path)
    # Load
    if "Llama-3.1" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    elif "Llama-3" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "Llama-2-13b-chat-hf" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16)
    elif "Llama-2-7b-chat-hf" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    elif "MentaLLaMA-chat-7B" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("klyang/MentaLLaMA-chat-7B-hf")
        model = AutoModelForCausalLM.from_pretrained("klyang/MentaLLaMA-chat-7B-hf")
    elif "MentaLLaMA-chat-13B" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("klyang/MentaLLaMA-chat-13B")
        model = AutoModelForCausalLM.from_pretrained("klyang/MentaLLaMA-chat-13B", torch_dtype=torch.float16)
    elif "Mistral-7B" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            cache_dir=args.cache_dir, 
            local_files_only=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            cache_dir=args.cache_dir,
            local_files_only=True,
            # torch_dtype=torch.bfloat16,
        )
    
    model.to(args.device)
    

    # Read Data
    with open(args.data_file, "r") as rf:
        datas = [json.loads(d) for d in rf]
    rf.close()

    label_list = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]
    
    model.eval()

    # input_max = []
    with open(args.output_separate_file, "w") as of, open(args.output_file, "w") as f:
        for data_idx, data in tqdm(enumerate(datas), total=len(datas), desc="inference..."):
            before_info = []
            before_text_info = []
            re_probability_list = []
            origin_probability_list = []
            if args.all_post:
                instruction = make_prompt_all_post(", ".join(data["Post"]), prompt_type = args.prompt)
                if "Mistral" in args.model_name_or_path: 
                    instruction = "[INST] " + instruction + " [/INST]"
                input_ids = tokenizer(instruction, return_tensors="pt").to(args.device)["input_ids"]
                input_ids = input_ids[:, :(args.max_input_len-256)] if "Mistral" not in args.model_name_or_path else torch.cat((input_ids[:, :(args.max_input_len-256)], input_ids[:, -5:]), dim=1)
                with torch.no_grad():
                    model_inputs = model.prepare_inputs_for_generation(input_ids)
                    outputs = model(**model_inputs, 
                                    output_attentions=False,
                                    output_hidden_states=False,
                                    return_dict=True
                                    )
                    logits = outputs["logits"][:, -1:, :]
                
                selected_indices = [tokenizer.encode(label)[args.start_idx] for label in [
                    "A) Supportive", "Supportive", "B) Indicator", "Indicator", "C) Ideation","Ideation", "D) Behavior", "Behavior", "E) Attempt", "Attempt"]]
                
                logits = masking_wo_selected_index(logits, selected_indices, device=args.device)
                probabilities = torch.softmax(logits, dim=-1)
                entropy = calculate_entropy(probabilities)
                re_probabilities = probabilities[:, :, [selected_indices[2*i] for i in range(0, len(label_list))]] + probabilities[:, :, [selected_indices[2*i+1] for i in range(0, len(label_list))]] 
                max_value, max_index = torch.max(re_probabilities, dim=-1)
                predicted_label = label_list[max_index.item()]
            else:
                for post_idx, post in enumerate(data["Post"]):              
                    # Classification
                    instruction1 = make_common_prompt(is_score=args.is_score, is_originprob=args.is_originprob, 
                                                    before_post_info=before_info, previous_step_num=args.previous_step_num)
                    if not args.is_originprob:
                        instruction1 += make_previous_step_num_prompt(BeforePost=args.BeforePost, previous_step_num=args.previous_step_num, 
                                                                        before_post = before_text_info, before_post_info=before_info)
                    if args.is_score:
                        instruction2 = make_score_prompt(post)
                    else: instruction2 = make_end_prompt(post, prompt_num=args.prompt)
                    instruction = instruction1 +instruction2
                    if "Mistral" in args.model_name_or_path: instruction = "[INST] " +instruction + " [/INST]"
                    if (data_idx == 0) and (post_idx < 3): print(instruction)
                    input_ids = tokenizer(instruction, return_tensors="pt").to(args.device)["input_ids"]
                    if input_ids.shape[1] > (args.max_input_len-256): 
                        if "Mistral" in args.model_name_or_path: last_input_ids = input_ids[:, -5:]
                        input_ids = tokenizer(instruction2, return_tensors="pt").to(args.device)["input_ids"][:, 1:] # remove cls token
                        if "Mistral" in args.model_name_or_path: input_ids = torch.cat((input_ids, last_input_ids), dim=1) 
                        max_input_token_len = (args.max_input_len-256)-input_ids.shape[1]
                        if "Mistral" in args.model_name_or_path: instruction1 = "[INST] " +instruction1 
                        input_ids = torch.cat((tokenizer(instruction1, return_tensors="pt").to(args.device)["input_ids"][:, :max_input_token_len], input_ids), dim=1)
                    # input_max.append(input_ids.shape[1])
                    with torch.no_grad():
                        model_inputs = model.prepare_inputs_for_generation(input_ids)
                        outputs = model(**model_inputs, 
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        return_dict=True
                                        )
                        logits = outputs["logits"][:, -1:, :]
                    
                    selected_indices = [tokenizer.encode(label)[args.start_idx] for label in [
                        "A) Supportive", "Supportive", "B) Indicator", "Indicator", "C) Ideation","Ideation", "D) Behavior", "Behavior", "E) Attempt", "Attempt"]]
                    
                    logits = masking_wo_selected_index(logits, selected_indices, device=args.device)
                    probabilities = torch.softmax(logits, dim=-1)
                    entropy = calculate_entropy(probabilities)
                    selected_probabilities = probabilities[:, :, [selected_indices[2*i] for i in range(0, len(label_list))]] + probabilities[:, :, [selected_indices[2*i+1] for i in range(0, len(label_list))]] 
                    
                    re_probabilities = selected_probabilities
                    max_value, max_index = torch.max(re_probabilities, dim=-1)
                    
                    origin_probability_list.append(selected_probabilities)
                    re_probability_list.append(re_probabilities)
                    predicted_label = label_list[max_index.item()]
                    before_info.append(predicted_label)
                    before_text_info.append(post)
                    before_text_info = before_text_info[(-1 * args.previous_step_num):]
                    of.write(json.dumps({"User": data["User"], "Post_Index": post_idx, "Predicted_Label": predicted_label, "Entropy": entropy, "Origin_Probabilities": selected_probabilities.tolist(), "Revised_Probabilities":re_probabilities.tolist()})+"\n")
            output = {"User": data["User"], "Correct": data["Label"], "Predicted_Label": predicted_label, "Entropy": entropy, "Revised_Probabilities":re_probabilities.tolist()}
            
            f.write(json.dumps(output)+"\n")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    # Directory
    cli_parser.add_argument("--data_file", type=str, default='./data/CSSRS_suicide_testing.jsonl')

    cli_parser.add_argument("--model_name_or_path", type=str, default="./MentaLLaMA-chat-7B-hf/init_weight")
    # ===================================================================================================
    cli_parser.add_argument("--output_dir", type=str, default="./MentaLLaMA-chat-7B-hf/output/entropy")
    
    cli_parser.add_argument("--output_separate_file", type=str, default="separate_originprob_A.jsonl")
    cli_parser.add_argument("--output_file", type=str, default="zeroshot_originprob_A.jsonl")
    cli_parser.add_argument("--max_input_len", type=int, default= 4096)
    # ---------------------------------------------------------------------------------------------------
    # instruction
    cli_parser.add_argument("--previous_step_num", type=int, default=0)
    cli_parser.add_argument("--all_post", type=bool, default=False)
    cli_parser.add_argument("--prompt", type=int, default=1)
    cli_parser.add_argument("--start_idx", type=int, default=1) #Qwen2 0 # llama 1
    
    cli_parser.add_argument("--BeforePost", type=bool, default=False)
    cli_parser.add_argument("--CoT", type=bool, default=False)
    cli_parser.add_argument("--is_score", type=bool, default=False)
    cli_parser.add_argument("--is_originprob", type=bool, default=False)
    
    # Running Mode
    cli_args = cli_parser.parse_args()

    main(cli_args)
