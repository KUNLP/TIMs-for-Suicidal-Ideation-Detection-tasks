import argparse 
import os 
import logging 
from attrdict import AttrDict 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from tqdm import tqdm
from sentence_splitter import split_text_into_sentences

import matplotlib.pyplot as plt
import seaborn as sns
       
def make_prompt(post, BeforePost=False, before_post_info=[], before_post=[], previous_step_num=5):

    start_post = "These are posts made by a specific user on the social media site. "
    task_info = "You should choose only one of the labels to represent the severity of each user's suicidal ideation based on their post. "

    label_info = "The labels represent the increasing severity of each user's suicidal ideation in the following order:\n"
    label_info += "(A) Supportive\n(B) Indication\n(C) Ideation\n(D) Behaviors\n(E) Attempt\n"
    
    post_info = "Consider the labels of the user's previously written posts to inform your decision.\n\n" if before_post_info else ""
    
    if len(before_post_info) > previous_step_num:
        label_counts = {label: before_post_info.count(label) for label in set(before_post_info) if label != "N"}
        post_info += "Summary of previous posts' labels:\n"
        for label, count in label_counts.items():
            post_info += f"Label {label}: {count} posts\n"
        
    if previous_step_num > 0:
        add_idx = (len(before_post_info) - previous_step_num) if len(before_post_info) > previous_step_num else 0
        for b_idx, (before_post_idx, before_post_text) in enumerate(zip(before_post_info[(-1 * previous_step_num):], before_post)):
            b_idx += add_idx
            num = f"{b_idx + 1}{'th' if 4 <= (b_idx + 1) % 10 else {0: 'th', 1: 'st', 2: 'nd', 3: 'rd'}.get((b_idx + 1) % 10)}"
            if BeforePost:
                post_info += f"The {num} post written by the user: {before_post_text}\n" 
            else:
                post_info += f"The {num} post written by the user is difficult to label.\n" if before_post_idx == "N" else f"The {num} post written by the user is labeled {before_post_idx}.\n"

    post_info += (f"The latest post written by the user: {post}.\n"
                  "Based on the latest post provided, you should choose one of the labels (A-E) to categorize the severity of the user's suicidal ideation.\n")

    instruction = (start_post + task_info + label_info + "\n\n"
                   + post_info +
                   "Q: What is the suicide ideation label for the latest post? "
                   "Output: ")
    return instruction


def main(cli_args):
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    args.cache_dir = args.model_name_or_path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    args.output_separate_file = os.path.join(args.output_dir, args.output_separate_file)

    print("\n\n"+args.output_file+"\n\n")
    print("previous_step_num: ", args.previous_step_num)
    print("BeforePost: ", args.BeforePost)
    print("Description: ", args.Description)
    print("CoT: ", args.CoT)
    print("RolePlay: ", args.RolePlay)


    # Load
    if "Llama-3.1" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    elif "Llama-3-" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "Mistral" in args.model_name_or_path:
       model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
       tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
    elif "Llama-2-13b-chat-hf" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16)
    elif "MentaLLaMA-chat-13B" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("klyang/MentaLLaMA-chat-13B")
        model = AutoModelForCausalLM.from_pretrained("klyang/MentaLLaMA-chat-13B", torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            cache_dir=args.cache_dir, 
            local_files_only=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            cache_dir=args.cache_dir,
            # device_map='cuda',
            local_files_only=True,
            # torch_dtype=torch.bfloat16,
        )
    model.to(args.device)
    
    # Read Data
    with open(args.data_file, "r") as rf:
        datas = [json.loads(d) for d in rf]
    rf.close()

    model.eval()

    with open(args.output_separate_file, "w") as of, open(args.output_file, "w") as f:
        for data_idx, data in tqdm(enumerate(datas), total=len(datas), desc="inference..."):
            before_info = []
            before_text_info = []
            for post_idx, post in enumerate(data["Post"]):
                # Classification
                instruction = make_prompt(post, 
                before_post_info = before_info, 
                before_post = before_text_info, BeforePost=args.BeforePost, 
                previous_step_num = args.previous_step_num)
                
                if "Mistral" in args.model_name_or_path: 
                    instruction = "[INST] " +instruction + " [/INST]"
                
                input_ids = tokenizer(instruction, return_tensors="pt").to(args.device)["input_ids"]
                with torch.no_grad():
                    generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0,
                    )                    
                    generated_ids = generated_ids[:, input_ids.shape[1]:]
                    outputs_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    outputs_texts  = outputs_texts.replace("\n", " ").replace("(A) Supportive (B) Indication (C) Ideation (D) Behaviors (E) Attempt", "").replace("A) Supportive B) Indication C) Ideation D) Behaviors E) Attempt", "")

                    if outputs_texts != "": 
                        outputs_text = split_text_into_sentences(text=outputs_texts,language='en')
                        if len(outputs_text) > 1: outputs_text = outputs_text[0] + outputs_text[-1]
                        else: outputs_text = outputs_text[0]
                    else: outputs_text = ""

                if ("(A) Supportive" in outputs_text) or ("(A)" in outputs_text) or ("Supportive" in outputs_text) or ("A" == outputs_texts.split("Reasoning:")[0].strip().split("Explanation:")[0].strip()):
                    label = "(A) Supportive"
                elif ("(B) Indication" in outputs_text) or ("(B)" in outputs_text) or ("Indication" in outputs_text) or ("B" == outputs_texts.split("Reasoning:")[0].strip().split("Explanation:")[0].strip()):
                    label = "(B) Indication"
                elif ("(C) Ideation" in outputs_text) or ("(C)" in outputs_text) or ("Ideation" in outputs_text) or ("C" == outputs_texts.split("Reasoning:")[0].strip().split("Explanation:")[0].strip()):
                    label = "(C) Ideation"
                elif ("(D) Behaviors" in outputs_text) or ("(D)" in outputs_text) or ("Behaviors" in outputs_text) or ("D" == outputs_texts.split("Reasoning:")[0].strip().split("Explanation:")[0].strip()):
                    label = "(D) Behaviors"
                elif ("(E) Attempt" in outputs_text) or ("(E)" in outputs_text) or ("Attempt" in outputs_text) or ("E" == outputs_texts.split("Reasoning:")[0].strip().split("Explanation:")[0].strip()):
                    label = "(E) Attempt"
                else: 
                    label = "N"
                before_info.append(label)
                before_text_info.append(post)
                before_text_info = before_text_info[(-1 * args.previous_step_num):]
                of.write(json.dumps({"User": data["User"], "Post_Index": post_idx, "Predicted_Label": label, "Predicted_Text": outputs_texts})+"\n")
            output = {"User": data["User"], "Correct": data["Label"], "Predicted_Label": label, "Predicted_Text": outputs_texts}
            f.write(json.dumps(output)+"\n")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    # Directory
    cli_parser.add_argument("--data_file", type=str, default='./data/CSSRS_suicide_testing.jsonl')

    cli_parser.add_argument("--model_name_or_path", type=str, default="./Mistral-7B-Instruct-v0.1/init_weight")
    # ===================================================================================================
    cli_parser.add_argument("--output_dir", type=str, default="./Mistral-7B-Instruct-v0.1/output/bs")
    
    cli_parser.add_argument("--output_separate_file", type=str, default="separate_4_PreviousStepNum1_testing.jsonl")
    cli_parser.add_argument("--output_file", type=str, default="ZeroShot_4_PreviousStepNum1_testing.jsonl")
    # ------------------------------------------------------------------------------------------------------------
    # instruction
    cli_parser.add_argument("--previous_step_num", type=int, default=1)    
    cli_parser.add_argument("--BeforePost", type=bool, default=False)
    
    # Running Mode
    cli_args = cli_parser.parse_args()

    main(cli_args)
