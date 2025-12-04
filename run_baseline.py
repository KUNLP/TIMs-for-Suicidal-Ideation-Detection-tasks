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

def main(cli_args):
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    args.cache_dir = args.model_name_or_path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # if args.visualization:
    #     if not os.path.exists(os.path.join(args.output_dir, "visualization")):
    #         os.makedirs(os.path.join(args.output_dir, "visualization"))
    args.output_file = os.path.join(args.output_dir, args.output_file)

    print("\n\n"+args.output_file+"\n\n")

    if "Llama-3.1" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.float16)
    elif "Llama-3-" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16)
    elif "Llama-2-13b-chat-hf" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16)
    elif "MentaLLaMA-chat-13B" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("klyang/MentaLLaMA-chat-13B")
        model = AutoModelForCausalLM.from_pretrained("klyang/MentaLLaMA-chat-13B", torch_dtype=torch.float16)
    else:
        # Load
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            cache_dir=args.cache_dir, 
            local_files_only=True,
            # force_download=True, 
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            cache_dir=args.cache_dir,
            # device_map='cuda',
            local_files_only=True,
            #force_download=True, 
        )
    model.to(args.device)
    
    # visualizer = Visualizer(model, tokenizer, device=args.device)

    # Read Data
    with open(args.data_file, "r") as rf:
        datas = [json.loads(d) for d in rf]
    rf.close()

    datas = datas[29:]
    
    model.eval()

    input_max = []
    with open(args.output_file, "a") as f:
        for data_idx, data in tqdm(enumerate(datas), total=len(datas), desc="inference..."):
            system_message = {
                "role": "system",
                "content": "You are a mental health specialist. Analyze the provided text for signs of mental health issues like depression, anxiety, or other mental disorders. Take social factors such as race, gender, age, sexuality, nationality, and religion into account. Provide a brief explanation of your prediction considering both social factors and the text given."
            }
            user_message = {
            "role": "user",
            "content": f"Given the post: '{data['Post']}', analyze the content to predict the user's suicide risk. The output should be in the form of 'OUTPUT: 0 (Ideation)' or 'OUTPUT: 1 (Supportive)' or 'OUTPUT: 2 (Indicator)' or 'OUTPUT: 3 (Attempt)' or 'OUTPUT: 4 (Behavior)'; REASONING:"
            }
            messages = [system_message, user_message]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(args.device)
            # if input_ids.shape[1] > 4000: input_ids = torch.cat((input_ids[:, :3916], input_ids[:, -84:]), dim=-1)
            input_max.append(input_ids.shape[1])
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0,
                )
                generated_ids = generated_ids[:, input_ids.shape[1]:]
                # generated_ids = model.generate(input_ids, max_new_tokens=1000, do_sample=True)
                outputs_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                # first setence in outputs_texts
                outputs_text = split_text_into_sentences(text=outputs_texts,language='en')[0]
            torch.cuda.empty_cache()
            if ("OUTPUT: 1 (Supportive)" in outputs_texts) or ("OUTPUT: 1" in outputs_texts) or ("Supportive" in outputs_text):
                label = "(A) Supportive"
            elif ("OUTPUT: 2 (Indicator)" in outputs_texts) or ("OUTPUT: 2" in outputs_texts) or ("Indicator" in outputs_text):
                label = "(B) Indication"
            elif ("OUTPUT: 0 (Ideation)" in outputs_texts) or ("OUTPUT: 0" in outputs_texts) or ("Ideation" in outputs_text):
                label = "(C) Ideation"
            elif ("OUTPUT: 4 (Behavior)" in outputs_texts) or ("OUTPUT: 4" in outputs_texts) or ("Behavior" in outputs_text):
                label = "(D) Behaviors"
            elif ("OUTPUT: 3 (Attempt)" in outputs_texts) or ("OUTPUT: 3" in outputs_texts) or ("Attempt" in outputs_text):
                label = "(E) Attempt"
            else: 
                label = "N"
                
            output = {"User": data["User"], "Correct": data["Label"], "Predicted_Label": label, "Predicted_Text": outputs_texts}
            f.write(json.dumps(output)+"\n")
            # if args.visualization:
            #     # Add by visualizer
            #     word_importance = visualizer.vis_by_grad(input_ids, label)
            #     visualizer.visualize_importance(word_importance, os.path.join(args.output_dir, "visualization", "User-{data_idx}-{label}.png"))            
            
    print(max(input_max)) # 3908
if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    # Directory
    # cli_parser.add_argument("--data_file", type=str, default='./data/500_anonymized_Reddit_users_posts_labels.jsonl')
    cli_parser.add_argument("--data_file", type=str, default='./data/CSSRS_suicide_testing.jsonl')

    # cli_parser.add_argument("--model_name_or_path", type=str, default="./Llama-2-7b-chat-hf/init_weight")
    # cli_parser.add_argument("--model_name_or_path", type=str, default="./MentaLLaMA-chat-7B-hf/init_weight")
    cli_parser.add_argument("--model_name_or_path", type=str, default="./Meta-Llama-3.1-8B-Instruct/init_weight")
    # ===================================================================================================
    # cli_parser.add_argument("--output_dir", type=str, default="./Llama-2-7b-chat-hf/output/bs")
    # cli_parser.add_argument("--output_dir", type=str, default="./MentaLLaMA-chat-7B-hf/output/bs")
    cli_parser.add_argument("--output_dir", type=str, default="./Meta-Llama-3.1-8B-Instruct/output/bs")
    
    cli_parser.add_argument("--output_file", type=str, default="ZeroShot_testing.jsonl")
    # ------------------------------------------------------------------------------------------------------------
    
    # Running Mode
    cli_args = cli_parser.parse_args()

    main(cli_args)
