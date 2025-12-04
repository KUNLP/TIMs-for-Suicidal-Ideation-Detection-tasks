import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch


def init_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        "./MentaLLaMA-chat-7B-hf/init_weight", 
        cache_dir= "./MentaLLaMA-chat-7B-hf/init_weight", 
        local_files_only=True,
        # force_download=True, 
    )
    # print(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
         "./MentaLLaMA-chat-7B-hf/init_weight", 
        cache_dir= "./MentaLLaMA-chat-7B-hf/init_weight",
        # device_map='cuda',
        local_files_only=True,
        #force_download=True, 
    )
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model.to("cuda")
    
    
    return tokenizer, model

def process_rows(rows, tokenizer, model, system_message, device):
    results = []
    
    for _, row in rows.iterrows():
        user_message = {
            "role": "user",
            "content": f"Given the post: '{row['text']}', analyze the content to predict the user's suicide risk. The output should be in the form of 'OUTPUT: 0 (Ideation)' or 'OUTPUT: 1 (Supportive)' or 'OUTPUT: 2 (Indicator)' or 'OUTPUT: 3 (Attempt)' or 'OUTPUT: 4 (Behavior)'; REASONING:"
        }


        messages = [system_message, user_message]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0,
        )
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)

        # Extract prediction and reasoning from the response
        prediction = None
        for i in range(5):
            if f"OUTPUT: {i}" in decoded_response:
                prediction = i
                break
        reasoning = decoded_response.split("REASONING:")[1].strip() if "REASONING:" in decoded_response else ""

        results.append((prediction, reasoning))
    
    return results

def main():
    model_id =  "./MentaLLaMA-chat-7B-hf/init_weight"
    df = pd.read_csv("./data/CSSRS_suicide_testing.csv")

    system_message = {
        "role": "system",
        "content": "You are a mental health specialist. Analyze the provided text for signs of mental health issues like depression, anxiety, or other mental disorders. Take social factors such as race, gender, age, sexuality, nationality, and religion into account. Provide a brief explanation of your prediction considering both social factors and the text given."
    }
        
    # Initialize the model and tokenizer
    tokenizer, model = init_model(model_id)

    # Split DataFrame into chunks for parallel processing
    num_gpus = torch.cuda.device_count()
    #chunk_size = len(df) // num_gpus
    chunk_size = (len(df) + num_gpus - 1) // num_gpus  # Ensure all rows are included
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    
    # Process rows in parallel
    results = []
    for chunk in chunks:
        #results += process_rows(chunk, tokenizer, model, system_message, device)
        results.extend(process_rows(chunk, tokenizer, model, system_message, device="cuda"))
    
    # Flatten the list of results
    predictions, reasonings = zip(*results)
    
    # Ensure the lengths match
    assert len(predictions) == len(df), f"Mismatch: {len(predictions)} predictions for {len(df)} rows"
    assert len(reasonings) == len(df), f"Mismatch: {len(reasonings)} reasonings for {len(df)} rows"
    
    # Add the predictions and reasoning to the DataFrame
    df['prediction'] = predictions
    df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce').astype('Int64')
    df['reasoning'] = reasonings

    # Save the DataFrame to a new CSV file
    df.to_csv('./MentaLLaMA-chat-7B-hf/output/bs/CSSRS_suicide_var_MentalLlama_7B.csv', index=False)

    print(f"Predictions and reasoning saved to original file")

if __name__ == "__main__":
    main()
