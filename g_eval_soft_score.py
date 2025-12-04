import json

for method  in ["Distr", "+PL", "+POST"]: 
    for model in ["Llama-2-7b-chat-hf", "MentaLLaMA-chat-7B-hf", "Mistral-7B-Instruct-v0.1"]:
        for eval_type in ["Coherence", "Information_Appropriateness", "Helpfulness"]:
            with open(f"g_eval/{eval_type}/{model}_{method}.jsonl", "r") as f:
                all_results = [json.loads(d) for d in f]
            # 평균 soft score 전체 출력
            all_soft = [o["soft_score"] for o in all_results]
            print(f"{eval_type}/{model}_{method}.jsonl")
            print("Average soft score:", round(sum(all_soft)/len(all_soft), 4)) 
            print("\n\n")
