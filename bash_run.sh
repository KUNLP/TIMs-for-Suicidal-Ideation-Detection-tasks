python main_prompt.py --output_separate_file separate_A.jsonl --output_file zeroshot_A.jsonl --prompt 1 --all_post true
python main_prompt.py --output_separate_file separate_B.jsonl --output_file zeroshot_B.jsonl --prompt 2 --all_post true
python main_prompt.py --output_separate_file separate_C.jsonl --output_file zeroshot_C.jsonl --prompt 3 --all_post true

python main_prompt.py --output_separate_file separate_originprob_A.jsonl --output_file zeroshot_originprob_A.jsonl --previous_step_num 0 --prompt 1 --is_originprob true
python main_prompt.py --output_separate_file separate_originprob_B.jsonl --output_file zeroshot_originprob_B.jsonl --previous_step_num 0 --prompt 2 --is_originprob true
python main_prompt.py --output_separate_file separate_originprob_C.jsonl --output_file zeroshot_originprob_C.jsonl --previous_step_num 0 --prompt 3 --is_originprob true

python main_prompt.py --output_separate_file separate_PreviousStepNum0_A.jsonl --output_file zeroshot_PreviousStepNum0_A.jsonl  --previous_step_num 0 --prompt 1
python main_prompt.py --output_separate_file separate_PreviousStepNum1_A.jsonl --output_file zeroshot_PreviousStepNum1_A.jsonl  --previous_step_num 1 --prompt 1
python main_prompt.py --output_separate_file separate_PreviousStepNum2_A.jsonl --output_file zeroshot_PreviousStepNum2_A.jsonl  --previous_step_num 2 --prompt 1
python main_prompt.py --output_separate_file separate_PreviousStepNum3_A.jsonl --output_file zeroshot_PreviousStepNum3_A.jsonl  --previous_step_num 3 --prompt 1
python main_prompt.py --output_separate_file separate_PreviousStepNum4_A.jsonl --output_file zeroshot_PreviousStepNum4_A.jsonl  --previous_step_num 4 --prompt 1
python main_prompt.py --output_separate_file separate_PreviousStepNum5_A.jsonl --output_file zeroshot_PreviousStepNum5_A.jsonl  --previous_step_num 5 --prompt 1

python main_prompt.py --output_separate_file separate_PreviousStepNum0_B.jsonl --output_file zeroshot_PreviousStepNum0_B.jsonl  --previous_step_num 0 --prompt 2
python main_prompt.py --output_separate_file separate_PreviousStepNum1_B.jsonl --output_file zeroshot_PreviousStepNum1_B.jsonl  --previous_step_num 1 --prompt 2
python main_prompt.py --output_separate_file separate_PreviousStepNum2_B.jsonl --output_file zeroshot_PreviousStepNum2_B.jsonl  --previous_step_num 2 --prompt 2
python main_prompt.py --output_separate_file separate_PreviousStepNum3_B.jsonl --output_file zeroshot_PreviousStepNum3_B.jsonl  --previous_step_num 3 --prompt 2
python main_prompt.py --output_separate_file separate_PreviousStepNum4_B.jsonl --output_file zeroshot_PreviousStepNum4_B.jsonl  --previous_step_num 4 --prompt 2
python main_prompt.py --output_separate_file separate_PreviousStepNum5_B.jsonl --output_file zeroshot_PreviousStepNum5_B.jsonl  --previous_step_num 5 --prompt 2

python main_prompt.py --output_separate_file separate_PreviousStepNum0_C.jsonl --output_file zeroshot_PreviousStepNum0_C.jsonl  --previous_step_num 0 --prompt 3
python main_prompt.py --output_separate_file separate_PreviousStepNum1_C.jsonl --output_file zeroshot_PreviousStepNum1_C.jsonl  --previous_step_num 1 --prompt 3
python main_prompt.py --output_separate_file separate_PreviousStepNum2_C.jsonl --output_file zeroshot_PreviousStepNum2_C.jsonl  --previous_step_num 2 --prompt 3
python main_prompt.py --output_separate_file separate_PreviousStepNum3_C.jsonl --output_file zeroshot_PreviousStepNum3_C.jsonl  --previous_step_num 3 --prompt 3
python main_prompt.py --output_separate_file separate_PreviousStepNum4_C.jsonl --output_file zeroshot_PreviousStepNum4_C.jsonl  --previous_step_num 4 --prompt 3
python main_prompt.py --output_separate_file separate_PreviousStepNum5_C.jsonl --output_file zeroshot_PreviousStepNum5_C.jsonl  --previous_step_num 5 --prompt 3

python main_prompt.py --output_separate_file separate_BeforePost1_A.jsonl --output_file zeroshot_BeforePost1_A.jsonl  --previous_step_num 1 --BeforePost true --prompt 1
python main_prompt.py --output_separate_file separate_BeforePost2_A.jsonl --output_file zeroshot_BeforePost2_A.jsonl  --previous_step_num 2 --BeforePost true --prompt 1
python main_prompt.py --output_separate_file separate_BeforePost3_A.jsonl --output_file zeroshot_BeforePost3_A.jsonl  --previous_step_num 3 --BeforePost true --prompt 1
python main_prompt.py --output_separate_file separate_BeforePost4_A.jsonl --output_file zeroshot_BeforePost4_A.jsonl  --previous_step_num 4 --BeforePost true --prompt 1
python main_prompt.py --output_separate_file separate_BeforePost5_A.jsonl --output_file zeroshot_BeforePost5_A.jsonl  --previous_step_num 5 --BeforePost true --prompt 1

python main_prompt.py --output_separate_file separate_BeforePost1_B.jsonl --output_file zeroshot_BeforePost1_B.jsonl  --previous_step_num 1 --BeforePost true --prompt 2
python main_prompt.py --output_separate_file separate_BeforePost2_B.jsonl --output_file zeroshot_BeforePost2_B.jsonl  --previous_step_num 2 --BeforePost true --prompt 2
python main_prompt.py --output_separate_file separate_BeforePost3_B.jsonl --output_file zeroshot_BeforePost3_B.jsonl  --previous_step_num 3 --BeforePost true --prompt 2
python main_prompt.py --output_separate_file separate_BeforePost4_B.jsonl --output_file zeroshot_BeforePost4_B.jsonl  --previous_step_num 4 --BeforePost true --prompt 2
python main_prompt.py --output_separate_file separate_BeforePost5_B.jsonl --output_file zeroshot_BeforePost5_B.jsonl  --previous_step_num 5 --BeforePost true --prompt 2

python main_prompt.py --output_separate_file separate_BeforePost1_C.jsonl --output_file zeroshot_BeforePost1_C.jsonl  --previous_step_num 1 --BeforePost true --prompt 3
python main_prompt.py --output_separate_file separate_BeforePost2_C.jsonl --output_file zeroshot_BeforePost2_C.jsonl  --previous_step_num 2 --BeforePost true --prompt 3
python main_prompt.py --output_separate_file separate_BeforePost3_C.jsonl --output_file zeroshot_BeforePost3_C.jsonl  --previous_step_num 3 --BeforePost true --prompt 3
python main_prompt.py --output_separate_file separate_BeforePost4_C.jsonl --output_file zeroshot_BeforePost4_C.jsonl  --previous_step_num 4 --BeforePost true --prompt 3
python main_prompt.py --output_separate_file separate_BeforePost5_C.jsonl --output_file zeroshot_BeforePost5_C.jsonl  --previous_step_num 5 --BeforePost true --prompt 3

