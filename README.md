# Leveraging timeline integration methods for effective suicidal ideation detection

The official PyTorch implementation for our paper:*[Leveraging timeline integration methods for effective suicidal ideation detection](https://www.sciencedirect.com/science/article/pii/S0957417425038059)*

## Environment
```bash
conda create -n TIMs python=3.8
conda activate TIMs
pip install -r requirements.txt
```
## Process Structure
<img src='process.png' width='1000'>

## Execute LLM Baseline and Timeline Integration Methods (TIMs)
```bash
bash bash_run.sh
```
## Generate the SID Results and the Reasoning applying with TIMs
```bash
python run_generate.py --model_name_or_path LLM_NAME --output_dir OUTPUT_DIR --output_separate_file OUTPUT_FILE_OF_TURN_LEVEL --output_file FINAL_OUTPUT_FILE --previous_step_num PREVIOUS_STEP_NUM --BeforePost WHICH_TIMs_TYPE
```
## Execute PLM Baselines
### Training
```bash
python run_plm.py --model_name_or_path LLM_NAME --from_init_weight true --output_dir OUTPUT_DIR --seed 42 --learning_rate 5e-6 --train_batch_size 16 --test_batch_size 16 --num_train_epochs 5 --do_train true --do_predict false
```

### Test
```bash
python run_plm.py --model_name_or_path LLM_NAME --output_dir OUTPUT_DIR --checkpoint BEST_CHECKPOINT--seed 42 --test_batch_size 16 --do_predict true
```
## G_Eval
```bash
bash g_eval.sh
```

## Citation

Please cite as:
```
@article{KIM2026130190,
title = {Leveraging timeline integration methods for effective suicidal ideation detection},
journal = {Expert Systems with Applications},
volume = {300},
pages = {130190},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.130190},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425038059},
author = {Seulgi Kim and Jaewook Lee and Kwangil Kim and Harksoo Kim},
keywords = {Suicidal ideation detection, Temporal dependency, Mental health, Social media, Prompt engineering},
abstract = {Suicidal Ideation Detection (SID) is a critical task that involves analyzing a user’s social media posts to assess the severity of suicidal ideation (SI) and enable early intervention. Recent research has leveraged the advanced language understanding capabilities of large language models (LLMs) to perform this task more effectively. However, existing methods often process multiple posts simultaneously, which may overlook temporal dependencies between posts and exceed input length limitations. To address these challenges, we introduce timeline integration methods (TIMs), which model temporal dependencies across posts by incorporating information from earlier time points through prompt components such as the predicted label distribution of previous posts or recent post texts. This approach enables LLMs to track the progression of SI risk and make more accurate severity assessments. Furthermore, it remains unclear whether domain-specific LLMs for the medical field outperform general-purpose LLMs in SID tasks. We apply TIMs across various LLMs and benchmark them against existing methods. Our key findings indicate that: (1) TIMs significantly improve the performance of LLMs on the SID task, particularly on long user posts; (2) the effectiveness of TIMs varies depending on the model size and its capability in reading comprehension; and (3) domain-specific LLMs for the medical field show superior performance in SID, particularly when enhanced with TIMs.}
}
```
