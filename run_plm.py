import torch
from transformers import AutoModelForSequenceClassification, RobertaModel, BertModel
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

import argparse 
import os, json
import logging
import random
from fastprogress.fastprogress import master_bar, progress_bar

from attrdict import AttrDict 
from transformers import AutoTokenizer 
from transformers import AutoConfig

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_sklearn_score(predicts, corrects):
    
    pred_num = np.array(predicts)
    corr_num = np.array(corrects)
    
    N = len(pred_num)
    FP = np.sum(pred_num > corr_num) / N
    FN = np.sum(pred_num < corr_num) / N
    
    # Grade Precision (GP) 계산
    GP = (1 - FP) / (1 + FN) if (1 + FN) != 0 else 0
    # Grade Recall (GR) 계산
    GR = (1 - FN) / (1 + FP) if (1 + FP) != 0 else 0
    # Grade F1-Score (GF1) 계산
    GF1 = 2 * (GP * GR) / (GP + GR) if (GP + GR) != 0 else 0
    
    result = {"accuracy": accuracy_score(corrects, predicts),
              "macro_precision": precision_score(corrects, predicts, average="macro"),
              "micro_precision": precision_score(corrects, predicts, average="micro"),
              "grade_precision": GP,
              "macro_f1": f1_score(corrects, predicts, average="macro"),
              "micro_f1": f1_score(corrects, predicts, average="micro"),
              "weighted_f1": f1_score(corrects, predicts, average="weighted"),
              "grade_f1": GF1,
              "macro_recall": recall_score(corrects, predicts, average="macro"),
              "micro_recall": recall_score(corrects, predicts, average="micro"),
              "grade_recall": GR
              }

    label_list = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]
    print(classification_report(corrects, predicts, target_names= label_list, digits=4))

    for k, v in result.items():
        # result[k] = round(v, 6)
        print(k + ": " + str(v))
    return result

def train(args, model, logger, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    # t_total: total optimization step
    # optimization 최적화 schedule 을 위한 전체 training step 계산
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    # optimizer 및 scheduler 선언
    optimizer = AdamW(model.parameters(), lr=args.learning_rate) #, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    num_warmup_steps = min(int(t_total * args.warmup_ratio), 10000)
    scheduler = get_cosine_schedule_with_warmup(
    # scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # loss buffer 초기화
    model.zero_grad()

    mb = master_bar(range(int(args.num_train_epochs)))
    set_seed(args)

    tr_loss, current_iter = 0.0, 0
    global_step = 1
    epoch_idx = 0
    if not args.from_init_weight: 
        global_step += int(args.checkpoint)
        epoch_idx += int(args.checkpoint)
    
    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = progress_bar(train_loader, parent=mb)
        # print(epoch_iterator)
        for step, batch in enumerate(epoch_iterator):
            # train 모드로 설정
            model.train()
            inputs = {k: v.to(args.device) for k, v in batch.items() if k not in ['user','labels']}
            labels = batch['labels'].to(args.device)
            # print([(k, v.shape) for k ,v in inputs.items()])
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            current_iter += 1
            
            # Loss 출력
            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step + 1), loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            # torch.cuda.empty_cache()
        epoch_idx += 1
        print(f'Epoch {epoch_idx}, Loss: {tr_loss/len(train_loader)}')
        output_dir = os.path.join(args.output_dir, "model/checkpoint-{}".format(epoch_idx))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch.pt"))
        logger.info("Saving model checkpoint to %s", os.path.join(output_dir, "pytorch.pt"))
        mb.write("Epoch {} done".format(epoch + 1))
    return global_step, tr_loss / global_step

def test(args, model, test_data, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # 최종 출력 파일 저장을 위한 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    
    if not args.all_post:
        user_idx2user = {user_idx: d["User"] for user_idx, d in enumerate(test_data)}
        user_idx2logit = {user_idx: [] for user_idx, d in enumerate(test_data)}
        user_idx2label = {user_idx: [] for user_idx, d in enumerate(test_data)}
        user = [d["User"] for d in test_data]
        post = [d["Post"] for d in test_data]
        label2idx = {"Supportive": 0, "Indicator": 1, "Ideation": 2, "Behavior": 3, "Attempt": 4}
        label = [label2idx[d["Label"]] for d in test_data]
        test_dataframe = pd.DataFrame({"user": user, "text": post, "label": label})
    
    for batch in progress_bar(test_loader):
        # 모델을 평가 모드로 변경
        model.eval()
        with torch.no_grad():
            inputs = {k: v.to(args.device) for k, v in batch.items() if k not in ['user','labels']}
            labels = batch['labels'].to(args.device)
            outputs = model(**inputs)
            if args.all_post:
                preds = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.shape[0]
                all_preds.extend(preds.cpu().numpy())
            else:
                for label, user, logit in zip(labels, batch['user'], outputs.logits):
                    user_idx2logit[user.item()].append(logit.unsqueeze(0))
                    if user_idx2label[user.item()] == []:
                        user_idx2label[user.item()]=label
    
    if not args.all_post:
        for user_idx, logit_list in user_idx2logit.items():
            preds = torch.argmax(torch.mean(torch.cat(logit_list, dim=0), dim=0), dim=0)
            labels = user_idx2label[user_idx]
            correct_predictions += (preds == labels).sum().item()
            total_predictions += 1
            all_preds.extend(preds.unsqueeze(0).cpu().numpy())
    
    assert len(all_preds) == len(test_data) == total_predictions
    
    label2idx = {"Supportive": 0, "Indicator": 1, "Ideation": 2, "Behavior": 3, "Attempt": 4}
    gold_labels = [label2idx[d["Label"]] for d in test_data]
    results = get_sklearn_score(all_preds, gold_labels)
    print(results)
    
    output_dir = os.path.join( args.output_dir, 'test-{}'.format(str(args.checkpoint)))
    out_file_type = 'a'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        out_file_type ='w'
    output_test_file = os.path.join(output_dir, "predicted_result.jsonl")
    idx2label = {0:"Supportive", 1:"Indicator", 2:"Ideation", 3:"Behavior", 4:"Attempt"}
    with open(output_test_file, out_file_type, encoding='utf-8') as f:
        print('\n\n=====================outputs=====================')
        for d, p in zip(test_data, all_preds):
            out = {"User": d["User"], "Label": d["Label"], "Pred":idx2label[p]}
            f.write(json.dumps(out)+"\n") 
        
    return results

def create_model(args):
    # Load
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, 
        #if args.from_init_weight else os.path.join(args.output_dir, "model/checkpoint-{}".format(args.checkpoint)),
        cache_dir=args.cache_dir, local_files_only=True,
        # force_download=True,
    )

    config.num_labels = args.num_labels
    
    config.output_attentions = True
    args.hidden_size = config.hidden_size

    # print(config)
    # print(config.hidden_dropout_prob)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        # do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir, local_files_only=True,
        # force_download=True, 
    )
    # print(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config = config, 
            cache_dir=args.cache_dir, local_files_only=True,
            #force_download=True, 
        )

    print("tokenizer.sep_token_id: ", tokenizer.sep_token_id)
    
    if not args.from_init_weight: model.load_state_dict(torch.load(os.path.join(args.output_dir, "model/checkpoint-{}/pytorch.pt".format(args.checkpoint))))
    model.to(args.device)
    ## gpu 여러개 사용시
    # model = torch.nn.DataParallel(model)

    return model, tokenizer

def make_dataframe(tokenizer, data):
    label2idx = {"Supportive": 0, "Indicator": 1, "Ideation": 2, "Behavior": 3, "Attempt": 4}
    user2idx = {d["User"]: user_idx for user_idx, d in enumerate(data)}
    dataframe = {"user": [], "label": [], "input_ids":[], "attention_mask": []}
    for d in data:
        user = user2idx[d["User"]]
        label = label2idx[d["Label"]]
        encoding = tokenizer.encode(d["Post"][0])[:512]
        for post in d["Post"][1:]:
            if len(encoding + tokenizer.encode(post)[1:512]) <= 512:
                encoding += tokenizer.encode(post)[1:512]
            else:
                attn_mask = [1] * len(encoding)
                if len(encoding) < 512:
                    attn_mask += [0] * (512-len(encoding))
                    encoding +=  [tokenizer.pad_token_id] * (512-len(encoding))
                dataframe["user"].append(user)
                dataframe["label"].append(label)
                dataframe["input_ids"].append(torch.tensor(encoding, dtype=torch.long))
                dataframe["attention_mask"].append(torch.tensor(attn_mask, dtype=torch.long))
                
                encoding = tokenizer.encode(post)[:512]
        attn_mask = [1] * len(encoding)
        if len(encoding) < 512:
            attn_mask += [0] * (512-len(encoding))
            encoding += [tokenizer.pad_token_id] * (512-len(encoding))
        dataframe["user"].append(user)
        dataframe["label"].append(label)
        dataframe["input_ids"].append(torch.tensor(encoding, dtype=torch.long))
        dataframe["attention_mask"].append(torch.tensor(attn_mask, dtype=torch.long))
    # print([(k, len(v)) for k, v in dataframe.items()])
    return pd.DataFrame(dataframe)

def main(cli_args):
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    # args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.cache_dir = args.model_name_or_path

    print("all_post: ", args.all_post)
    if not args.all_post: 
        args.output_dir = args.output_dir.replace("all_post", "Chunk") #"majority_rule")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = logging.getLogger(__name__)

    init_logger()
    set_seed(args)

    model, tokenizer = create_model(args)

    class CSSRS_All_Dataset(Dataset):
        def __init__(self, dataframe):
            self.dataframe = dataframe
            
        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            encoding = tokenizer.encode_plus(
                ", ".join(row['text']),
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            label = row['label']

            return {
                'input_ids': torch.tensor(encoding['input_ids'].squeeze(0), dtype=torch.long), 
                'attention_mask': torch.tensor(encoding['attention_mask'].squeeze(0), dtype=torch.long), 
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
    class CSSRS_Chunk_Dataset(Dataset):
        def __init__(self, dataframe):
            self.dataframe = dataframe
            
        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            label = row['label']
            return {
                'input_ids': row['input_ids'], 
                'attention_mask': row['attention_mask'], 
                'user': torch.tensor(row["user"], dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    label2idx = {"Supportive": 0, "Indicator": 1, "Ideation": 2, "Behavior": 3, "Attempt": 4}
    if args.do_train:
        with open(os.path.join(args.data_dir, args.train_file), "r") as f:
            train_data = [json.loads(d) for d in f]
        if args.all_post:
            user = [d["User"] for d in train_data]
            post = [d["Post"] for d in train_data]    
            label = [label2idx[d["Label"]] for d in train_data]
            train_dataframe = pd.DataFrame({"user": user, "text": post, "label": label})
            train_dataset = CSSRS_All_Dataset(train_dataframe)
        else:
            train_dataframe = make_dataframe(tokenizer, train_data)
            train_dataset = CSSRS_Chunk_Dataset(train_dataframe)

        train(args, model, logger, train_dataset)
        
    if args.do_predict:
        with open(os.path.join(args.data_dir, args.test_file), "r") as f:
            test_data = [json.loads(d) for d in f]
            
        if args.all_post:
            user = [d["User"] for d in test_data]
            post = [d["Post"] for d in test_data]
            label = [label2idx[d["Label"]] for d in test_data]
            test_dataframe = pd.DataFrame({"user": user, "text": post, "label": label})
            test_dataset = CSSRS_All_Dataset(test_dataframe)    
        else:
            test_dataframe = make_dataframe(tokenizer, test_data)
            test_dataset = CSSRS_Chunk_Dataset(test_dataframe)
        
        test(args, model, test_data, test_dataset)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    # Directory

    cli_parser.add_argument("--data_dir", type=str, default="./data")
    cli_parser.add_argument("--train_file", type=str, default= 'CSSRS_suicide_training.jsonl')
    cli_parser.add_argument("--test_file", type=str, default='CSSRS_suicide_testing.jsonl')
    
    cli_parser.add_argument("--output_dir", type=str, default="./mental-bert-base-uncased/output/all_post")
    cli_parser.add_argument("--model_name_or_path", type=str, default="mental-bert-base-uncased/init_weight")
    cli_parser.add_argument("--cache_dir", type=str, default="./mental-bert-base-uncased/init_weight")
    
    cli_parser.add_argument("--num_labels", type=int, default=5)
    # ------------------------------------------------------------------------------------------------------------
    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=5e-6)
    cli_parser.add_argument("--train_batch_size", type=int, default=16)
    cli_parser.add_argument("--test_batch_size", type=int, default=16)
    cli_parser.add_argument("--num_train_epochs", type=int, default=5)

    cli_parser.add_argument("--logging_steps", type=int, default=100)
    cli_parser.add_argument("--seed", type=int, default=1234) #1000) #42)
    cli_parser.add_argument("--threads", type=int, default=8)

    cli_parser.add_argument("--weight_decay", type=float, default=0.1) # 모델의 손실 함수에 가중치의 크기에 비례하는 항을 추가하여 가중치의 크기가 크지 않도록 제한 -> 일반화 성능 향상
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-8)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    cli_parser.add_argument("--warmup_ratio", type=int, default=0.06) # learning rate 조절
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--do_lower_case", type=bool, default=False)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    # ------------------------------------------------------------------------------------------------------------

    # Running Mode
    cli_parser.add_argument("--from_init_weight", type=bool, default= False) #True)
    cli_parser.add_argument("--checkpoint", type=str, default="4")

    cli_parser.add_argument("--all_post", type=bool, default= False)

    cli_parser.add_argument("--do_train", type=bool, default = False) #True) 
    cli_parser.add_argument("--do_predict", type=bool, default=True)
    
    # ------------------------------------------------------------------------------------------------------------
    cli_args = cli_parser.parse_args()

    main(cli_args)
