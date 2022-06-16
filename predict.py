import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import sys

import random
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)

from data import create_dataloader, read_text_pair, DatasetRetriever
from model import *
from transformers import *
from train import set_seed
from utils import *

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--model_path", type=str, required=True, help="path of pretrain model")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--task", default=1, type=int, help='task1 is pcl, 2 is multilabel')
parser.add_argument("--model_type", default='last3hidden', type=str, help='model type')
parser.add_argument("--threshold", default=0.5, type=float, help='hyper param of mlabel threshold')
parser.add_argument("--seed", default=2021, type=int, help='seed')
parser.add_argument("--test", default=0, type=int, help='calculate metrics if equal to 0')
parser.add_argument('--device', choices=['cpu', 'cuda'], default="cuda",
                    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()




def predict(model, data_loader, task, is_test=False):
    """
    Predicts the data labels.
    Args:
        model (obj:`QuestionMatching`): A model to calculate whether the question pair is semantic similar or not.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    print(args.seed)
    set_seed(args.seed)
    batch_logits = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            input_ids, token_type_ids, attention_mask = batch['input_ids'], batch['token_type_ids'], batch['attention_mask']
            input_ids, token_type_ids, attention_mask = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda()
            batch_logit, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, do_evaluate=True, task=task)
            
            if task == 1:
                pred_cal = torch.argsort(batch_logit, dim=-1, descending=True)
                pred_cal = pred_cal[..., :1]
                batch_logits.append(pred_cal.cpu().numpy().tolist())
            else:
                act_fct = nn.Sigmoid()
                batch_logit = act_fct(batch_logit)
                pred_cal = (batch_logit > args.threshold).float()
                batch_logits.append(pred_cal.cpu().numpy().tolist())

        batch_logits = np.vstack(batch_logits)
        if task == 1:
            if is_test:
                labels2file([k for k in batch_logits], 'data/result/test/task1.txt')
            else:
                labels2file([k for k in batch_logits], 'data/result/dev/task1.txt')
                precision, recall, f1 = eval_task1()
        else:
            if is_test:
                labels2file(batch_logits, 'data/result/test/task2.txt')
            else:
                labels2file(batch_logits, 'data/result/dev/task2.txt')
                f1 = eval_task2()
        return


if __name__ == "__main__":

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    is_test = True if args.test == 1 else False
    if args.task == 1:
        test_set = read_text(args.input_file, task=1, is_test=is_test)
        print('task1')
    else:
        test_set = read_text(args.input_file, task=2, is_test=is_test)
        print('task2')
    test_dataset = DatasetRetriever(test_set, tokenizer, args.max_seq_length, is_test=True)

    test_loader = create_dataloader(test_dataset, mode='test', batch_size=args.batch_size)
    
    if args.task == 1:
        num_labels = 2
    else:
        num_labels = 7
    if args.model_type == 'last2cls':
        print('use last2cls model')
        model = ModelLastTwoCLS(args.model_path)
    else:
        model = Model(args.model_path, num_labels=num_labels)
    #model = nn.DataParallel(model, device_ids=[0,1])
    model.cuda()

    if args.params_path and os.path.isfile(args.params_path):
        model.load_state_dict(torch.load(args.params_path))
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")


    y_probs = predict(model, test_loader, task=args.task, is_test=is_test)
    print('finish')
