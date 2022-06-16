import torch
import numpy as np
import torch.nn as nn
from transformers import *
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler, WeightedRandomSampler
)
import os
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
import random
from datasets import load_dataset

city = {'lk':'Sri Lanka', 'ca':'Canada', 'ie':'Ireland', 'in':'India', 'sg':'Singapore', 'us':'United States', 'tz':'Tanzania', 'za':'South Africa', 'jm':'Jamaica', 'pk':'Pakistan', 
'ph':'Philipines', 'ng':'Nigeria', 'gb':'UK', 'ke':'Kenya', 'bd':'Bangladesh', 'my':'Malaysia', 'nz':'New Zealand', 'hk':'Hong Kong', 'gh':'Ghana', 'au':'Australia'}

class DatasetRetriever(Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.input_id = [e['input_id'] for e in data]
        self.attn_mask = [e['attn_mask'] for e in data]
        self.pos = [e['pos'] for e in data]
        if is_test == False:
            self.label = [e['label'] for e in data]
        self.is_test = is_test
    def __len__(self):
        return len(self.data)
    
    def get_classes(self):
        return self.label, np.bincount(self.label)

    def __getitem__(self, item):
        if self.is_test==False and isinstance(self.label[item], str):
            self.label[item] = int(self.label[item])
        if self.is_test:
            return {
                'input_ids': torch.tensor(self.input_id[item], dtype=torch.long),
                'token_type_ids': torch.tensor([0] * len(self.input_id[item]), dtype=torch.long),
                'attention_mask': torch.tensor(self.attn_mask[item], dtype=torch.long),
                'pos': torch.tensor(self.pos[item], dtype=torch.long),
            }
        else:
            return {
                'input_ids': torch.tensor(self.input_id[item], dtype=torch.long),
                'token_type_ids': torch.tensor([0] * len(self.input_id[item]), dtype=torch.long),
                'attention_mask': torch.tensor(self.attn_mask[item], dtype=torch.long),
                'pos': torch.tensor(self.pos[item], dtype=torch.long),
                'label': torch.tensor(self.label[item], dtype=torch.long),
            }


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1
    return optimal_value


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      weightsample=0):

    if mode == 'train':
        if weightsample == 1:
            all_label, label_num = dataset.get_classes()
            weights = 1./ torch.tensor(label_num, dtype=torch.float)
            samples_weights = weights[all_label]
            sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
        else:
            sampler = RandomSampler(dataset)

    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )


def read_text(data_path, tokenizer, max_len=512, kw=1, is_test=False, task=1):
    res_data = []
    if task == 1:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                par_id=line.strip().split('\t')[0]
                art_id = line.strip().split('\t')[1]
                keyword=line.strip().split('\t')[2]
                keyword = keyword.replace('-',' ')
                country=line.strip().split('\t')[3]
                text=line.strip().split('\t')[4].lower()
                l=line.strip().split('\t')[5]
                label = int(l)
                cls_id = tokenizer.cls_token_id
                sep_id = tokenizer.sep_token_id
                if kw == 1:
                    pre_id = 'from '+ art_id + ', keyword: ' + keyword +', country: ' + city[country] +', '
                    pre_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pre_id))
                    text_left, _, text_right = [s.lower().strip() for s in text.partition(keyword)]
                    
                    left_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_left))
                    key_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(keyword))
                    right_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_right))

                    while len(left_id) + len(right_id) + len(key_id) > max_len - 2 - len(pre_id):
                        if len(left_id) > len(right_id):
                            left_id = left_id[1:]
                        else:
                            right_id = right_id[:-1]
                    input_id = [cls_id] + pre_id + left_id + key_id + right_id + [sep_id]
                    pos = [len(pre_id) + len(left_id) + 1]
                else:
                    text_left, _, text_right = [s.lower().strip() for s in text.partition(keyword)]
                    left_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_left))
                    key_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(keyword))
                    right_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_right))
                    while len(left_id) + len(right_id) + len(key_id) > max_len - 2:
                        if len(left_id) > len(right_id):
                            left_id = left_id[1:]
                        else:
                            right_id = right_id[:-1]
                            
                    input_id = [cls_id] + left_id + key_id + right_id + [sep_id]
                    pos = [len(left_id) + 1]
                pad_len = max_len - len(input_id)
                attn_mask = [1] * len(input_id) 
                attn_mask += [0] * pad_len
                input_id += [0] * pad_len
                assert len(input_id) == len(attn_mask)
                assert len(input_id) == max_len
                res_data.append({'input_id': input_id, 'attn_mask': attn_mask, 'label': label, 'pos': pos})
                
                '''
                if is_test == False:
                    if len(data) == 1:
                        res_data.append({'text': '', 'label': data[0]})
                    else:
                        res_data.append({'text': data[1], 'label': data[0]})
                else:
                    res_data.append({'text': data[0]})
                 '''
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.strip().split("\t")
                if is_test == False:
                    if len(data) == 1:
                        res_data.append({'text': '', 'label': labels})
                    else:
                        label = data[0].split(',')
                        labels = [int(l) for l in label]
                        res_data.append({'text': data[1], 'label': labels})
                else:
                    res_data.append({'text': data[0]})
    return res_data

def read_text_pair(dpm, mode, task):
    if task == 1:
        if mode == 'train':
            rows = [] # will contain par_id, label and text
            for idx in range(len(trids)):
                parid = trids.par_id[idx]
                # select row from original dataset to retrieve `text` and binary label
                text = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].text.values[0]
                label = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].label.values[0]
                rows.append({
                    'par_id': parid,
                    'text': text,
                    'label': label
                })
            trdf = pd.DataFrame(rows)
            npos = len(trdf[trdf.label==1])
            train_set1 = pd.concat([trdf[trdf.label==1], trdf[trdf.label==0][:npos*2]])
            cnt0, cnt1 = 0, 0
            rows = []
            for idx in range(len(train_set1)):
                text = dpm.train_task1_df.loc[idx].text
                label = dpm.train_task1_df.loc[idx].label
                rows.append({
                    'par_id': parid,
                    'text': text,
                    'label': label
                })
            return rows
        elif mode == 'dev':
            rows = [] # will contain par_id, label and text
            for idx in range(len(teids)):
                parid = teids.par_id[idx]
                # select row from original dataset to retrieve `text` and binary label
                text = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].text.values[0]
                label = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].label.values[0]
                rows.append({
                    'par_id': parid,
                    'text': text,
                    'label': label
                })
    else:
        if mode == 'train':
            rows2 = [] # will contain par_id, label and text
            for idx in range(len(trids)):
                parid = trids.par_id[idx]
                label = trids.label[idx]
                # select row from original dataset to retrieve the `text` value
                text = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].text.values[0]
                rows2.append({
                    'par_id': parid,
                    'text': text,
                    'label': label
                })
        else:
            rows2 = [] # will contain par_id, label and text
            for idx in range(len(teids)):
                parid = teids.par_id[idx]
                label = teids.label[idx]
                # print(parid)
                # select row from original dataset to access the `text` value
                text = dpm.train_task1_df.loc[dpm.train_task1_df.par_id == parid].text.values[0]
                rows2.append({
                    'par_id': parid,
                    'text': text,
                    'label': label
                })
    return rows
    
class DontPatronizeMe:

    def __init__(self, train_path, test_path):

        self.train_path = train_path
        self.test_path = test_path
        self.train_task1_df = None
        self.train_task2_df = None
        self.test_set = None

    def load_task1(self, filename='dontpatronizeme_pcl.tsv'):
        """
		Load task 1 training set and convert the tags into binary labels. 
		Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
		Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
		It returns a pandas dataframe with paragraphs and labels.
		"""
        rows = []
        with open(os.path.join(self.train_path, filename)) as f:
            for line in f.readlines()[4:]:
                par_id = line.strip().split('\t')[0]
                art_id = line.strip().split('\t')[1]
                keyword = line.strip().split('\t')[2]
                country = line.strip().split('\t')[3]
                t = line.strip().split('\t')[4].lower()
                l = line.strip().split('\t')[-1]
                if l == '0' or l == '1':
                    lbin = 0
                else:
                    lbin = 1
                rows.append(
                    {'par_id': par_id,
                     'art_id': art_id,
                     'keyword': keyword,
                     'country': country,
                     'text': t,
                     'label': lbin,
                     'orig_label': l
                     }
                )
        df = pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label'])
        self.train_task1_df = df

    def load_task2(self, return_one_hot=True):
        # Reads the data for task 2 and present it as paragraphs with binarized labels (a list with seven positions, "activated or not (1 or 0)",
        # depending on wether the category is present in the paragraph).
        # It returns a pandas dataframe with paragraphs and list of binarized labels.
        tag2id = {
            'Unbalanced_power_relations': 0,
            'Shallow_solution': 1,
            'Presupposition': 2,
            'Authority_voice': 3,
            'Metaphors': 4,
            'Compassion': 5,
            'The_poorer_the_merrier': 6
        }
        print('Map of label to numerical label:')
        print(tag2id)
        data = defaultdict(list)
        with open(os.path.join(self.train_path, 'dontpatronizeme_categories.tsv')) as f:
            for line in f.readlines()[4:]:
                par_id = line.strip().split('\t')[0]
                art_id = line.strip().split('\t')[1]
                text = line.split('\t')[2].lower()
                keyword = line.split('\t')[3]
                country = line.split('\t')[4]
                start = line.split('\t')[5]
                finish = line.split('\t')[6]
                text_span = line.split('\t')[7]
                label = line.strip().split('\t')[-2]
                num_annotators = line.strip().split('\t')[-1]
                labelid = tag2id[label]
                if not labelid in data[(par_id, art_id, text, keyword, country)]:
                    data[(par_id, art_id, text, keyword, country)].append(labelid)

        par_ids = []
        art_ids = []
        pars = []
        keywords = []
        countries = []
        labels = []

        for par_id, art_id, par, kw, co in data.keys():
            par_ids.append(par_id)
            art_ids.append(art_id)
            pars.append(par)
            keywords.append(kw)
            countries.append(co)

        for label in data.values():
            labels.append(label)

        if return_one_hot:
            labels = MultiLabelBinarizer().fit_transform(labels)
        df = pd.DataFrame(list(zip(par_ids,
                                   art_ids,
                                   pars,
                                   keywords,
                                   countries,
                                   labels)), columns=['par_id',
                                                      'art_id',
                                                      'text',
                                                      'keyword',
                                                      'country',
                                                      'label',
                                                      ])
        self.train_task2_df = df

    def load_test(self):
        # self.test_df = [line.strip() for line in open(self.test_path)]
        rows = []
        with open(self.test_path) as f:
            for line in f.readlines()[4:]:
                t = line.strip().split('\t')[3].lower()
                rows.append(t)
        self.test_set = rows