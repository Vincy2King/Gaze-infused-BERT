# from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from transformers import BertTokenizer,RobertaTokenizer

import numpy as np
import os
from transformers import BertPreTrainedModel, BertModel,RobertaPreTrainedModel,RobertaModel
from TorchCRF import CRF
import timeit
import subprocess
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from matplotlib import pyplot as plt
import datetime
from config import Config as config
import spacy

# tokenizer = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
log_soft = F.log_softmax
import sys
from optparse import OptionParser
import csv
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
n_gpu = torch.cuda.device_count()
print(n_gpu)
# exit(0)
# device='cuda:1'
print(torch.cuda)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# print(device)
# exit(0)

all_token = []
all_ffd = []
all_gd = []
all_gpt = []
all_trt = []
all_nFix = []
all_layer_dict = {}
for i in range(config.num_layers):
    all_layer_dict[i] = []


# to initialize the network weight with fix seed.
def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# seed_torch()

from collections import OrderedDict


# 'FFD', 'GD', 'GPT', 'TRT', 'nFix'
# read the corpus and return them into list of sentences of list of tokens
def corpus_reader(path, delim='\t', word_idx=0, label_idx=6):
    tokens, labels, ffd, gd, gpt, trt, nFix = [], [], [], [], [], [], []
    tmp_tok, tmp_lab, tmp_ffd, tmp_gd, tmp_gpt, tmp_trt, tmp_nFix = [], [], [], [], [], [], []
    label_set = []
    with open(path, 'r') as reader:
        for line in reader:
            line = line.strip()
            # print(line)
            cols = line.split(delim)
            # tmp_tok.append(cols[word_idx])
            # tmp_lab.append(cols[label_idx])
            # label_set.append(cols[label_idx])
            # print(cols)
            if len(cols) < 2:
                # print(tmp_tok)
                if len(tmp_tok) > 0:
                    tokens.append(tmp_tok);
                    labels.append(tmp_lab)
                    ffd.append(tmp_ffd);
                    trt.append(tmp_trt)
                    gd.append(tmp_gd);
                    gpt.append(tmp_gpt)
                    nFix.append(tmp_nFix)
                tmp_tok = []
                tmp_lab = []
                tmp_ffd = []
                tmp_trt = []
                tmp_gd = []
                tmp_gpt = []
                tmp_nFix = []
            else:
                tmp_tok.append(cols[word_idx])
                tmp_lab.append(cols[label_idx])
                label_set.append(cols[label_idx])
                tmp_ffd.append(cols[1])
                tmp_gd.append(cols[2])
                tmp_gpt.append(cols[3])
                tmp_trt.append(cols[4])
                tmp_nFix.append(cols[5])
    return tokens, labels, list(OrderedDict.fromkeys(label_set)), ffd, gd, gpt, trt, nFix


class NER_Dataset(data.Dataset):
    def __init__(self, tag2idx, sentences, labels, ffds, gds, gpts, trts, nFixs, tokenizer_path='', do_lower_case=True):
        self.tag2idx = tag2idx
        self.sentences = sentences
        self.labels = labels
        if config.type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
        if config.type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)

        self.ffds = ffds
        self.gds = gds
        self.gpts = gpts
        self.trts = trts
        self.nFixs = nFixs

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = []
        for x in self.labels[idx]:
            if x in self.tag2idx.keys():
                label.append(self.tag2idx[x])
            else:
                label.append(self.tag2idx['O'])
        ffd = self.ffds[idx]
        gd = self.gds[idx]
        gpt = self.gpts[idx]
        trt = self.trts[idx]
        nFix = self.nFixs[idx]

        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append('[CLS]')
        token_ffd, token_gd, token_gpt, token_trt, token_nFix = [], [], [], [], []
        token_ffd.append(0)
        token_gd.append(0)
        token_gpt.append(0)
        token_trt.append(0)
        token_nFix.append(0)
        # append dummy label 'X' for subtokens
        modified_labels = [self.tag2idx['X']]
        for i, token in enumerate(sentence):
            if len(bert_tokens) >= 512:
                break
            orig_to_tok_map.append(len(bert_tokens))
            modified_labels.append(label[i])
            new_token = self.tokenizer.tokenize(token)
            # new_token = self.tokenizer.encode(token)
            bert_tokens.extend(new_token)
            # print('new_token:',new_token)
            for each in new_token:
                token_ffd.append(float(ffd[i]))
                token_gd.append(float(gd[i]))
                token_gpt.append(float(gpt[i]))
                token_trt.append(float(trt[i]))
                token_nFix.append(float(nFix[i]))
            modified_labels.extend([self.tag2idx['X']] * (len(new_token) - 1))

        bert_tokens.append('[SEP]')
        token_ffd.append(0)
        token_gd.append(0)
        token_gpt.append(0)
        token_trt.append(0)
        token_nFix.append(0)
        modified_labels.append(self.tag2idx['X'])
        # print(bert_tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        if len(token_ids) > 511:
            token_ids = token_ids[:512]
            modified_labels = modified_labels[:512]
        return token_ids, len(
            token_ids), orig_to_tok_map, modified_labels, bert_tokens, token_ffd, token_gd, token_gpt, token_trt, token_nFix  # self.ffds[idx], self.trts[idx]
        # self.sentences[idx], self.ffds[idx], self.trts[idx]


# K折数据划分
def load_data_kfold(sentences, labels, ffd, gd, gpt, trt, nFix, k, n, random_seed):
    print("Cross validation第{}折正在划分数据集".format(n + 1))
    l = len(sentences)
    # print(l)
    shuffle_dataset = True
    indices = list(range(l))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)  # shuffle
    # Collect indexes of samples for validation set.
    val_indices = indices[int(l / k) * n:int(l / k) * (n + 1)]
    train_indices = list(set(indices).difference(set(val_indices)))

    train_sentences = []
    train_labels = []
    train_ffds = []
    train_gds = []
    train_gpts = []
    train_trts = []
    train_nFixs = []

    val_sentences = []
    val_labels = []
    val_ffds = []
    val_gds = []
    val_gpts = []
    val_trts = []
    val_nFixs = []
    for i in range(len(sentences)):
        if i in val_indices:
            val_sentences.append(sentences[i])
            val_labels.append(labels[i])
            val_ffds.append(ffd[i])
            val_gds.append(gd[i])
            val_gpts.append(gpt[i])
            val_trts.append(trt[i])
            val_nFixs.append(nFix[i])
        else:
            train_sentences.append(sentences[i])
            train_labels.append(labels[i])
            train_ffds.append(ffd[i])
            train_gds.append(gd[i])
            train_gpts.append(gpt[i])
            train_trts.append(trt[i])
            train_nFixs.append(nFix[i])
        # print(dataset[i][1])

    return train_sentences, train_labels, train_ffds, train_gds, train_gpts, train_trts, train_nFixs, \
           val_sentences, val_labels, val_ffds, val_gds, val_gpts, val_trts, val_nFixs


def pad(batch):
    # print('pad')
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    tok_ids = do_pad(0, maxlen)
    attn_mask = [[(i > 0) for i in ids] for ids in tok_ids]
    LT = torch.LongTensor
    label = do_pad(3, maxlen)
    sents = do_pad(4, maxlen)
    ffds = do_pad(5, maxlen)
    gds = do_pad(6, maxlen)
    gpts = do_pad(7, maxlen)
    trts = do_pad(8, maxlen)
    nFixs = do_pad(9, maxlen)

    # sort the index, attn mask and labels on token length
    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)
    # print('ffds:',ffds)
    # print('sorted_idx',sorted_idx)
    tok_ids = LT(tok_ids)[sorted_idx]
    # new_ffds, new_trts = [], []
    # for ffd in ffds:
    #     new_ffds.append(float(ffd))
    # for trt in trts:
    #     new_trts.append(float(trt))

    ffds = LT(ffds)  # [sorted_idx]
    gds = LT(gds)
    gpts = LT(gpts)
    trts = LT(trts)  # [sorted_idx]
    nFixs = LT(nFixs)

    attn_mask = LT(attn_mask)  # [sorted_idx]
    labels = LT(label)  # [sorted_idx]
    # sents = torch.tensor(sents)[sorted_idx]
    org_tok_map = get_element(2)
    # sents = get_element(4)

    # ffds = get_element(5)
    # trts = get_element(6)

    return tok_ids, attn_mask, org_tok_map, labels, sents, list(sorted_idx.cpu().numpy()), ffds, gds, gpts, trts, nFixs


class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attn_masks, labels=None):  # dont confuse this with _forward_alg above.
        # print(input_ids.device,attn_masks.device)
        outputs = self.bert(input_ids, attn_masks)
        # print(outputs.device)
        sequence_output = outputs[0]

        attention = outputs[-1]
        # attention.shape = num_layers * bs * num_head * seq_len * seq_len
        # print('attn_masks:',attn_masks)
        print('attention', len(attention[0][0][0][0]), len(attention[0][1][0][0]))
        print('attention', len(attention[0][0][0]), len(attention[0][1][0]))
        # print('num_head:',len(attention[0][0]),attention[0][0])
        num_layers = len(attention)
        print(num_layers)
        # exit(0)
        bs = len(attention[0])
        num_head = len(attention[0][0])
        seq_len = len(attention[0][0][0])
        for i in range(num_layers):
            for j in range(bs):
                for k in range(seq_len):
                    if config.multi_head == 'avg':
                        avg_self = 0
                        for q in range(num_head):
                            # print(type(attention[i][j][q][k][k]))
                            avg_self += attention[i][j][q][k][k].item()
                        avg_self /= num_head
                        print(avg_self)
                        all_layer_dict[i].append(avg_self)
                    elif config.multi_head == 'max':
                        all_layer_dict[i].append(attention[i][j][num_head - 1][k][k].item())

        #             print(attention[i][j][0][k][k][:num])
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction

class RoBerta_CRF(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RoBerta_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attn_masks, labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.bert(input_ids, attn_masks)
        sequence_output = outputs[0]

        attention = outputs[-1]
        # attention.shape = num_layers * bs * num_head * seq_len * seq_len
        # print('attn_masks:',attn_masks)
        print('attention', len(attention[0][0][0][0]), len(attention[0][1][0][0]))
        print('attention', len(attention[0][0][0]), len(attention[0][1][0]))
        # print('num_head:',len(attention[0][0]),attention[0][0])
        num_layers = len(attention)
        print(num_layers)
        # exit(0)
        bs = len(attention[0])
        num_head = len(attention[0][0])
        seq_len = len(attention[0][0][0])
        for i in range(num_layers):
            for j in range(bs):
                for k in range(seq_len):
                    if config.multi_head == 'avg':
                        avg_self = 0
                        for q in range(num_head):
                            # print(type(attention[i][j][q][k][k]))
                            avg_self += attention[i][j][q][k][k].item()
                        avg_self /= num_head
                        print(avg_self)
                        all_layer_dict[i].append(avg_self)
                    elif config.multi_head == 'max':
                        all_layer_dict[i].append(attention[i][j][num_head - 1][k][k].item())

        #             print(attention[i][j][0][k][k][:num])
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction

def generate_training_data(n, config, bert_tokenizer="bert-base", do_lower_case=True):
    training_data, validation_data = config.data_dir + config.training_data, config.data_dir + config.val_data
    sentences, labels, label_set, ffd, gd, gpt, trt, nFix = corpus_reader(training_data, delim='\t')
    train_sentences, train_labels, train_ffds, train_gds, train_gpts, train_trts, train_nFixs, \
    dev_sentences, dev_labels, dev_ffds, dev_gds, dev_gpts, dev_trts, dev_nFixs = load_data_kfold(sentences, labels,
                                                                                                  ffd, gd, gpt, trt,
                                                                                                  nFix, config.k_fold,
                                                                                                  n, config.random_seed)
    # print('==train_sentences===')
    # print(train_sentences)
    # print('===train_labels===')
    # print(train_labels)
    # print("==label_set===")
    # print(label_set)
    label_set.append('X')
    tag2idx = {t: i for i, t in enumerate(label_set)}
    # print('Training datas: ', len(train_sentences))
    train_dataset = NER_Dataset(tag2idx, train_sentences, train_labels, train_ffds, train_gds, train_gpts, train_trts,
                                train_nFixs, tokenizer_path=bert_tokenizer, do_lower_case=do_lower_case)
    # save the tag2indx dictionary. Will be used while prediction
    with open(config.apr_dir + 'tag2idx.pkl', 'wb') as f:
        pickle.dump(tag2idx, f, pickle.HIGHEST_PROTOCOL)
    # dev_sentences, dev_labels, _ = corpus_reader(validation_data, delim='\t')
    dev_dataset = NER_Dataset(tag2idx, dev_sentences, dev_labels, dev_ffds, dev_gds, dev_gpts, dev_trts, dev_nFixs,
                              tokenizer_path=bert_tokenizer, do_lower_case=do_lower_case)

    # print(len(train_dataset))
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return train_iter, eval_iter, tag2idx


def generate_test_data(config, tag2idx, bert_tokenizer="bert-base", do_lower_case=True):
    test_data = config.data_dir + config.test_data
    test_sentences, test_labels, _ = corpus_reader(test_data, delim=' ')
    test_dataset = NER_Dataset(tag2idx, test_sentences, test_labels, tokenizer_path=bert_tokenizer,
                               do_lower_case=do_lower_case)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    return test_iter


def calculate(trans_ffds):
    minn = 999
    maxx = -99
    # print('===========ffd==============')
    print(len(trans_ffds))
    for ffd in trans_ffds:
        # print('float(ffd):',ffd,float(ffd))
        if float(ffd) < minn:
            minn = float(ffd)
        if float(ffd) > maxx:
            maxx = float(ffd)
    # print('min:', minn, 'max:', maxx)
    norm_ffd = []
    for ffd in trans_ffds:
        # print('-', (float(ffd) - minn) / (maxx - minn))
        norm_ffd.append((float(ffd) - minn) / (maxx - minn))
        # all_ffd.append((float(ffd) - minn) / (maxx - minn))
    return norm_ffd


def train(train_iter, eval_iter, tag2idx, config, bert_model="bert-base-uncased"):
    # print('#Tags: ', len(tag2idx))
    unique_labels = list(tag2idx.keys())
    if config.type =='bert':
        model = Bert_CRF.from_pretrained(bert_model, num_labels=len(tag2idx), output_attentions=True)
    else:
        model = RoBerta_CRF.from_pretrained(bert_model, num_labels=len(tag2idx), output_attentions=True)
    model.train()
    # if torch.cuda.is_available():
    model.cuda()
    num_epoch = config.epoch
    gradient_acc_steps = 1
    t_total = len(train_iter) // gradient_acc_steps * num_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    global_step = 0
    model.zero_grad()
    model.train()
    training_loss = []
    validation_loss = []
    train_iterator = trange(num_epoch, desc="Epoch", disable=0)
    start_time = timeit.default_timer()

    for epoch in (train_iterator):
        epoch_iterator = tqdm(train_iter, desc="Iteration", disable=-1)
        tr_loss = 0.0
        tmp_loss = 0.0
        model.train()
        for step, batch in enumerate(epoch_iterator):
            # print('step:',step)
            s = timeit.default_timer()
            # print(batch)
            token_ids, attn_mask, _, labels, sentences, _, ffds, gds, gpts, trts, nFixs = batch
            # print(labels)
            print(device)
            inputs = {'input_ids': token_ids.to(device),
                      'attn_masks': attn_mask.to(device),
                      'labels': labels.to(device)
                      }
            print('sentences:', len(sentences))  # ,sentences)
            # print(len(sentences[0]))

            for i in range(len(sentences)):
                total = 0
                print(len(sentences[i]))
                for j in range(len(sentences[i])):
                    all_token.append(sentences[i][j])

                norm_ffd = calculate(ffds[i])
                for each in norm_ffd:
                    all_ffd.append(each)
                norm_gd = calculate(gds[i])
                for each in norm_gd:
                    all_gd.append(each)
                norm_gpt = calculate(gpts[i])
                for each in norm_gpt:
                    all_gpt.append(each)
                norm_trt = calculate(trts[i])
                for each in norm_trt:
                    all_trt.append(each)
                norm_nFix = calculate(nFixs[i])
                for each in norm_nFix:
                    all_nFix.append(each)
                '''
                minn = 999
                maxx = -99
                print('len(trts[i]):',len(trts[i]))
                for trt in trts[i]:
                    if float(trt) < minn:
                        minn = float(trt)
                    if float(trt) > maxx:
                        maxx = float(trt)
                norm_trt = []
                # print('===========trt==============')
                # print('min:', minn, 'max:', maxx)
                for trt in trts[i]:
                    # print('-', (float(trt) - minn) / (maxx - minn))
                    norm_trt.append((float(trt) - minn) / (maxx - minn))
                    all_trt.append((float(trt) - minn) / (maxx - minn))
                '''
            loss = model(**inputs)
            loss.backward()
            tmp_loss += loss.item()
            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            if step == 0:
                print('\n%s Step: %d of %d Loss: %f' % (
                str(datetime.datetime.now()), (step + 1), len(epoch_iterator), loss.item()))
            if (step + 1) % 100 == 0:
                print('%s Step: %d of %d Loss: %f' % (
                str(datetime.datetime.now()), (step + 1), len(epoch_iterator), tmp_loss / 1000))
                tmp_loss = 0.0

        # log_path = 'ZuCo1_multi_attn.csv'
        file = open(config.log_path, 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        if config.num_layers == 12:
            name_list = ['sentence', 'layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'layer 6'
                , 'layer 7', 'layer 8', 'layer 9', 'layer 10', 'layer 11', 'layer 12', 'FFD', 'GD', 'GPT', 'TRT', 'nFix']
            csv_writer.writerow(name_list)
            for i in range(len(all_token)):
                csv_writer.writerow(
                    [all_token[i], all_layer_dict[0][i], all_layer_dict[1][i], all_layer_dict[2][i], all_layer_dict[3][i],
                     all_layer_dict[4][i], all_layer_dict[5][i], all_layer_dict[6][i], all_layer_dict[7][i],
                     all_layer_dict[8][i],
                     all_layer_dict[9][i], all_layer_dict[10][i], all_layer_dict[11][i], all_ffd[i], all_gd[i], all_gpt[i],
                     all_trt[i], all_nFix[i]])
        else:
            name_list = ['sentence', 'layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'layer 6'
                , 'layer 7', 'layer 8', 'layer 9', 'layer 10', 'layer 11', 'layer 12',
                         'layer 13', 'layer 14', 'layer 15', 'layer 16'
                , 'layer 17', 'layer 18', 'layer 19', 'layer 20', 'layer 21', 'layer 22','layer 23', 'layer 24', 'FFD', 'GD', 'GPT', 'TRT', 'nFix']
            csv_writer.writerow(name_list)
            for i in range(len(all_token)):
                csv_writer.writerow(
                    [all_token[i], all_layer_dict[0][i], all_layer_dict[1][i], all_layer_dict[2][i], all_layer_dict[3][i],
                     all_layer_dict[4][i], all_layer_dict[5][i], all_layer_dict[6][i], all_layer_dict[7][i],
                     all_layer_dict[8][i],
                     all_layer_dict[9][i], all_layer_dict[10][i], all_layer_dict[11][i],
                     all_layer_dict[12][i], all_layer_dict[13][i],
                     all_layer_dict[14][i], all_layer_dict[15][i], all_layer_dict[16][i], all_layer_dict[17][i],
                     all_layer_dict[18][i],
                     all_layer_dict[19][i], all_layer_dict[20][i], all_layer_dict[21][i],
                     all_layer_dict[22][i], all_layer_dict[23][i],
                     all_ffd[i], all_gd[i], all_gpt[i],
                     all_trt[i], all_nFix[i]])
        exit(0)
        print("Training Loss: %f for epoch %d" % (tr_loss / len(train_iter), epoch))
        training_loss.append(tr_loss / len(train_iter))
        # '''
        # Y_pred = []
        # Y_true = []
        val_loss = 0.0
        model.eval()
        writer = open(config.apr_dir + 'prediction_' + str(epoch) + '.csv', 'w')
        for i, batch in enumerate(eval_iter):
            token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx, ffds, trts = batch
            # attn_mask.dt
            inputs = {'input_ids': token_ids.to(device),
                      'attn_masks': attn_mask.to(device)
                      }

            dev_inputs = {'input_ids': token_ids.to(device),
                          'attn_masks': attn_mask.to(device),
                          'labels': labels.to(device)
                          }
            with torch.torch.no_grad():
                tag_seqs = model(**inputs)
                tmp_eval_loss = model(**dev_inputs)
            val_loss += tmp_eval_loss.item()
            # print(labels.numpy())
            y_true = list(labels.cpu().numpy())
            for i in range(len(sorted_idx)):
                o2m = org_tok_map[i]
                pos = sorted_idx.index(i)
                for j, orig_tok_idx in enumerate(o2m):
                    writer.write(original_token[i][j] + '\t')
                    writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                    pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                    if pred_tag == 'X':
                        pred_tag = 'O'
                    writer.write(pred_tag + '\n')
                writer.write('\n')

        validation_loss.append(val_loss / len(eval_iter))
        writer.flush()
        print('Epoch: ', epoch)
        command = "python conlleval.py < " + config.apr_dir + "prediction_" + str(epoch) + ".csv"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        result = process.communicate()[0].decode("utf-8")
        print(result)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss / len(train_iter),
        }, config.apr_dir + 'model_' + str(epoch) + '.pt')

    total_time = timeit.default_timer() - start_time
    print('Total training time: ', total_time)
    return training_loss, validation_loss


'''
    raw_text should pad data in raw data prediction
'''


def test(config, test_iter, model, unique_labels, test_output):
    model.eval()
    writer = open(config.apr_dir + test_output, 'w')
    for i, batch in enumerate(test_iter):
        token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx = batch
        # attn_mask.dt
        inputs = {'input_ids': token_ids.to(device),
                  'attn_masks': attn_mask.to(device)
                  }
        with torch.torch.no_grad():
            tag_seqs = model(**inputs)
        y_true = list(labels.cpu().numpy())
        for i in range(len(sorted_idx)):
            o2m = org_tok_map[i]
            pos = sorted_idx.index(i)
            for j, orig_tok_idx in enumerate(o2m):
                writer.write(original_token[i][j] + '\t')
                writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                if pred_tag == 'X':
                    pred_tag = 'O'
                writer.write(pred_tag + '\n')
            writer.write('\n')
    writer.flush()
    command = "python conlleval.py < " + config.apr_dir + test_output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    result = process.communicate()[0].decode("utf-8")
    print(result)


def parse_raw_data(padded_raw_data, model, unique_labels, out_file_name='raw_prediction.csv'):
    model.eval()
    token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx = padded_raw_data
    # attn_mask.dt
    writer = open(out_file_name, 'w')
    inputs = {'input_ids': token_ids.to(device),
              'attn_masks': attn_mask.to(device)
              }
    with torch.torch.no_grad():
        tag_seqs = model(**inputs)
    y_true = list(labels.cpu().numpy())
    for i in range(len(sorted_idx)):
        o2m = org_tok_map[i]
        pos = sorted_idx.index(i)
        for j, orig_tok_idx in enumerate(o2m):
            writer.write(original_token[i][j] + '\t')
            writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
            pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
            if pred_tag == 'X':
                pred_tag = 'O'
            writer.write(pred_tag + '\n')
        writer.write('\n')
    print("Raw data prediction done!")


def show_graph(training_loss, validation_loss, resource_dir):
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
    plt.plot(range(1, len(training_loss) + 1), validation_loss, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Loss Vs Testing Loss")
    plt.legend()
    plt.show()
    plt.savefig(resource_dir + 'Loss.png')


def load_model(config, do_lower_case=True):
    f = open(config.apr_dir + 'tag2idx.pkl', 'rb')
    tag2idx = pickle.load(f)
    unique_labels = list(tag2idx.keys())
    if config.type == 'bert':
        model = Bert_CRF.from_pretrained(bert_model, num_labels=len(tag2idx), output_attentions=True)
    else:
        model = RoBerta_CRF.from_pretrained(bert_model, num_labels=len(tag2idx), output_attentions=True)

    checkpoint = torch.load(config.apr_dir + config.model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    global bert_tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=do_lower_case)
    # if torch.cuda.is_available():
    model.cuda()
    model.eval()
    return model, bert_tokenizer, unique_labels, tag2idx


def raw_processing(doc, bert_tokenizer, word_tokenizer):
    tic = time.time()
    spans = re.split("[\n\r]", doc)
    offset = 0
    batch = []
    for span in spans:
        sentences = sentence_segmenter(span)
        for s_idx, sentence in enumerate(sentences.sents):
            bert_tokens = []
            orig_to_tok_map = []
            bert_tokens.append('[CLS]')
            begins = []
            ends = []
            for tok in tokenzer(word):
                token = tok.text
                offset = doc.find(token, offset)
                current_begins.append(offset)
                ends.append(offset + len(token))
                offset += len(token)
                orig_to_tok_map.append(len(bert_tokens))
                new_token = bert_tokenizer.tokenize(token)
                bert_tokens.extend(new_token)
                print('new_token:', new_token)
            bert_tokens.append('[SEP]')
            token_id = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
            if len(token_id) > 511:
                token_id = token_id[:512]
            dummy_labels = ['X'] * len(token_id)
            dummy_f_names = ['f_names'] * len(token_id)
            sample = (token_id, len(token_id), orig_to_tok_map, dummy_labels, original_token)
            batch.append(sample)
    pad_data = pad(batch)
    return pad_data


def usage(parameter):
    parameter.print_help()
    print("Example usage (training):\n", \
          "\t python bert_crf.py --mode train ")

    print("Example usage (testing):\n", \
          "\t python bert_crf.py --mode test ")


if __name__ == "__main__":
    user_input = OptionParser()
    user_input.add_option("--mode", dest="model_mode", metavar="string", default='train',
                          help="mode of the model (required)")
    (options, args) = user_input.parse_args()

    if options.model_mode == "train":
        for i in range(config.k_fold):
            train_iter, eval_iter, tag2idx = generate_training_data(i, config=config, bert_tokenizer=config.bert_model,
                                                                    do_lower_case=True)
            t_loss, v_loss = train(train_iter, eval_iter, tag2idx, config=config, bert_model=config.bert_model)
            show_graph(t_loss, v_loss, config.apr_dir)
    elif options.model_mode == "test":
        model, bert_tokenizer, unique_labels, tag2idx = load_model(config=config, do_lower_case=True)
        test_iter = generate_test_data(config, tag2idx, bert_tokenizer=config.bert_model, do_lower_case=True)
        print('test len: ', len(test_iter))
        test(config, test_iter, model, unique_labels, config.test_out)
    elif options.model_mode == "raw_text":
        if config.raw_text == None:
            print('Please provide the raw text path on config.raw_text')
            import sys

            sys.exit(1)
        model, bert_tokenizer, unique_labels, tag2idx = load_model(config=config, do_lower_case=True)
        doc = open(config.raw_text).read()
        pad_data = raw_processing(doc, bert_tokenizer)
        parse_raw_data(pad_data, model, unique_labels, out_file_name=config.raw_prediction_output)
    else:
        usage(user_input)
