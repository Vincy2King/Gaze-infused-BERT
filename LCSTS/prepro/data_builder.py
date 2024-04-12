import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from tqdm import tqdm
from os.path import join as pjoin

import torch
from multiprocess import Pool
from pytorch_pretrained_bert import BertTokenizer

from utils.logging import logger
from prepro.utils import _get_word_ngrams

SPLIT_TOKEN = re.compile(r'[。，；]|[!?！？]+')


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = _rouge_clean(' '.join(abstract_sent_list)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = _rouge_clean(' '.join(abstract_sent_list)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('/data10T/wangbb/bert-large-chinese', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src, tgt, src_gaze, tgt_gaze, oracle_ids):
        # print('src:',src)
        if (len(src) == 0):
            logger.info(f'(len = 0) Remove ({src}, {tgt}).')
            return None

        original_src_txt = src

        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            logger.info(f'(len < {self.args.min_nsents}) Remove ({src}, {tgt}).')
            return None
        if (len(labels) == 0):
            logger.info(f'(labels = 0) Remove ({src}, {tgt}).')
            return None
        if (sum(labels) == 0):
            logger.info(f'(all label = 0) Remove ({src}, {tgt}).')
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        # print('src_subtokens 1:',src_subtokens)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        # print('src_subtokens 2:', src_subtokens)
        # exit()
        src_gaze = src_gaze.split('\t')
        tgt_gaze = tgt_gaze.split('\t')

        src_nfix = eval(src_gaze[1])
        src_ffd = eval(src_gaze[2])
        src_gpt = eval(src_gaze[3])
        src_trt = eval(src_gaze[4])
        src_gd =eval(src_gaze[5])
        src_gaze = [src_nfix, src_ffd,src_gpt,src_trt,src_gd]

        tgt_nfix = eval(tgt_gaze[1])
        tgt_ffd = eval(tgt_gaze[2])
        tgt_gpt = eval(tgt_gaze[3])
        tgt_trt = eval(tgt_gaze[4])
        tgt_gd = eval(tgt_gaze[5])
        tgt_gaze = [tgt_nfix, tgt_ffd, tgt_gpt, tgt_trt, tgt_gd]
        # print('src_gaze:',type(src_gaze))
        # print('src_ffd:',src_ffd)
        # exit()

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = tgt
        src_txt = [original_src_txt[i] for i in idxs]

        # print('tgt:',len(tgt_txt),len(tgt_ffd),tgt_txt)
        # print('src',len(src_txt),len(src_ffd),src_txt)

        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt,src_gaze,tgt_gaze

    def convert_format(self, src):
        if (len(src) == 0):
            raise RuntimeError('Input has empty segment.')

        original_src_txt = src

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        src = src[:self.args.max_nsents]

        src_txt = [' '.join(sent) for sent in src]

        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        # print(src_subtokens)
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, segments_ids, cls_ids, src_txt


def format_to_bert(args, result):
    def _white_clean(s):
        return re.sub(r'\s', '', s)

    print("Preparing to format to bert...")

    tgt_files = {'train': args.train_tgt_path, 'valid': args.valid_tgt_path, 'test': args.test_tgt_path}
        # ,'train_gaze': args.gaze_train_tgt_path, 'valid_gaze': args.gaze_valid_tgt_path, 'test_gaze': args.gaze_test_tgt_path}
    for corpus_type in args.dataset:
        # get src and tgt
        srcs = result[corpus_type]
        with open(tgt_files[corpus_type], 'r', encoding='utf-8') as f:
            tgts = f.readlines()
        srcs_gaze = result[corpus_type + '_src_gaze'] # train_src_path
        tgts_gaze = result[corpus_type + '_tgt_gaze']
        # print('tgt_files[corpus_type]:',tgt_files[corpus_type],corpus_type,
        #       len(srcs),len(tgts),len(srcs_gaze),len(tgts_gaze))
        print('len:',len(srcs))
        # exit()
        bert = BertData(args)

        pt_result = []
        # num=0
        for src, tgt, src_gaze, tgt_gaze in tqdm(zip(srcs, tgts,srcs_gaze,tgts_gaze), total=len(srcs), desc=f'    [{corpus_type} formating]'):
            # print(num)
            # num+=1
            tgt = _white_clean(tgt)
            if (args.oracle_mode == 'greedy'):
                oracle_ids = greedy_selection(src, tgt, 3)
            elif (args.oracle_mode == 'combination'):
                oracle_ids = combination_selection(src, tgt, 3)
            b_data = bert.preprocess(src, tgt, src_gaze, tgt_gaze, oracle_ids)
            # print('b_data:',b_data)
            if b_data is None:
                continue
            indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, src_g, tgt_g = b_data
            b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                            'src_txt': src_txt, "tgt_txt": tgt_txt,
                           'src_gaze':src_g,'tgt_gaze':tgt_g}
                           # 'ffds':ffds,'trts':trts,'nfixs':nfixs,'gpts':gpts,'gds':gds}
            pt_result.append(b_data_dict)
            
        print(f"    Complete format {corpus_type} set containing {len(pt_result)} pairs.")
        # exit()
        torch.save(pt_result, f'large_data/LCSTS_{corpus_type}.pt')
        
        if args.save_temp:
            path_dir = 'oracle_result'
            os.makedirs(path_dir, exist_ok=True)
            with open(os.path.join(path_dir, f'{corpus_type}_src.oracle'), 'w', encoding='utf-8') as f:
                for line in pt_result:
                    f.write(' '.join(line['src_txt']) + '\n')
            with open(os.path.join(path_dir, f'{corpus_type}_tgt.oracle'), 'w', encoding='utf-8') as f:
                for line in pt_result:
                    f.write(line['tgt_txt'] + '\n')
            with open(os.path.join(path_dir, f'{corpus_type}_lab.oracle'), 'w', encoding='utf-8') as f:
                for line in pt_result:
                    f.write(' '.join(str(line['labels'])) + '\n')
            print(f'Successfully saved results to {path_dir}')

    print("Successfully finished formating.\n")


def tokenize(args):
    def _white_clean(s):
        return re.sub(r'\s', '', s)

    datasets = ['train', 'valid', 'test',
                'train_src_gaze','valid_src_gaze','test_src_gaze',
                'train_tgt_gaze','valid_tgt_gaze','test_tgt_gaze']
    paths = [args.train_src_path, args.valid_src_path, args.test_src_path,
             args.gaze_train_src_path, args.gaze_valid_src_path, args.gaze_test_src_path,
             args.gaze_train_tgt_path, args.gaze_valid_tgt_path, args.gaze_test_tgt_path]
    # gaze_paths = [args.gaze_train_src_path, args.gaze_valid_src_path, args.gaze_test_src_path]
    result = {'train': [], 'valid': [], 'test': [],'train_src_gaze':[],'valid_src_gaze':[],'test_src_gaze':[],
              'train_tgt_gaze':[],'valid_tgt_gaze':[],'test_tgt_gaze':[]}

    token = re.compile(r'[。，；]|[!?！？]+')

    print("Preparing to tokenize...")
    for dataset, path in zip(datasets, paths):
        # if not dataset in args.dataset:
        #     continue

        # read file
        # num = 0
        with open(path, 'r', encoding='utf-8') as f:
            # if num > 100:
            #     break
            # num+=1
            lines = f.readlines()
        length = len(lines)//30
        # print(len(lines),length)
        # exit()
        # sentences split
        if 'gaze' in dataset:
            num = 0
            # print('gaze dataset')
            # print(len(lines))
            # exit()
            for line in tqdm(lines, desc=f'    [{dataset} tokenizing]'):
            # for line in lines:
            #     print(len(eval(line.split('\t')[1])),len(eval(line.split('\t')[2])),
            #           len(eval(line.split('\t')[3])),len(eval(line.split('\t')[4])),
            #           len(eval(line.split('\t')[5])))
            #     if 'train' in dataset:
            #         if num > length:
            #             break
            #     num+=1
                print('gaze num:',num)
                result[dataset].append(line)
            # exit()
        else:
            num = 0
            for line in tqdm(lines, desc=f'    [{dataset} tokenizing]'):
            # for line in lines:
                # if num > 100:
                #     break
                print(dataset,num)
                if 'train' in dataset:
                    if num > length:
                        break

                num+=1
                print('num:',num)
                splits = re.split(SPLIT_TOKEN, line)
                # print('_white_clean(sent):',[_white_clean(sent) for sent in splits])
                result[dataset].append([_white_clean(sent) for sent in splits])
            # exit()
        print(f"    Complete spliting {dataset} set containing {len(lines)} documents.")

    if args.save_temp:
        path_dir = 'token_result'
        os.makedirs(path_dir, exist_ok=True)
        for dataset in datasets:
            if not dataset in args.dataset:
                continue
            with open(os.path.join(path_dir, f'{dataset}.split'), 'w', encoding='utf-8') as f:
                for line in result[dataset]:
                    f.write('<split>'.join(line) + '\n')
        print(f'Successfully saved results to {path_dir}')
    print("Successfully finished tokenizing.\n")

    return result
