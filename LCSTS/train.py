import os
import random
import argparse
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import BertConfig#,\
    # BertTokenizer, RobertaConfig,RobertaTokenizer,\
    # AdamW
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam,  WarmupLinearSchedule

from utils.logging import logger, init_logger
from utils.utils import compute_files_ROUGE
from summarizer.model_builder import Summarizer

import torch
import torchvision
from pytorch_model_summary import summary

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask, label_id,
                 nfixs, ffds, trts, gpts, gds):
        self.sent_idx = sent_idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.clss_ids = clss_ids
        self.clss_mask = clss_mask
        self.label_id = label_id
        self.nfixs = nfixs
        self.ffds = ffds
        self.trts = trts
        self.gpts = gpts
        self.gds = gds

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_examples_to_features(examples):
    """Loads a data file into a list of `InputBatch`s."""

    max_seq_length = max([len(e['src']) for e in examples])
    max_cls_lenght = max([len(e['clss']) for e in examples])

    features = []
    for (ex_index, example) in enumerate(examples):
        input_ids = example['src']
        segment_ids = example['segs']
        clss_ids = example['clss']
        label_id = example['labels']
        src_gazes = example['src_gaze']
        tgt_gazes = example['tgt_gaze']

        src_nfix = src_gazes[0]
        src_ffd = src_gazes[1]
        src_gpt = src_gazes[2]
        src_trt = src_gazes[3]
        src_gd = src_gazes[4]
        tgt_nfix, tgt_ffd, tgt_gpt, tgt_trt, tgt_gd = tgt_gazes[0], tgt_gazes[1], \
                                                      tgt_gazes[2], tgt_gazes[3], tgt_gazes[4]
        # print( len(src_nfix), len(src_ffd), len(src_trt),
        #       len(src_gpt), len(src_gd), len(input_ids))
        src_ffd = src_ffd[:-2]
        src_nfix = src_nfix[:-2]
        src_gpt = src_gpt[:-2]
        src_trt = src_trt[:-2]
        src_gd = src_gd[:-2]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # print('input id:',len(input_ids),input_ids) # 比src少了两个
        # print('src ffd:', len(src_ffd),src_ffd[:-2])
        # exit()
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        gaze_padding = [0]*(max_seq_length-len(src_nfix))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        src_nfix += gaze_padding
        src_ffd += gaze_padding
        src_trt += gaze_padding
        src_gpt += gaze_padding
        src_gd += gaze_padding
        if len(src_nfix)>max_seq_length:
            src_nfix = src_nfix[:max_seq_length]
            src_ffd = src_ffd[:max_seq_length]
            src_trt = src_trt[:max_seq_length]
            src_gpt = src_gpt[:max_seq_length]
            src_gd = src_gd[:max_seq_length]

        # print(len(padding),len(src_nfix),len(src_ffd),len(src_trt),
        #       len(src_gpt),len(src_gd),len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(src_nfix) == max_seq_length
        assert len(src_ffd) == max_seq_length
        assert len(src_trt) == max_seq_length
        assert len(src_gpt) == max_seq_length
        assert len(src_gd) == max_seq_length

        clss_mask = [1] * len(clss_ids)
        clss_padding = [0] * (max_cls_lenght - len(clss_ids))
        clss_ids += clss_padding
        clss_mask += clss_padding
        label_id += clss_padding

        assert len(clss_ids) == max_cls_lenght
        assert len(clss_mask) == max_cls_lenght
        assert len(label_id) == max_cls_lenght

        features.append(
                InputFeatures(sent_idx=ex_index,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              clss_ids=clss_ids,
                              clss_mask=clss_mask,
                              label_id=label_id,
                              nfixs = src_nfix,
                              ffds = src_ffd,
                              trts = src_trt,
                              gpts = src_gpt,
                              gds = src_gd))
    return features


def get_dataset(features):
    """Pack the features into dataset"""
    all_sent_idx = torch.tensor([f.sent_idx for f in features], dtype=torch.int)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_clss_ids = torch.tensor([f.clss_ids for f in features], dtype=torch.long)
    all_clss_mask = torch.tensor([f.clss_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    all_nFix = torch.tensor([f.nfixs for f in features], dtype=torch.long)
    all_FFD = torch.tensor([f.ffds for f in features], dtype=torch.long)
    all_GPT = torch.tensor([f.trts for f in features], dtype=torch.long)
    all_TRT = torch.tensor([f.gpts for f in features], dtype=torch.long)
    all_GD = torch.tensor([f.gds for f in features], dtype=torch.long)
    
    return TensorDataset(all_sent_idx, all_input_ids, all_input_mask, all_segment_ids, all_clss_ids, all_clss_mask, all_label_ids,
                         all_nFix,all_FFD,all_GPT,all_TRT,all_GD)


def accuracy(out, labels, mask):
    outputs = [(i == j) * m for i, j, m in zip(out, labels, mask)]
    count = sum(map(sum, mask))
    return sum(map(sum, outputs)) / count


from rouge import Rouge


def rouge_scores(ref, pred):
    rouge = Rouge()

    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for r, p in zip(ref, pred):
        scores = rouge.get_scores(hyps=p, refs=r)

        rouge_1_scores.append(scores[0]['rouge-1']['f'])
        rouge_2_scores.append(scores[0]['rouge-2']['f'])
        rouge_l_scores.append(scores[0]['rouge-l']['f'])

    rouge_1_scores = sum(rouge_1_scores)/len(rouge_1_scores)
    rouge_2_scores = sum(rouge_2_scores)/len(rouge_2_scores)
    rouge_l_scores = sum(rouge_l_scores)/len(rouge_l_scores)
    print(rouge_l_scores)
    return rouge_1_scores, rouge_2_scores, rouge_l_scores

def select_sent(sent_idx, examples, scores, mask):
    """Find the prediction, gold, and selected ids."""
    sent_scores = scores + mask
    selected_ids = np.argsort(-sent_scores, 1) # [batch size, clss length, 1]

    gold = []
    pred = []
    result = np.zeros_like(sent_scores)

    for i, idx in enumerate(selected_ids):
        _pred = []
        _idx = []

        # get example data
        example = examples[sent_idx[i]]
        src_str = example['src_txt']
        tgt_str = example['tgt_txt']

        # if(len(src_str) == 0):
        #     continue
        for j in selected_ids[i][:len(src_str)]:
            if(j >= len(src_str)):
                continue
            candidate = src_str[j].strip()
            _pred.append(candidate)
            _idx.append(j)

            if len(_pred) >= 3:
                break

        _pred = ' '.join(_pred)

        pred.append(_pred)
        gold.append(tgt_str)
        result[i, _idx] = 1
    
    return pred, gold, result


def baseline(args):
    """
    Compute the ROUGE score of oracle summaries.
    """
    dataset = torch.load(args.test_data)
    all_src = [f['src_txt'] for f in dataset]
    all_ref = [f['tgt_txt'] for f in dataset]
    all_label = [f['labels'] for f in dataset]
    eval_data = zip(all_src, all_ref, all_label)

    refs = []
    preds = []
    # extract oracle summaries
    for src, ref, label in tqdm(eval_data, total=len(all_src), desc='[Oracle summaries generating] '):
        oracles = [src[i] for i, l in enumerate(label) if l == 1]

        refs.append(ref)
        preds.append(' '.join(oracles))

    # output results
    logger.info(f"Store the oracle result...")
    path_dir = args.result_path
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, 'targets.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(refs))
    with open(os.path.join(path_dir, 'preds.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(preds))

    compute_files_ROUGE(args, os.path.join(path_dir, 'targets.txt'), os.path.join(path_dir, 'preds.txt'))


def train(args):
    # initial device and gpu number
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # get train example
    train_examples = None
    num_train_optimization_steps = None

    print('[Train] Load data pt file...')
    train_examples = torch.load(args.train_data)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    print('[Train] Finish loading data pt file...')

    # Prepare model
    print('[Train] Initial model...')
    if args.bert_config:
        model_file = os.path.join(args.bert_config, WEIGHTS_NAME)
        config_file = os.path.join(args.bert_config, CONFIG_NAME)
        print('[Test] Load model...')
        # config = BertConfig.from_json_file(config_file)
        config = BertConfig.from_pretrained('/data10T/wangbingbing/bert-large')  # from_json_file(config_file)

        config.attention_probs_dropout_prob = 0.05
        config.hidden_dropout_prob = 0.05
        config.use_structural_loss = args.use_structural_loss
        config.struct_loss_coeff = args.struct_loss_coeff
        config.num_syntactic_heads = args.num_syntactic_heads
        config.syntactic_layers = args.syntactic_layers
        config.max_syntactic_distance = args.max_syntactic_distance
        config.eye_max_syntactic_distance = args.eye_max_syntactic_distance
        config.device = args.device
        config.eye_length = 20000
        # config.device = args.device
        config.revise_gat = args.revise_gat
        # print('config:',config.syntactic_layers)
        model = Summarizer(args, device, load_pretrained_bert=True, bert_config=config)
        # load check points
        # model.load_cp(torch.load(model_file))
    else:
        model = Summarizer(args, device, load_pretrained_bert=True)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Total - ', pytorch_total_params)
    print('Trainable - ', trainable_pytorch_total_params)
    # exit()
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    
    # prepare training data
    train_features = convert_examples_to_features(train_examples)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    train_data = get_dataset(train_features)
    # print(len(train_data))
    # exit()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # prepare evaluation data
    eval_dataloader = None
    eval_path = args.test_data
    if os.path.isfile(eval_path):
        logger.info(f"***** Prepare evaluation data from {eval_path} *****")

        eval_examples = torch.load(eval_path)
        eval_features = convert_examples_to_features(eval_examples)
        eval_data = get_dataset(eval_features)
        
        logger.info("  Num examples = %d", len(eval_examples))
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    global_step = 0
    tr_loss = 0
    
    loss_f = torch.nn.BCELoss(reduction='none')
    best_performance = 1000000
    best_epoch = 0
    output_model_file = os.path.join(args.model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(args.model_path, CONFIG_NAME)
    # begin training
    for epoch_i in range(int(args.num_train_epochs)):
        print('[ Epoch', epoch_i, ']')
        rouge_1, rouge_2, rouge_l = 0,0,0
        # training
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask, label_ids,\
                nfixs, ffds, gpts, trts, gds = batch

            sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask,
                                        ffds=ffds,
                                        gds=gds,
                                        gpts=gpts,
                                        trts=trts,
                                        nfixs=nfixs)
            sent_scores_new = sent_scores.detach().cpu().numpy()
            mask_new = mask.detach().cpu().numpy()
            _pred, _ref, selected_ids = select_sent(sent_idx, train_examples, sent_scores_new, mask_new)
            rouge_1_scores, rouge_2_scores, rouge_l_scores = rouge_scores(_ref, _pred)
            # print('_pred:', _pred)
            # print('_ref:', _ref)
            print("Train ROUGE-1 Scores:", rouge_1_scores)
            print("Train ROUGE-2 Scores:", rouge_2_scores)
            print("Train ROUGE-L Scores:", rouge_l_scores)
            rouge_l+=rouge_l_scores
            rouge_1+=rouge_1_scores
            rouge_2+=rouge_2_scores

            loss = loss_f(sent_scores, label_ids.float())
            loss = (loss * mask.float()).sum()
            loss = loss/loss.numel()
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * WarmupLinearSchedule(global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        print("Epoch ",epoch_i," ROUGE-1 Scores:", rouge_1/args.train_batch_size)
        print("Epoch ",epoch_i," ROUGE-2 Scores:", rouge_2/args.train_batch_size)
        print("Epoch ",epoch_i," ROUGE-L Scores:", rouge_l/args.train_batch_size)
        train_loss = tr_loss / nb_tr_steps
        cur_loss = train_loss

        # evaluation
        if eval_dataloader and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask, label_ids,\
                    nfixs, ffds, gpts, trts, gds in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                clss_ids = clss_ids.to(device)
                clss_mask = clss_mask.to(device)
                label_ids = label_ids.to(device)
                nfixs = nfixs.to(device)
                ffds = ffds.to(device)
                gpts = gpts.to(device)
                trts = trts.to(device)
                gds = gds.to(device)

                with torch.no_grad():
                    sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask,
                                        ffds=ffds,
                                        gds=gds,
                                        gpts=gpts,
                                        trts=trts,
                                        nfixs=nfixs)

                    tmp_eval_loss = loss_f(sent_scores, label_ids.float())
                    tmp_eval_loss = (tmp_eval_loss * mask.float()).sum()
                    tmp_eval_loss = tmp_eval_loss/tmp_eval_loss.numel()

                sent_scores = sent_scores.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                # find selected sentences
                _pred, _ref, selected_ids = select_sent(sent_idx, eval_examples, sent_scores, mask)

                # print('_pred:',_pred)
                # print('_ref:',_ref)
                tmp_eval_accuracy = accuracy(selected_ids, label_ids, mask)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            cur_loss = eval_loss
            
        # output result of an epoch
        print(f'  - (Training)   loss: {train_loss: 8.5f}')
        print(f'  - (Validation) loss: {eval_loss: 8.5f}, accuracy: {100 * eval_accuracy: 3.3f} %')
        
        # record best model
        if cur_loss < best_performance:
            best_performance = cur_loss
            best_epoch = epoch_i
            # Save a trained model and the associated configuration
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            logger.info("Successfully save the best model.")


def test(args):
    # initial device and gpu number
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model_file = os.path.join(args.model_path, WEIGHTS_NAME)
    config_file = os.path.join(args.model_path, CONFIG_NAME)
    # Prepare model
    print('[Test] Load model...')
    config = BertConfig.from_json_file(config_file)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    if args.fp16:
        model.half()
    model.to(device)
    # load check points
    model.load_cp(torch.load(model_file))
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare testing data
    eval_dataloader = None
    eval_path = args.test_data
    logger.info(f"***** Prepare testing data from {eval_path} *****")

    eval_examples = torch.load(eval_path)
    eval_features = convert_examples_to_features(eval_examples)
    eval_data = get_dataset(eval_features)
    
    logger.info("  Num examples = %d", len(eval_examples))
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # testing
    loss_f = torch.nn.BCELoss(reduction='none')
    refs, preds = [], []
    if eval_dataloader and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for sent_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            clss_ids = clss_ids.to(device)
            clss_mask = clss_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask)

                tmp_eval_loss = loss_f(sent_scores, label_ids.float())
                tmp_eval_loss = (tmp_eval_loss * mask.float()).sum()
                tmp_eval_loss = tmp_eval_loss/tmp_eval_loss.numel()

            sent_scores = sent_scores.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            # find selected sentences
            _pred, _ref, selected_ids = select_sent(sent_idx, eval_examples, sent_scores, mask)
            preds.extend(_pred)
            refs.extend(_ref)
            # print('preds',preds)
            # print('ref:',refs)
            tmp_eval_accuracy = accuracy(selected_ids, label_ids, mask)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
            
        # print result
        print(f'  - (Testing) loss: {eval_loss: 8.5f}, accuracy: {100 * eval_accuracy: 3.3f} %')

    # output results
    path_dir = args.result_path
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, 'targets.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(refs))
    with open(os.path.join(path_dir, 'preds.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(preds))

    compute_files_ROUGE(args, os.path.join(path_dir, 'targets.txt'), os.path.join(path_dir, 'preds.txt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    gaze_mode = 'org'
    parser.add_argument("--mode", default='train', type=str, choices=['train', 'test', 'oracle'])
    parser.add_argument("--train_data", default='large_data/LCSTS_train.pt')
    parser.add_argument("--test_data", default='large_data/LCSTS_test.pt')
    # parser.add_argument("--model_path", default='models/')
    parser.add_argument("--model_path", default='1204_models/'+gaze_mode+'/')
    parser.add_argument("--result_path", default='results/'+gaze_mode+'/')

    parser.add_argument("--bert_model", default='/data10T/wangbingbing/bert-large', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_config", type=str,default='/data10T/wangbingbing/bert-chinese-uncased/',
                        help="Bert config path.")

    parser.add_argument(
        "--syntactic_layers",
        # type=str, default='0',
        # type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
        type=str, default='0,1,2,3,4,5,6,7,8,9,10,11',
        # type=str, default='0,1,2,3,4,5,6,7',
        # type=str, default='4,5,6,7,8,9,10,11',
        help="comma separated layer indices for syntax fusion",
    )
    parser.add_argument(
        "--max_syntactic_distance",
        default=1, type=int,
        help="Max distance to consider during graph attention",
    )
    parser.add_argument(
        "--eye_max_syntactic_distance",
        default=0.5, type=float,
        help="Max distance to consider during graph attention",
    )
    parser.add_argument(
        "--use_structural_loss",
        type=bool,
        default=False,
        help="Whether to use structural loss",
    )
    parser.add_argument(
        "--struct_loss_coeff",
        default=1.0, type=float,
        help="Multiplying factor for the structural loss",
    )
    parser.add_argument(
        "--num_syntactic_heads",
        default=12, type=int,
        help="Number of syntactic heads",
    )
    parser.add_argument("--revise_gat", default=gaze_mode, type=str, required=False,
                        help="eye feature of syntax")
    parser.add_argument("--device", type=str, default='cuda:1',
                        help="wait N times of decreasing dev score before early stop during training")

    parser.add_argument('--log_file', default='')
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=200.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=3047,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--param_init", default=0, type=float)
    parser.add_argument("--param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    args = parser.parse_args()
    init_logger(args.log_file)

    # initial remote debugger
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    # create result folder
    if os.path.exists(args.result_path) and os.listdir(args.result_path) and args.mode != 'train':
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.result_path))
    if not os.path.exists(args.result_path) and args.mode != 'train':
        os.makedirs(args.result_path)
    # create model folder
    if os.path.exists(args.model_path) and os.listdir(args.model_path) and args.mode == 'train':
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.model_path))
    if not os.path.exists(args.model_path) and args.mode == 'train':
        os.makedirs(args.model_path)

    # begin mode work
    if args.mode == 'oracle':
        baseline(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)