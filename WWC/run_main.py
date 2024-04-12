import argparse
import glob
import logging
import os
import time
import random
import shutil
from typing import List, Optional

import numpy as np
import torch

# from pytorch_transformers import (BertConfig, BertTokenizer, RobertaConfig, \
#         RobertaTokenizer)
from transformers import BertConfig,BertTokenizer, RobertaConfig,RobertaTokenizer,\
    AdamW
from pytorch_transformers import WarmupLinearSchedule
from multitask_modeling import BertForSequenceClassificationMultiTask,\
        RobertaForSequenceClassificationMultiTask, BertForSpanClassification
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import data_utils
import trainer

logger = logging.getLogger(__name__)

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ())
ALL_TASKS = [ 'RTE', 'CB', 'COPA','WiC']

# `bert-span` is not to be confused with spanBERT.
# `bert-span` is a modified BERT classifier where the classification layer is
# added on top of one or more specified span(s), instead of the CLS token
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassificationMultiTask, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassificationMultiTask, RobertaTokenizer),
    'bert-span': (BertConfig, BertForSpanClassification, BertTokenizer),
}


def set_seed(args, n_gpu):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _get_validated_args(input_args: Optional[List[str]] = None) -> argparse.Namespace:
    """validate arguments"""
    parser = argparse.ArgumentParser()
    # ALL_TASKS = ['BoolQ', 'BoolQNLI', 'RTE', 'CB', 'COPA', 'WiC']
    # These parameters are always required
    # WSC/WiC: bert-span
    parser.add_argument("--tasks", default="WSC", type=str, required=False,
                       help="Choose one or multiple tasks from: " + ', '.join(ALL_TASKS))
    parser.add_argument("--model_type", default="bert-span", type=str, required=False,
                        choices=['bert', 'roberta', 'bert-span'],
                        help="Choose one from `bert`, `roberta`, or `bert-span`.")
    parser.add_argument("--model_name", default="/data10T/wangbingbing/bert-large", type=str,
                        required=False,#, choices=", ".join(ALL_MODELS),help="pretrained model from: " + ", ".join(ALL_MODELS)
                        )
    parser.add_argument("--output_dir", type=str, required=False,default='output/',
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--test_set", default="test", type=str, choices=["test", "val", "train"],
                        help="Dataset to conduct evaluation for.")

    parser.add_argument("--do_train", default=True, action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_predict", default=True, action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--fine_tune", action='store_true',
                        help="Whether run in fine-tuning mode on a specific model")

    ## Other optional parameters
    parser.add_argument("--transfer_from", default=None, type=str,
                        choices=[None, "squad1-bert-large", "squad2-bert-large", "squad2-bert-large-wwm"],
                        help="On which task to do supervised transfer learning from")
    parser.add_argument("--patience_for_task_dropping", default=10, type=int,
                        help="wait this many epochs to drop the task if no improvement is made on validation accuracy")
    parser.add_argument("--demo", action='store_true',
                        help="Whether to use toy examples. If set to True, all tasks will use 20 samples from training "
                             "set for both training and validation.")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size", default=6, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--load_n_epoch", default=None, type=int,
                        help="Load checkpoint for evaluation.")
    parser.add_argument("--train_from", default=None, type=int,
                        help="Load checkpoint and keep training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=200.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps. (suggestion: this should be larger when training batch "
                             "size is large.)")
    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--logging_freq", type=int, default=5,
                        help="Log information to tensorboard this many times per epoch. If set as a list then apply"
                             "individual frequencies for different tasks")
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
    parser.add_argument("--revise_gat", default='org', type=str, required=False,
                        help="eye feature of syntax")
    parser.add_argument("--device", type=str, default='cuda:1',
                        help="wait N times of decreasing dev score before early stop during training")

    args = parser.parse_args(input_args)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    assert args.do_train or args.do_predict, \
        "At least one of do_train and do_predict needs to be True."

    args.tasks = [k.strip() for k in args.tasks.split(",")]

    if args.fine_tune:
        assert len(args.tasks) == 1, "In fine-tuning mode only one task is allowed."
        assert isinstance(args.logging_freq, int), "In fine-tuning mode logging frequency has to be an integer."
        assert args.mtl_path is not None, "MTL model path need to be specified!"

        args.mtl_path = data_utils.EXP_PATH + args.mtl_path
        args.finetuned_task_id = 1

    if args.do_train:
        args.test_set = "val"

    args.output_dir = data_utils.EXP_PATH# + args.output_dir

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.demo:
            # in demo mode, delete entire folder if exists
            shutil.rmtree(args.output_dir)

        # elif args.do_train:
        #     raise ValueError("Output directory ({}) already exists and is not empty.".
        #             format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args

def _build_pretrained_model(args, config_class, model_class, label_count_info):
    config = config_class.from_pretrained(args.model_name)
    print(type(config))
    config.attention_probs_dropout_prob = 0.05
    config.hidden_dropout_prob = 0.05
    config.use_structural_loss = args.use_structural_loss
    config.struct_loss_coeff = args.struct_loss_coeff
    config.num_syntactic_heads = args.num_syntactic_heads
    config.syntactic_layers = args.syntactic_layers
    config.max_syntactic_distance = args.max_syntactic_distance
    config.eye_max_syntactic_distance = args.eye_max_syntactic_distance
    config.eye_length = 20000
    config.device = args.device
    config.revise_gat = args.revise_gat
    args_list =[args.use_structural_loss,args.struct_loss_coeff,args.num_syntactic_heads,
                args.syntactic_layers,args.max_syntactic_distance,args.eye_max_syntactic_distance]
    print(type(config))
    model = model_class.from_pretrained(args.model_name,
                                        config=config,
                                        # batch_size = args.batch_size,
                                        # args_list,
                                        tasks=args.tasks,
                                        task_num_labels={task: label_count_info[task]
                                                         for task in args.tasks}
                                        ,ignore_mismatched_sizes=True)
    return model

def load_trained_model(args, config_class, model_class, label_count_info, n_gpu, device):
    """Load trained model for evaluation"""
    model = _build_pretrained_model(args, config_class, model_class, label_count_info)
    model.to(device)

    model_path = args.output_dir + "/epoch_%d" % args.load_n_epoch
    ckpt_state = torch.load(model_path)['model_state']
    model.load_state_dict(ckpt_state)

    return model



def load_or_create_model(args, config_class, model_class, label_count_info,
                         n_gpu, device, num_train_optimization_steps):
    """
    Load model or create model from scratch.

     Params:
        `args`: arguments for model and task information
        `label_count_info`: a dictionary mapping task name to its label count.
        `n_gpu`: an interger denoting the number of gpus available
        `device`: either cuda:n or cpu
        `num_train_optimization_steps`: an integer denoting the total number of
            iterations (batches)
    """
    model = _build_pretrained_model(args, config_class, model_class, label_count_info)
   # for transfer learning setup, load pretrained weights in specified path
    if args.transfer_from is not None:
        transfer_path = data_utils.TRANSFER_PATH[args.transfer_from]
        logger.info("Transfer from %s" % args.transfer_from)

        transfer_model = torch.load(transfer_path)
        if "model_state" in transfer_model:
            transfer_model = transfer_model["model_state"]

        # TODO: needs work to be able to determine whether only loading the
		# BertModel part (as opposed to load everything including the
		# classififer
        model.load_state_dict(transfer_model)

    model.to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print('Total - ', pytorch_total_params)
    # print('Trainable - ', trainable_pytorch_total_params)
    # exit()

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer, scheduler



def build_data_loaders(tasks: List[str]=[]) -> dict:
    """according to the specified tasks, create data loaders"""
    data_loaders = dict()
    for task in tasks:
        data_loaders[task] = data_utils.TASK_TO_LOADER[task]()
    return data_loaders


def main(input_args: Optional[List[str]] = None):
    args = _get_validated_args(input_args)

    logger.info("Loading tasks...")
    start_time = time.time()


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = 1#torch.cuda.device_count()

    set_seed(args, n_gpu)
    logging.info('Using output directory: %s' % args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained("/data10T/wangbingbing/bert-large")
    # text = 'Snowflakes began to fall from the sky.'
    # print(tokenizer.tokenize(text))
    # exit()
    data_loaders = build_data_loaders(args.tasks)
    label_count_info = {task: len(task_loader.get_labels()) \
                        for task, task_loader in data_loaders.items()}

    if args.do_train:
        # need to be consistent given the same task sequence, so that continuing
        # training from checkpoint is easy.
        task_train_samples = dict()
        task_val_samples = dict()
        task_test_samples = dict()
        total_sample_count = 0

        for task in args.tasks:
            # For each task, load training and validation data
            cur_task_raw_train_samples = data_loaders[task].get_train_examples(args.demo)
            task_train_samples[task] = cur_task_raw_train_samples

            # If demo mode is on, load 100 samples from training set for both train and val
            val_set = "train" if args.demo else "val"
            task_val_samples[task] = data_loaders[task].get_test_examples(
                args.demo, test_set=val_set)

            total_sample_count += len(cur_task_raw_train_samples)


        num_train_optimization_steps = (total_sample_count // args.batch_size)  * args.num_train_epochs

        processed_train_samples = {task: data_utils.SuperGlueDataset(data_loaders[task],
                                                                    task_samples,
                                                                    tokenizer,
                                                                    args.max_seq_length,
                                                                    args.model_type)
                                   for task, task_samples in task_train_samples.items()}

        processed_val_samples = {task: data_utils.SuperGlueDataset(data_loaders[task],
                                                                  task_samples,
                                                                  tokenizer,
                                                                  args.max_seq_length,
                                                                  args.model_type)
                                 for task, task_samples in task_val_samples.items()}

        for task in args.tasks:
            cur_task_raw_test_samples = data_loaders[task].get_test_examples(args.demo, test_set="val")
            task_test_samples[task] = cur_task_raw_test_samples

        processed_test_samples = {task: data_utils.SuperGlueDataset(data_loaders[task],
                                                                    task_samples,
                                                                    tokenizer,
                                                                    args.max_seq_length,
                                                                    args.model_type)
                                  for task, task_samples in task_test_samples.items()}

        for task in args.tasks:
            logging.info("Task: %s\tTraining: %d\tValidation: %d" % (task, len(processed_train_samples[task]),
                                                                     len(processed_val_samples[task])))

        model, optimizer, scheduler = load_or_create_model(args, config_class, model_class, label_count_info,
                                                           n_gpu, args.device, num_train_optimization_steps)

        trainer.run_training(processed_train_samples=processed_train_samples,
                             processed_val_samples=processed_val_samples,
                             processed_test_samples=processed_test_samples,
                             model=model, optimizer=optimizer,
                             scheduler=scheduler, tokenizer=tokenizer, args=args, device=args.device,
                             n_gpu=n_gpu,task_name=args.tasks[0])

    # TODO: predict
    elif args.do_predict:
        task_test_samples = dict()
        for task in args.tasks:
            cur_task_raw_test_samples = data_loaders[task].get_test_examples(args.demo, test_set="val")
            task_test_samples[task] = cur_task_raw_test_samples

        processed_test_samples = {task: data_utils.SuperGlueDataset(data_loaders[task],
                                                                    task_samples,
                                                                    tokenizer,
                                                                    args.max_seq_length,
                                                                    args.model_type)
                                  for task, task_samples in task_test_samples.items()}
        model = load_trained_model(args, config_class, model_class, label_count_info, n_gpu, device)
        trainer.run_evaluation(processed_test_samples=processed_test_samples, model=model, tokenizer=tokenizer,
                               args=args, device=args.device, n_gpu=n_gpu)


if __name__ == '__main__':
    main()
