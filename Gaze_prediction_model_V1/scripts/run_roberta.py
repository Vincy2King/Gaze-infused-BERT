import sys
sys.path.append('../')
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['CUDA_VISIBLE_DEVICES']='1'

import pandas as pd
import argparse

import src.model

import torch
import torchvision.models as models
parser = argparse.ArgumentParser()

parser.add_argument('--num-ensembles', type=int,default=1)
parser.add_argument('--use-provo', type=bool,default=True)

parser.add_argument('--mode', type=str,default='submission')

args = parser.parse_args()

val_list = [#'test.title-bert','test.article-bert',
            # 'valid.title-bert','valid.title-bert',]
            'valid.article-bert']
  #,'train.title-bert',]
for val in val_list:
  valid_file = "data/null_dev"
  if args.mode == 'dev':
    train_df = pd.read_csv("gaze_train.csv")
    valid_df = pd.read_csv("gaze_valid.csv")
  else:
    train_df = pd.read_csv("data/training_data/train.csv")
    valid_df = pd.read_csv(valid_file+'.csv')

  model_trainer = src.model.ModelTrainer(model_name='roberta-base')

  predict_df = model_trainer.predict(valid_df)
  predict_df.to_csv(valid_file+'_new.csv', index=False)