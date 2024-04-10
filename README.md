# Gaze-infused BERT

This research delves into the intricate connection between self-attention mechanisms
in large-scale pre-trained language models, like BERT, and human gaze patterns, with
the aim of harnessing gaze information to enhance the performance of natural
language processing (NLP) models.

## Analysis of Self-Attention and Gaze

1. `cd Analysis` and run `python bert_crf.py` to obtain self-attention of BERT saved to `corpus/`

2. Calculate the spearmanr between self-attention and gaze by running `plot.py`.

## Gaze-infused BERT Method

1. For those dataset without gaze signals, first `cd Gaze_prediction_model_V1/scripts/` and run `run_roberta.py`

2. For GLUE and SNLI dataset, `cd source/`, and run corresponding python file, such as 
    * CoLA2.py
    * MNLI.py
    * MRPC.py
    * QNLI.py
    * QQP.py
    * RTE.py
    * SNLI.py
    * SST-2.py
    * STS-B.py

3. For WSC WiC COPA dataset, `cd WWC/`, and revised the correpsonding dataset to run `python run_main.py`

4. For LCSTS dataset, `cd LCSTS/`, and `python train.py`
