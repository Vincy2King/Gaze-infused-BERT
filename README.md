# Gaze-infused BERT

This research delves into the intricate connection between self-attention mechanisms
in large-scale pre-trained language models, like BERT, and human gaze patterns, with
the aim of harnessing gaze information to enhance the performance of natural
language processing (NLP) models.

## Analysis of Self-Attention and Gaze
1. Download [dataset](https://pan.baidu.com/s/1lArAz2-wo7GKIYsRlxZyOg?pwd=4x0l) pwd:4x0l

2. `cd Analysis` and run `python bert_crf.py` to obtain self-attention of BERT saved to `corpus/`

3. Calculate the spearmanr between self-attention and gaze by running `plot.py`.

## Gaze-infused BERT Method

1. For those dataset without gaze signals, first `cd Gaze_prediction_model_V1/scripts/` and run `run_roberta.py`

2. For GLUE and SNLI datasets, `cd source/`, and run the corresponding Python file, such as 
    * CoLA2.py, [download](https://pan.baidu.com/s/1Mp7SOwzgvYO73_xoYrBceQ?pwd=q36f) q36f
    * MNLI.py, [download](https://pan.baidu.com/s/1WXqHfPgVuqAUiam4FkR4NA?pwd=ykp7) ykp7 
    * MRPC.py, [download](https://pan.baidu.com/s/1fIjXowhpfHQ593D8UJW9zg?pwd=48sb) 48sb 
    * QNLI.py, [download](https://pan.baidu.com/s/15xoUSB_4b_VC_jhzHze_kA?pwd=ftwa) ftwa
    * QQP.py, [download](https://pan.baidu.com/s/10EOh-4SQjUFRcCo0kMVnOw?pwd=vld2) vld2
    * RTE.py, [download](https://pan.baidu.com/s/19yDBxX75NUBLvkQZCawRIQ?pwd=ypnm) ypnm
    * SNLI.py, [download](https://pan.baidu.com/s/1jKRUY-miKj3F2ZV6ANTL1A?pwd=wlkw) wlkw
    * SST-2.py, [download](https://pan.baidu.com/s/1HnLJcntmVYduQydv1OVvNg?pwd=dsc4) dsc4
    * STS-B.py, [download](https://pan.baidu.com/s/1TnWFFt8qZW9MfF3LyPp8YA?pwd=aiuw) aiuw

3. For WSC, WiC, and COPA datasets, `cd WWC/`, and revised the corresponding dataset to run `python run_main.py`

4. For LCSTS dataset, `cd LCSTS/`, and `python train.py`
