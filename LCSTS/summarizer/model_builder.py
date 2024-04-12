import os

import torch
import torch.nn as nn
# from transformers import BertModel,BertPreTrainedModel
from modeling_bert1 import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch.nn.init import xavier_uniform_

from summarizer.encoder import Classifier


class Bert(nn.Module):
    def __init__(self, bert_model, cache_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()

        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained(bert_model, cache_dir=cache_dir,
                                                   config=bert_config,ignore_mismatched_sizes=True)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask,FFDs, GDs, GPTs, TRTs, nFixs):
        encoded_layers, _ = self.model(x, attention_mask =mask, token_type_ids=segs,FFDs=FFDs, GDs=GDs,GPTs=GPTs,TRTs=TRTs,nFixs=nFixs)
        # print('encoded:',encoded_layers.shape)
        top_vec = encoded_layers#[-1]
        # print(top_vec.size())
        return top_vec



class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, bert_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device

        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
        self.bert = Bert(args.bert_model, cache_dir, load_pretrained_bert, bert_config=bert_config)
        self.encoder = Classifier(self.bert.model.config.hidden_size)
        self.config = bert_config#self.bert.model.config

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

    def load_cp(self, pt):
        self.load_state_dict(pt, strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, nfixs, ffds, trts, gpts, gds, sentence_range=None):

        top_vec = self.bert(x, segs, mask,FFDs=ffds,
                            GDs=gds,
                            GPTs=gpts,
                            TRTs=trts,
                            nFixs=nfixs) # [batch size, sequence length, hidden size]

        # print('top_vec:',len(top_vec))
        # top_vec = top_vec[0]    #32,142,768
        # print(torch.arange(top_vec.size(0)).shape,torch.arange(top_vec.size(0)).unsqueeze(1).size(),clss.size())
        # sents_vec = torch.gather(top_vec, 1, clss.unsqueeze(-1).expand(-1, -1, top_vec.size(-1)))
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss] # [batch size, clss length, hidden size]
        sents_vec = sents_vec * mask_cls[:, :, None].float() # [batch size, clss length, hidden size]
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1) # [batch size, clss length, 1]
        return sent_scores, mask_cls
