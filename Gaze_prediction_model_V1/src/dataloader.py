import torch
import transformers
import pandas as pd
FEATURES_NAMES = ['nFix', 'FFD', 'GPT', 'TRT', 'GD']

class EyeTrackingCSV(torch.utils.data.Dataset):
  """Tokenize sentences and load them into tensors. Assume dataframe has sentence_id."""

  def __init__(self, df, model_name='roberta-base'):
    self.model_name = model_name
    self.df = df.copy()

    # Re-number the sentence ids, assuming they are [N, N+1, ...] for some N
    self.df.sentence_id = self.df.sentence_id - self.df.sentence_id.min()
    self.num_sentences = self.df.sentence_id.max() + 1
    print('self.num_sentences:',self.num_sentences,self.df.sentence_id.nunique())
    assert self.num_sentences == self.df.sentence_id.nunique()

    self.texts = []
    import chardet
    for i in range(self.num_sentences):
      rows = self.df[self.df.sentence_id == i]
      text = rows.word.tolist()
      j = 0
      for i in range(len(text)):
        if type(text[i]).__name__ != 'str':
          text[i]='null'
        j += 1
      text[-1] = text[-1].replace('<EOS>', '')
      # print(text)
      self.texts.append(text)
      # print(len(text),text)
    # Tokenize all sentences
    if 'roberta' in model_name:
      self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True,max_length=512)#,max_length=512)
    elif 'bert' in model_name:
      self.tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    self.ids = self.tokenizer(self.texts, padding=True, is_split_into_words=True, return_offsets_mapping=True,max_length =512,pad_to_max_length=True)


  def __len__(self):
    return self.num_sentences
  

  def __getitem__(self, ix):
    # print('ix:',ix)
    input_ids = self.ids['input_ids'][ix]
    offset_mapping = self.ids['offset_mapping'][ix]
    attention_mask = self.ids['attention_mask'][ix]

    if len(input_ids)>514:
      if input_ids[514] == 2 or input_ids[514] == 1:
        input_ids = input_ids[:514]
        offset_mapping = offset_mapping[:514]
        attention_mask = attention_mask[:514]
      else:
        input_ids = input_ids[:513] + [2]
        offset_mapping = offset_mapping[:513] + [offset_mapping[0]]
        attention_mask = attention_mask[:513] + [attention_mask[0]]

    input_tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids]

    if 'roberta' in self.model_name:
      is_first_subword = [t[0] == 'Ġ' for t in input_tokens]
    elif 'bert' in self.model_name:
      is_first_subword = [t0 == 0 and t1 > 0 for t0, t1 in offset_mapping]

    features = -torch.ones((len(input_ids), 5))

    features[is_first_subword] = torch.Tensor(
      self.df[self.df.sentence_id == ix][FEATURES_NAMES].to_numpy()
    )

    return (
      input_tokens,
      torch.LongTensor(input_ids),
      torch.LongTensor(attention_mask),
      features,
    )
