import random
import numpy as np
import torch
import transformers
from transformers import BertTokenizer,RobertaTokenizer
import src.dataloader
import sys
sys.path.append('src')
import eval_metric
device = torch.device('cuda:0')
# device = 'cpu'
class RobertaRegressionModel(torch.nn.Module):
  def __init__(self, model_name='roberta-base'):
    super(RobertaRegressionModel, self).__init__()

    if 'roberta' in model_name:
      self.roberta = transformers.RobertaModel.from_pretrained(model_name)
      # self.roberta = RobertaTokenizer.from_pretrained(model_name)
    elif 'bert' in model_name:
      self.roberta = transformers.BertModel.from_pretrained(model_name)

    EMBED_SIZE = 1024 if 'large' in model_name else 768
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear(EMBED_SIZE, 5)
    )


  def forward(self, X_ids, X_attns, predict_mask):
    """
    X_ids: (B, seqlen) tensor of token ids
    X_attns: (B, seqlen) tensor of attention masks, 0 for [PAD] tokens and 1 otherwise
    predict_mask: (B, seqlen) tensor, 1 for tokens that we need to predict

    Output: (B, seqlen, 5) tensor of predictions, only predict when predict_mask == 1
    """

    temp = self.roberta(X_ids, attention_mask=X_attns).last_hidden_state

    # (B, seqlen, 5)
    Y_pred = self.decoder(temp)

    # Where predict_mask == 0, set Y_pred to -1
    Y_pred[predict_mask == 0] = -1

    return Y_pred


class ModelTrainer():
  """Handles training and prediction given CSV"""

  def __init__(self, model_name='roberta-base'):
    self.model_name = model_name
    self.model = RobertaRegressionModel(model_name).to(device)


  def train(self, train_df, valid_df=None, num_epochs=5, lr=5e-5, batch_size=64, feature_ids=[0,1,2,3,4]):
    train_data = src.dataloader.EyeTrackingCSV(train_df, model_name=self.model_name)

    random.seed(12345)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    # self.model.load_state_dict(torch.load('model_weights.pth'))
    self.model.train()
    best_result = 99999
    for epoch in range(num_epochs):
      for X_tokens, X_ids, X_attns, Y_true in train_loader:
        opt.zero_grad()
        X_ids = X_ids.to(device)
        X_attns = X_attns.to(device)
        Y_true = Y_true.to(device)
        predict_mask = torch.sum(Y_true, axis=2) >= 0
        Y_pred = self.model(X_ids, X_attns, predict_mask)
        loss = mse(Y_true[:,:,feature_ids], Y_pred[:,:,feature_ids])
        loss.backward()
        opt.step()

      print('Epoch:', epoch+1)

      if valid_df is not None:
        predict_df = self.predict(valid_df)
        # src.eval_metric.evaluate(predict_df, valid_df)
        result = eval_metric.evaluate(predict_df, valid_df)
        if result<best_result:
          best_result=result
          print('save')
          torch.save(self.model.state_dict(), str(best_result)+'model_weights.pth')

  def predict(self, valid_df):
    valid_data = src.dataloader.EyeTrackingCSV(valid_df, model_name=self.model_name)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=128
                                               )

    predict_df = valid_df.copy()
    predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']] = 9999

    # Assume one-to-one matching between nonzero predictions and tokens
    predictions = []
    self.model.load_state_dict(torch.load('33.80439399848807model_weights.pth'))
    self.model.eval()
    num = 0
    for X_tokens, X_ids, X_attns, Y_true in valid_loader:
      print(num,len(valid_loader))
      num+=1
      X_ids = X_ids.to(device)
      # print(X_ids)
      # exit()
      X_attns = X_attns.to(device)
      predict_mask = torch.sum(Y_true, axis=2) >= 0
      with torch.no_grad():
        # print(X_ids.shape,X_attns.shape,predict_mask.shape)
        Y_pred = self.model(X_ids, X_attns, predict_mask).cpu()
        # print(Y_pred)
        # exit()
      for batch_ix in range(X_ids.shape[0]):
        # print('1')
        for row_ix in range(X_ids.shape[1]):
          token_prediction = Y_pred[batch_ix, row_ix]
          # print('toke:',token_prediction)
          # exit()
          if token_prediction.sum() != -5.0:
            token_prediction[token_prediction < 0] = 0
            predictions.append(token_prediction)
    # print(predictions)
    predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']] = np.vstack(predictions)
    return predict_df