class Config(object):	
	apr_dir = '../model/'
	data_dir = '../corpus/'
	model_name = 'model_4.pt'
	epoch = 1
	bert_model = '/data10T/wangbb/bert-large-uncased'
	type = 'bert'
	num_layers = 24
	lr = 5e-5
	eps = 1e-8
	batch_size = 32
	mode = 'prediction' # for prediction mode = "prediction"
	training_data = 'ner2_revise_GECO_English_1.txt'
	log_path = 'GECO_English_1_attn_bert_large.csv'
	val_data = '2token-en-ZuCo1-utf.txt'
	test_data = '2token-en-ZuCo1-utf.txt'
	test_out = 'test_prediction.csv'
	raw_prediction_output = 'raw_prediction_GECO1.csv'
	k_fold = 10
	random_seed = 42
	multi_head = 'avg'
