import pandas as pd
import csv

def file(input_filelist,output_file):
    csv_file = open(output_file, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['sentence_id', 'word_id', 'word', 'nFix', 'FFD', 'GPT', 'TRT', 'GD'])

    before_sentence_id = -1
    sum = 0
    for input_file in input_filelist:
        sum+=1
        data = pd.read_csv(input_file, encoding='utf-8')
        sentence_id = data['sentence_id']
        print(input_file,int(sentence_id[len(sentence_id)-1]))
        # continue
        word_id = data['word_id']
        word = data['word']
        nFix = data['nFix']
        GD = data['GD']
        FFD = data['FFD']
        GPT = data['GPT']
        TRT = data['TRT']


        now_sentence_id = int(sentence_id[0])
        print(now_sentence_id,before_sentence_id+1)
        assert before_sentence_id+1 == now_sentence_id
        for i in range(len(sentence_id)):
            print(sum,i/len(sentence_id))
            csv_writer.writerow([sentence_id[i], word_id[i], word[i], nFix[i], FFD[i], GPT[i], TRT[i], GD[i]])
        before_sentence_id = int(sentence_id[len(sentence_id)-1])
    print(before_sentence_id)


input_list = ['predict-yelp2013-train0-0.csv','predict-yelp2013-train1-0.csv','predict-yelp2013-train22-0.csv','predict-yelp2013-train23-0.csv'
              ,'predict-yelp2013-train24-0.csv','predict-yelp2013-train25-0.csv','predict-yelp2013-train26-0.csv'
              ,'predict-yelp2013-train27-0.csv','predict-yelp2013-train3-0.csv','predict-yelp2013-train4-0.csv',
              'predict-imdb-train50000-0.csv'
              ,'predict-yelp2013-train5-0.csv']
output_file = 'predict-yelp2013-train-0.csv'
file(input_list,output_file)