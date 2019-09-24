#importing required libraries
import random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import tensorflow as tf
from keras.models import load_model
from keras.utils import np_utils
from keras.preprocessing.text import text_to_word_sequence
import language_check
import string
import tensorflow as tf
import re
import io
import os
from flask import Flask,request
import json
import csv
import requests

app = Flask(__name__)
port = int(os.getenv('PORT', '3000'))

@app.route('/getsummary',methods=['GET','POST'])
def getsummary():
    try:

        df = pd.read_csv(request.files.get('data_file'))
        print("Dataframe:",df)
        text=(open("summary_types1.txt").read())
        text=text.lower()
        text = text.replace(".", " .")
        uniquewords = set(text_to_word_sequence(text,filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'))
        print(uniquewords)
        complete_seq=text_to_word_sequence(text,filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
        #Sorting set of words in text and mapping them
        words = sorted(list(uniquewords))
        n_to_word = {n:word for n, word in enumerate(words)}
        word_to_n = {word:n for n, word in enumerate(words)}
        #creating word sequences and mapping them to their's respective next word
        X = []
        Y = []
        length = len(text_to_word_sequence(text,filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'))
        seq_length = 120
        print("seq length:",seq_length)
        for i in range(0, length-seq_length, 1):
            sequence = complete_seq[i:i + seq_length]
            label =complete_seq[i + seq_length]
            X.append([word_to_n[word] for word in sequence])
            Y.append(word_to_n[label])

        graph = tf.Graph()
        with graph.as_default():
          session = tf.Session()
          with session.as_default():
              model = load_model("word_model.h5")
        model_graph = graph
        model_session = session
        #Predict next sequence of words that will occur after a given random sequence of words
        rno=random.randint(1,len(X))
        print("Random Number:",rno)
        string_mapped = X[rno]
        full_string = [n_to_word[value] for value in string_mapped]
        # generating words
        for i in range(160):
            x = np.reshape(string_mapped,(1,len(string_mapped), 1))
            x = x / float(len(words))
            with model_graph.as_default():
                with model_session.as_default():
                    pred_index = np.argmax(model.predict(x, verbose=0))
            seq = [n_to_word[value] for value in string_mapped]
            full_string.append(n_to_word[pred_index])
            string_mapped.append(pred_index)
            string_mapped = string_mapped[1:len(string_mapped)]

        #combining words to generate summary
        txt=""
        for word in full_string:
            txt = txt+" "+word
        print("Raw summary:",txt)
        #array of features
        abs_val_features=['PROD ID','LOC ID','SALES HISTORY SDATE','SALES HISTORY EDATE','PEAK VAL','PEAK DATE','MINVAL','MINDATE','AVG SALES','HIST START YEAR','HIST END YEAR','YEALRY AVG','PFS YEAR','PFE YEAR','PMAX SALES','PMAX DATE','PMIN SALES','PMIN DATE','FP AVG','FPF YEARLY AVG','HIST TREND','FP START YEAR','FP END YEAR','FPS TREND','CFS MAX DATE','CFS MIN DATE','CFS MAX','CFS MIN','CF AVG','CF START YEAR','CF END YEAR','CF YEARLY AVG','CF TREND','OFS MAX DATE','OFS MIN DATE','OFS MAX','OFS MIN','OF AVG','OF START YEAR','OP END YEAR','OF YEARLY AVG','OF TREND']
        excel_features=['PROD_ID','LOC_ID','SALES_HISTORY_SDATE','SALES_HISTORY_EDATE','PEAK_VAL','PEAK_DATE','MINVAL','MINDATE','AVG_SALES','HIST_START_YEAR','HIST_END_YEAR','YEALRY_AVG','PFS_YEAR','PFE_YEAR','PMAX_SALES','PMAX_DATE','PMIN_SALES','PMIN_DATE','FP_AVG','FPF_YEARLY_AVG','HIST_TREND','FP_START_YEAR','FP_END_YEAR','FPS_TREND','CFS_MAX_DATE','CFS_MIN_DATE','CFS_MAX','CFS_MIN','CF_AVG','CF_START_YEAR','CF_END_YEAR','CF_YEARLY_AVG','CF_TREND','OFS_MAX_DATE','OFS_MIN_DATE','OFS_MAX','OFS_MIN','OF_AVG','OF_START_YEAR','OP_END_YEAR','OF_YEARLY_AVG','OF_TREND']
        #replace features with their absolute values in the generated summary
        for index, row in df.iterrows():
            for i in range(0,len(abs_val_features)):
                try:
                    #txt=txt.replace(abs_val_features[i].lower(), str(row[abs_val_features[i]]))
                    txt=re.sub(r""+abs_val_features[i].lower(),str(row[excel_features[i]]),txt)
                except:
                    print("An exception occurred for:",abs_val_features[i].lower())

        txt=txt.replace('\n', '')
        first_halfsent=txt.find('.')
        txt=txt[first_halfsent+1:]
        last_halfsent=txt.rfind('.')
        txt=txt[:last_halfsent+1]
        spos=txt.find('start of summary ')
        epos=txt.find('end of summary ')
        if(spos>epos):
            sum=txt[spos+16:]+txt[:epos]+"."
        else:
            sum=txt[spos+16:epos]+"."

        #grammer check
        #print("Generated summary:",sum)
        tool = language_check.LanguageTool('en-US')
        matches = tool.check(sum)
        final_summary=language_check.correct(sum, matches)
        print("Final Summary:",final_summary)
        return final_summary

    except Exception as e:
        return str(e)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)