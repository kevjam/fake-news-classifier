# -------------- Preprocessing Packages --------------
# For tokenizing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# For Chinese word segmentation
import jieba

# -------------- General Packages --------------
# General Use
import pandas as pd
import numpy as np

# For Saving/Loading Files
from .storage import save_pickle
import os

# Convert labels to integers for predictions
def encode_labels(df):
    # encoding the labels
    labels = {'unrelated':0,'agreed':1,'disagreed':2}
    df['label'].replace(labels,inplace=True)
    df = df.reset_index()
    return df

# Splits Chinese characters into segments using jieba
def segment_zh(text):
    return ' '.join(jieba.cut(text, cut_all=False))

# Segment Chinese text in a dataframe
def segment_zh_data(df):
    df['title1_zh'] = df['title1_zh'].apply(segment_zh)
    df['title2_zh'] = df['title2_zh'].apply(segment_zh)
    return df

# Create a English word tokenizer given dataframe(s)
def create_en_tokenizer(df, num_words=None, lower=True, split=' ', oov_token=None, directory = './tokenizers/', filename='tokenizer'):
    # create the tokenizer
    t = Tokenizer(num_words=num_words, lower=lower, split=split, oov_token=oov_token)
    
    # fit tokenizer
    t.fit_on_texts(df['title1_en'].append(df['title2_en']))
    
    # save for future use
    os.makedirs(directory, exist_ok=True)
    save_pickle(t,directory+'en_'+filename)
    return t
    
# Create a Chinese word tokenizer given a dataframe
def create_zh_tokenizer(df, num_words=None, lower=True, split=' ', oov_token=None, directory = './tokenizers/', filename='tokenizer'):
    # create the tokenizer
    t = Tokenizer(num_words=num_words, lower=lower, split=split, oov_token=oov_token)
    
    # fit tokenizer
    t.fit_on_texts(df['title1_zh'].append(df['title2_zh']))
    
    # save for future use
    os.makedirs(directory, exist_ok=True)
    save_pickle(t,directory+'zh_'+filename)
    return t

# Tokenizes titles using the EN and ZH tokenizers
def tokenize(t_en, t_zh, df, en_maxlen=22, zh_maxlen=16, label_exists=True):
    # tokenize the documents
    data1 = pad_sequences(sequences=t_en.texts_to_sequences(df['title1_en']), maxlen=en_maxlen)
    data2 = pad_sequences(sequences=t_en.texts_to_sequences(df['title2_en']), maxlen=en_maxlen)
    data3 = pad_sequences(sequences=t_zh.texts_to_sequences(df['title1_zh']), maxlen=zh_maxlen)
    data4 = pad_sequences(sequences=t_zh.texts_to_sequences(df['title2_zh']), maxlen=zh_maxlen)
    
    # recombine
    if label_exists: df = pd.DataFrame(np.concatenate((data1,data2,data3,data4),axis=1)).join(df['label'])
    else: df = pd.DataFrame(np.concatenate((data1,data2,data3,data4),axis=1))
    return df