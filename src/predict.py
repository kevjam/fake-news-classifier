from preprocess import filter_dataset
from utils.tokenizing import segment_zh_data, tokenize
from utils.storage import load_tokenizers

# -------------- General Packages --------------
# Data Manipulation
import pandas as pd
import numpy as np

# For Saving/Loading Files
import os
from keras.models import load_model

# For parsing command-line arguments
import argparse

# Prepare a dataset for preprocessing, returns ids and a preprocessed dataframe
def prepare(df, t_EN, t_ZH, EN_SENTENCE_SIZE=22, ZH_SENTENCE_SIZE=16):
    df = filter_dataset(df, drop_columns=['tid1','tid2'], label_exists=False)
    ids = df['id']
    df = segment_zh_data(df.drop(columns=['id']))
    df = tokenize(t_EN,t_ZH,df,EN_SENTENCE_SIZE,ZH_SENTENCE_SIZE,label_exists=False)
    return ids,df

# Get predictions on a model
def get_predictions(df, model, EN_SENTENCE_SIZE=22, ZH_SENTENCE_SIZE=16):
    class_predictions = model.predict([df.iloc[:,:EN_SENTENCE_SIZE*2],df.iloc[:,EN_SENTENCE_SIZE*2:]])
    label_predictions = np.argmax(class_predictions, axis=1)
    return label_predictions

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--modeldir', help='Directory of the model to obtain metrics for', default='./best_model/BiLSTM.hdf5', type=str)
    parser.add_argument('-i','--inputcsv', help='Directory for the input csv', default='./data/test.csv', type=str)
    parser.add_argument('-o','--outputcsv', help='Directory for the output csv', default='./data/submission.csv', type=str)
    args = parser.parse_args()

    # Directories
    MODEL_DIR = args.modeldir
    TOKENIZER_DIR = './tokenizers/'
    INPUT_CSV = args.inputcsv

    # Load the necessary files
    t_EN,t_ZH = load_tokenizers(TOKENIZER_DIR)
    df = pd.read_csv(INPUT_CSV,encoding='utf-8-sig',error_bad_lines=False)
    model = load_model(MODEL_DIR, compile=False)

    # Prepare data for preprocessing and return ids/clean data
    ids,df = prepare(df,t_EN,t_ZH)

    # Get predictions of the dataset
    label_predictions = get_predictions(df,model)
    df_predictions = pd.DataFrame({'id':ids, 'label':label_predictions})

    # Unencode the labels
    map_dict = {0: 'unrelated', 1:'agreed', 2:'disagreed'}
    df_predictions['label'] = df_predictions["label"].map(map_dict)

    # Save predictions to a CSV
    df_predictions.to_csv(args.outputcsv, index=False)