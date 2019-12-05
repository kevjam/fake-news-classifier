from utils.tokenizing import encode_labels, segment_zh_data, create_en_tokenizer, create_zh_tokenizer, tokenize
from utils.storage import save_pickle, save_data

# -------------- General Packages --------------
# General Use
import pandas as pd
import numpy as np

# Removes punctuation and other useless symbols from the dataset
def filter_en(data):
    EN_replacements = {'-':' ', '&':' and ', '< / [a-zA-Z] >': '', '< [a-zA-Z] >':'', '[^a-zA-Z0-9 ]':'', ' s ':' ', '  ':' '}
    for replacee in EN_replacements: data = data.str.replace(replacee,EN_replacements[replacee])
    return data

# Removes punctuation/non-chinese characters from the Chinese dataset
def filter_zh(data):
    return data.str.replace('[^㕛-马0-9 ]','')

# Replaces empty strings with nan values
def emptystr_to_nan(data):
    return data.replace('',np.nan, inplace=True)

# Filters corrupted, unusable/unused data from the dataset
def filter_dataset(df, drop_columns=['id','tid1','tid2'], label_exists=True):
    df = df.drop(columns=drop_columns)
    df[['title1_en','title2_en']] = df[['title1_en','title2_en']].apply(filter_en)   
    df[['title1_zh','title2_zh']] = df[['title1_zh','title2_zh']].apply(filter_zh)
    df[['title1_en','title2_en','title1_zh','title2_zh']].apply(emptystr_to_nan)    
    if label_exists: df = df[df.label.isin(['unrelated','agreed','disagreed'])] # Remove rows with no label
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Remove Unnamed columns
    df = df.dropna() # Drop rows with null values
    return df

# Returns the mean title size + one standard deviation as the size to pad/truncate titles to.
# Separate sizes for both the EN and ZH sets.
def get_pad_sizes(df_train,df_test,std=1):
    EN_TITLE_LENGTHS = df_train['title1_en'].append(df_train['title2_en']).str.split().apply(len)
    ZH_TITLES_LENGTHS = df_train['title1_zh'].append(df_train['title2_zh']).str.split().apply(len)
    EN_SENTENCE_MEAN,ZH_SENTENCE_MEAN = np.mean(EN_TITLE_LENGTHS),np.mean(ZH_TITLES_LENGTHS)
    EN_SENTENCE_STD,ZH_SENTENCE_STD = np.std(EN_TITLE_LENGTHS),np.std(ZH_TITLES_LENGTHS)
    EN_SENTENCE_SIZE = round(EN_SENTENCE_MEAN + EN_SENTENCE_STD)
    ZH_SENTENCE_SIZE = round(ZH_SENTENCE_MEAN + ZH_SENTENCE_STD)
    return EN_SENTENCE_SIZE,ZH_SENTENCE_SIZE

# Performs all of the necessary preprocessing, may take several minutes to complete.
# Saves two tokenizers and clean feature/label split data.
def preprocess_data(df_train,df_test,SPLIT_DATA_DIR='./split_data/', EN_NUM_WORDS=35000,ZH_NUM_WORDS=70000,OOV_TOKEN='<UNK>'):
    print('\nCleaning datasets. . .')
    df_train = filter_dataset(df_train)
    df_test = filter_dataset(df_test)
    
    print('\nSegmenting Chinese text. . .')
    df_train = segment_zh_data(df_train)
    df_test = segment_zh_data(df_test)

    print('\nCreating tokenizers. . .')
    EN_SENTENCE_SIZE,ZH_SENTENCE_SIZE = get_pad_sizes(df_train,df_test)
    t_EN= create_en_tokenizer(df_train, num_words=EN_NUM_WORDS, oov_token=OOV_TOKEN)
    t_ZH = create_zh_tokenizer(df_train, num_words=ZH_NUM_WORDS, oov_token=OOV_TOKEN)
    
    print('\nEncoding labels. . .')
    df_train = encode_labels(df_train)
    df_test = encode_labels(df_test)
    
    print('\nTokenizing datasets. . .')
    df_train = tokenize(t_EN,t_ZH,df_train,EN_SENTENCE_SIZE,ZH_SENTENCE_SIZE)
    df_test = tokenize(t_EN,t_ZH,df_test,EN_SENTENCE_SIZE,ZH_SENTENCE_SIZE)
    
    print('\nSaving data. . .')
    save_data(train=df_train, test=df_test,
              en_maxlen=EN_SENTENCE_SIZE, zh_maxlen=ZH_SENTENCE_SIZE,
              save_dir=SPLIT_DATA_DIR)
    print('\nPreprocessing complete.')

if __name__== "__main__":
    # Directories
    input_train = './data/train.csv'
    input_validation = './data/validataion.csv'

    # Loading the data
    df_train = pd.read_csv(input_train,encoding='utf-8-sig',error_bad_lines=False)
    df_test = pd.read_csv(input_validation,encoding='utf-8-sig',error_bad_lines=False)

    preprocess_data(df_train,df_test)