# -------------- General Packages --------------
# For Saving/Loading Files
import pickle
import os

# Saves a file using pickle
def save_pickle(data,directory):
    pickle_out = open(os.path.join(directory+'.pickle'), 'wb')
    pickle.dump(data,pickle_out)
    pickle_out.close()

# Given train and test data, split into features and labels and save
def save_data(train, test, en_maxlen=20, zh_maxlen=20, save_dir='./split_data/'):
    # Save data to files
    split_data = {'EN_X_train':train.iloc[:,:en_maxlen*2], 
                  'EN_X_test':test.iloc[:,:en_maxlen*2], 
                  'ZH_X_train':train.iloc[:,en_maxlen*2:en_maxlen*2+zh_maxlen*2], 
                  'ZH_X_test':test.iloc[:,en_maxlen*2:en_maxlen*2+zh_maxlen*2], 
                  'y_train':train['label'].to_numpy(), 
                  'y_test':test['label'].to_numpy()}

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    for name,data in split_data.items():
        save_pickle(data,save_dir+name)

# Load a pickle file given a directory and return its contents
def load_pickle(directory):
    pickle_in = open(directory,'rb')
    return pickle.load(pickle_in)

# Given the split dataset directory, return the train/test split
def load_data(split_data_dir):
    EN_X_train = load_pickle(split_data_dir+'EN_X_train.pickle')
    EN_X_test = load_pickle(split_data_dir+'EN_X_test.pickle')
    ZH_X_train = load_pickle(split_data_dir+'ZH_X_train.pickle')
    ZH_X_test = load_pickle(split_data_dir+'ZH_X_test.pickle')
    y_train = load_pickle(split_data_dir+'y_train.pickle')
    y_test = load_pickle(split_data_dir+'y_test.pickle')
    return EN_X_train,EN_X_test,ZH_X_train,ZH_X_test,y_train,y_test
    
# Load both the English and Chinese tokenizers and return them
def load_tokenizers(TOKENIZER_DIR):
    t_EN = load_pickle(TOKENIZER_DIR+'en_tokenizer.pickle')
    t_ZH = load_pickle(TOKENIZER_DIR+'zh_tokenizer.pickle')
    return t_EN,t_ZH