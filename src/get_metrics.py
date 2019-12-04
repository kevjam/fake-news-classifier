# For parsing command-line arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--modeldir', help='Directory of the model to obtain metrics for', default='./best_model/BiLSTM.hdf5', type    =str)
args = parser.parse_args()

from utils.storage import load_data

# -------------- Model Evaluation Packages --------------
from keras.models import load_model
from sklearn.metrics import classification_report, f1_score

# -------------- General Packages --------------
# Data Manipulation
import numpy as np

# Directories
MODEL_DIR = args.modeldir
SPLIT_DATA_DIR = './split_data/'
TOKENIZER_DIR = './tokenizers/'

# Loading the dataset
EN_X_train,EN_X_test,ZH_X_train,ZH_X_test,y_train,y_test = load_data(SPLIT_DATA_DIR)

# Loading the keras model
print('\nLoading model {}'.format(MODEL_DIR))
model = load_model(MODEL_DIR, compile=False)

# Getting predictions
print('\nGetting predictions...')
class_predictions = model.predict([EN_X_test,ZH_X_test])
label_predictions = np.argmax(class_predictions, axis=1)

# Printing the classification report
print('\nClassification report for {}'.format(MODEL_DIR))
print(classification_report(y_test,label_predictions, digits=4))