# For parsing command-line arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-env','--envocab', help='Specify English vocab size.', default=35000, type=int)
parser.add_argument('-zhv','--zhvocab', help='Specify English vocab size.', default=70000, type=int)
parser.add_argument('-em','--embedding', help='Specify embedding layer size.', default=50, type=int)
parser.add_argument('-r','--recurrent', help='Specify recurrent layer size.', default=100, type=int)
parser.add_argument('-rl','--recurrentlayers', help='Specify the number of recurrent layers.', default=1, type=int)
parser.add_argument('-e','--epochs', help='Specify the number of epochs.', default=10, type=int)
parser.add_argument('-b','--batchsize', help='Specify the batch size.', default=1024, type=int)
parser.add_argument('-p','--patience', help='Specify the patience for EarlyStopping.', default=3, type=int)
parser.add_argument('-sd','--spatialdropout', help='Specify the spatial dropout size.', default=0.75, type=float)
parser.add_argument('-rd','--recurrentdropout', help='Specify the recurrent dropout size in RNN layers.', default=0.25, type=float)
args = parser.parse_args()

# For Saving/Loading Files
from utils.storage import load_data,load_tokenizers

# -------------- Modelling Packages --------------
# For modeling
from keras.models import Model, Sequential
from keras.layers import Embedding, LSTM, Bidirectional
from keras.layers import Input, Reshape, SpatialDropout1D, Dense, Flatten
from keras.layers import Concatenate
from keras import optimizers

# Callback Functions
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# For Timestamping Models
import time

# For balancing class weights
from sklearn.utils import class_weight

# -------------- General Packages --------------
# Data Manipulation
import numpy as np

# Saving/Loading
import os

# Directories
MODEL_DIR = './models/'
SPLIT_DATA_DIR = './split_data/'
LOG_DIR = 'logs'
TOKENIZER_DIR = './tokenizers/'

# Loading the dataset/tokenizers
EN_X_train,EN_X_test,ZH_X_train,ZH_X_test,y_train,y_test = load_data(SPLIT_DATA_DIR)
t_EN,t_ZH = load_tokenizers(TOKENIZER_DIR)

# Making the models directory if not already made
os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)

# -------------- Tokenizer Values --------------
EN_SENTENCE_SIZE = int(EN_X_train.shape[1]/2)
ZH_SENTENCE_SIZE = int(ZH_X_train.shape[1]/2)
en_vocab_size = args.envocab
zh_vocab_size = args.zhvocab

# -------------- TUNABLE HYPERPARAMETERS --------------
# Balancing class weights
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

EMBED_SIZE = args.embedding
RNN_SIZE = args.recurrent
RNN_LAYERS = args.recurrentlayers
SPATIAL_DROPOUT = args.spatialdropout
RECURRENT_DROPOUT = args.recurrentdropout

loss = 'sparse_categorical_crossentropy'
optimizer = optimizers.adam()
metrics = ['accuracy']

epochs = args.epochs
batch_size = args.batchsize
PATIENCE = args.patience
                
# -------------- MODEL NAMING --------------
NAME = 'BiLSTM-{}E-{}x{}L-({}, {})Dropout-{}.hdf5'.format(EMBED_SIZE,
                                                        RNN_LAYERS,RNN_SIZE,
                                                        SPATIAL_DROPOUT,RECURRENT_DROPOUT,
                                                        time.time())
print('Creating {}'.format(NAME))
MODEL_LOG_DIR = os.path.join(LOG_DIR,NAME)

# -------------- Callbacks --------------
# access tensorboard from the command line: tensorboard --logdir=logs/
tensorboard = TensorBoard(log_dir=MODEL_LOG_DIR) 
checkpointer = ModelCheckpoint(MODEL_DIR+NAME, 
                                monitor='val_accuracy', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='auto')
earlystop = EarlyStopping(monitor='val_loss', patience=PATIENCE)

callbacks=[tensorboard,checkpointer,earlystop]

# -------------- EN MODEL CREATION --------------
EN_INPUT = Input(shape=(EN_SENTENCE_SIZE*2,))
EN_MODEL = Reshape((-1,2,EN_SENTENCE_SIZE))(EN_INPUT)
EN_MODEL = Embedding(en_vocab_size,
                EMBED_SIZE,
                input_shape=(2,EN_SENTENCE_SIZE),
                trainable=True)(EN_MODEL)
EN_MODEL = Reshape((2,EN_SENTENCE_SIZE*EMBED_SIZE,))(EN_MODEL)

if SPATIAL_DROPOUT > 0: EN_MODEL = SpatialDropout1D(SPATIAL_DROPOUT)(EN_MODEL)
for layer in range(RNN_LAYERS-1):
    EN_MODEL = Bidirectional(LSTM(RNN_SIZE,return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT))(EN_MODEL)
EN_MODEL = Bidirectional(LSTM(RNN_SIZE,return_sequences=True,recurrent_dropout=RECURRENT_DROPOUT))(EN_MODEL)

# -------------- ZH MODEL CREATION --------------
ZH_INPUT = Input(shape=(ZH_SENTENCE_SIZE*2,))
ZH_MODEL = Reshape((-1,2,ZH_SENTENCE_SIZE))(ZH_INPUT)
ZH_MODEL = Embedding(zh_vocab_size,
                EMBED_SIZE,
                input_shape=(2,ZH_SENTENCE_SIZE),
                trainable=True)(ZH_MODEL)
ZH_MODEL = Reshape((2,ZH_SENTENCE_SIZE*EMBED_SIZE,))(ZH_MODEL)

if SPATIAL_DROPOUT > 0: ZH_MODEL = SpatialDropout1D(SPATIAL_DROPOUT)(ZH_MODEL)
for layer in range(RNN_LAYERS-1):
    ZH_MODEL = Bidirectional(LSTM(RNN_SIZE,return_sequences=True, recurrent_dropout=RECURRENT_DROPOUT))(ZH_MODEL)
ZH_MODEL = Bidirectional(LSTM(RNN_SIZE,return_sequences=True,recurrent_dropout=RECURRENT_DROPOUT))(ZH_MODEL)
    
# -------------- MERGE MODEL --------------
merged = Concatenate(1)([EN_MODEL,ZH_MODEL])
merged = LSTM(RNN_SIZE,return_sequences=True,recurrent_dropout=RECURRENT_DROPOUT)(merged)
merged = Flatten()(merged)
merged = Dense(3, activation='softmax')(merged)

model = Model(inputs=[EN_INPUT,ZH_INPUT], outputs=merged)
model.compile(optimizer=optimizer, loss=loss,metrics=metrics)

# -------------- Training the model --------------
model.fit([EN_X_train,ZH_X_train], y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([EN_X_test,ZH_X_test], y_test),
            callbacks=callbacks,
            class_weight=class_weights)