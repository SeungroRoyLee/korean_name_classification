from __future__ import print_function

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout
from keras.models import Sequential

# Determine how long names are
maxlen = 30
labels = 2

# Import csv file
input = pd.read_csv("origin_in.csv")
    # input.columns = ['name', 'n_or_f', 'namelen']

# Shows the break down of native and foreign 
input.groupby('n_or_f')['name'].count()

# Extract columns and Create Dictionary with Character & Index
names = input['name']
origin = input['n_or_f']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)
#print(len_vocab)
char_index = dict((c, i) for i, c in enumerate(vocab))
print(char_index)
char_df = pd.DataFrame.from_dict(char_index, orient = 'index')
char_df.to_csv("char_index.csv")
# Randomly break the data into training and testing
msk = np.random.rand(len(input)) < 0.8
    #print("msk : %s" %msk)
train = input[msk]
test = input[~msk]

# Tag if a name is native or foreign
def tag_origin(n_or_f):
    result = []
    for elem in n_or_f:
        if elem == 'n':
            result.append([1, 0])
        else:
            result.append([0, 1])
    return result

# Create an array of arrays (matrix) of names represented based on index of each character
def name_matrix(trunc_name_input, char_index_input, maxlen_input):
    result = []
    for i in trunc_name_input:
        tmp = [set_flag(char_index_input[j]) for j in str(i)]
        for k in range(0, maxlen_input - len(str(i))):
            tmp.append(set_flag(char_index_input["END"]))
        result.append(tmp)
    return result

# Within a zeros vector, change 0 to 1 if a character matches 
def set_flag(i):
    tmp = np.zeros(56)
    tmp[i] = 1
    return tmp

# Run name_matrix and tag_origin function on both training and testing data
trunc_train_name = [str(i)[0:maxlen] for i in train.name]
train_X = name_matrix(trunc_train_name, char_index, maxlen)
train_Y = tag_origin(train.n_or_f)

trunc_test_name = [str(i)[0:maxlen] for i in test.name]
#print(trunc_test_name)
test_X = name_matrix(trunc_test_name, char_index, maxlen)
test_Y = tag_origin(test.n_or_f)
#print(test_X)

# Build a machine learning model (LSTM architecture)
model = Sequential()
    #model.add(LSTM(32, return_sequences=True, input_shape=(maxlen, len_vocab)))
model.add(LSTM(32, return_sequences=True, input_shape=(maxlen, 56)))
model.add(Dropout(0.15))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.15))
model.add(Dense(2))

    #model.add(Flatten())
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #batch_size = 182415
batch_size = 32
    #model.fit(train_X, train_Y, batch_size= batch_size , epochs=10, validation_data=(test_X, test_Y))
model.fit(np.array(train_X), np.array(train_Y), batch_size= batch_size , epochs=15, validation_data=(np.array(test_X), np.array(test_Y)))

# Run predictions on given array of numbers
korean_name = ["miyoun song","park hyunjoon", "lee seungro", "park jaeho", "kim seonho", "lee mingeun", "namgung hwangsil", "kim bomi", "moon jaein", "lee myungbak"]
english_name = ["smith alice", "disney walt", "jobs steve", "trump donald", "hernandez maria", "rogers steve", "rodrigues maria","mandela nelson", "king luther", "downey robert"]
trunc_name_kor = [i[0:maxlen] for i in korean_name]
trunc_name_eng = [i[0:maxlen] for i in english_name]
X_kor = name_matrix(trunc_name_kor, char_index, maxlen)
X_eng = name_matrix(trunc_name_eng, char_index, maxlen)
pred_kor = model.predict(np.asarray(X_kor))
print("pred_kor is:")
print(pred_kor)
pred_eng = model.predict(np.asarray(X_eng))
print("pred_eng is:")
print(pred_eng)

# Print out test score and test accuracy of the model
score, acc = model.evaluate(np.array(test_X), np.array(test_Y))
print('Test score:', score)
print('Test accuracy:', acc)

# Save our model and data
model.save_weights('origin_model.h5', overwrite=True)
model.save('origin_model_entire.h5')
train.to_csv("train_split.csv")
test.to_csv("test_split.csv")

# Perform prediction on test for evaluation
evals = model.predict(np.array(test_X))
prob_n = [i[0] for i in evals]

# Create dataframe to output the results of evaluation
out = pd.DataFrame(prob_n)
out['name'] = test.name.reset_index()['name']
out['n_or_f'] = test.n_or_f.reset_index()['n_or_f']

out.head(10)
out.columns = ['prob_n', 'name', 'actual']
out.head(10)
out.to_csv("origin_pred_out.csv")