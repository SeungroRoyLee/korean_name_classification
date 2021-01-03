import numpy as np
import pandas as pd
import csv
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout
from keras.models import Sequential
from keras.models import load_model

import os
# from nc.py import tag_origin, name_matrix, set_flag

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

# Loading Model
model = load_model("origin_model_entire.h5")
maxlen = 30

fileDir = os.path.abspath(os.path.dirname(__file__))
char_index_file = os.path.join(fileDir, "char_index.csv")


# Reading char_index 
fileDir = os.path.abspath(os.path.dirname(__file__))
char_index_file = os.path.join(fileDir, "char_index.csv")

import csv
reader = csv.reader(open(char_index_file, 'r'))
char_index= {}
for row in reader:
   k, v = row
   char_index[k[-1:]] = int(v)

# Replacing dictionary key
char_index["END"] = char_index.pop("D")

# Modify names to check
korean_name = ["seungro lee", "miyoun song", "mingeun lee", "jaemin kim", "bora kim", "jaein moon", "myungbak lee", "seonho kim", "jaeho park", "domin lee", "eunjung jung", "sangeun shin", "hyunjae cho"]
english_name = ["walt disney", "steve jobs", "donald trump", "andrew yang", "robert downey", "steve rogers", "nelson mandela", "maria hernandez", "alice smith", "martin luther", "jesus christ"]
check_name = []


while True:
    check_name.append(input("Add a name to check (Last name + First name): ").lower())
    if input("Do you want to add more name? (y/n): ") == "n":
        print("")
        break

# Truncate and make name matrix
trunc_name_kor = [i[0:maxlen] for i in korean_name]
trunc_name_eng = [i[0:maxlen] for i in english_name]
trunc_name_check = [i[0:maxlen] for i in check_name]
X_kor = name_matrix(trunc_name_kor, char_index, maxlen)
X_eng = name_matrix(trunc_name_eng, char_index, maxlen)
X_check = name_matrix(trunc_name_check, char_index, maxlen)

# Print out predicted results
pred_kor = model.predict(np.asarray(X_kor))
print("pred_kor is:")
for i in range(0,len(pred_kor)):
    print(korean_name[i]," ",pred_kor[i])
print("")

pred_eng = model.predict(np.asarray(X_eng))
print("pred_eng is:")
for i in range(0,len(pred_eng)):
    print(english_name[i]," ",pred_eng[i])
print("")

pred_check = model.predict(np.asarray(X_check))
print("pred_check is:")
for i in range(0,len(pred_check)):
    print(check_name[i]," ",pred_check[i])
print("")


