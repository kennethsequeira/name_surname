#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.layers import LSTM, GRU, Input, Dense, Flatten, Dropout, BatchNormalization, Embedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


# Import Keras objects for Deep Learning


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1750)])
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"], 'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"], 'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"], 'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],
            'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)


# Reading files
data_path = ['data']
filepath = os.sep.join(data_path + ['Churn_Modelling.csv'])
churn_Modelling = pd.read_csv(filepath, sep=',', engine='python')

filepath = os.sep.join(data_path + ['engwales_surnames.csv'])
engwales_surnames = pd.read_csv(filepath, sep=',', engine='python')

filepath = os.sep.join(data_path + ['languages-and-dialects-geo.csv'])
lan_names = pd.read_csv(filepath, sep=',', engine='python')

filepath = os.sep.join(data_path + ['NationalNames.csv'])
nationalNames = pd.read_csv(filepath, sep=',', engine='python')

filepath = os.sep.join(data_path + ['allFirstNames.txt'])
allFirstNames = pd.read_csv(filepath, sep=',', engine='python')

filepath = os.sep.join(data_path + ['allSurnames.txt'])
allSurnames = pd.read_csv(filepath, sep=',', engine='python')

# Removing everything exept name col
churn_Modelling = churn_Modelling.drop(
churn_Modelling.columns[[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], axis=1)
churn_Modelling.sample(5)


churn_Modelling_1 = pd.DataFrame(churn_Modelling.astype(str))
churn_Modelling_1['label'] = 0
churn_Modelling_1.rename(columns={'Surname': 'Name'}, inplace=True)
churn_Modelling_1.sample(5)


engwales_surnames1 = pd.DataFrame(engwales_surnames.astype(str))
engwales_surnames1 = engwales_surnames1.drop(
engwales_surnames1.columns[[1, 2]], axis=1)
engwales_surnames1['label'] = 0
engwales_surnames1.sample(5)


nationalNames1 = pd.DataFrame(nationalNames.astype(str))
nationalNames1['label'] = 1
nationalNames1 = nationalNames.drop(nationalNames.columns[[0, 2, 3]], axis=1)
nationalNames1.sample(5)


nationalNames1 = nationalNames.drop(
nationalNames.columns[[0, 2, 3, 4]], axis=1)
nationalNames1['label'] = 1
nationalNames1.sample(5)


lan_names1 = pd.DataFrame(lan_names.astype(str))
lan_names1 = lan_names1.drop(lan_names1.columns[[0, 2, 3, 4, 5, 6]], axis=1)
lan_names1['label'] = 1
lan_names1.rename(columns={'name': 'Name'}, inplace=True)
lan_names1.sample(5)


allFirstNames1 = pd.DataFrame(allFirstNames.astype(str))
allFirstNames1['label'] = 1
allFirstNames1.rename(columns={'Añaterve': 'Name'}, inplace=True)
allFirstNames1.sample(5)


allSurnames1 = pd.DataFrame(allSurnames.astype(str))
allSurnames1['label'] = 0
allSurnames1.rename(columns={'Ñeco': 'Name'}, inplace=True)
allSurnames1.sample(5)

#Checking if there are duplicates in names
Names = [allFirstNames1, lan_names1, nationalNames1]
result_Names = pd.concat(Names)

# Adding the label 
result_Names_NoDupl = (result_Names.Name).drop_duplicates()
result_Names_NoDupl = pd.DataFrame(result_Names_NoDupl)
result_Names_NoDupl['label'] = 1


#Checking if there are duplicates in surnames
Surnames = [allSurnames1, engwales_surnames1, churn_Modelling_1]
result_Surnames = pd.concat(Surnames)

# Adding the label 
result_Surnames_NoDupl = (result_Surnames.Name).drop_duplicates()
result_Surnames_NoDupl = pd.DataFrame(result_Surnames_NoDupl)
result_Surnames_NoDupl['label'] = 0

# Final dataframe
frames = [result_Surnames_NoDupl, result_Names_NoDupl]
result_nodup = pd.concat(frames)


# In the case of a middle name, we will simply use the first name only
result_nodup['Name'] = result_nodup['Name'].apply(lambda x: str(x).split(' ', 1)[0])


# we drop all name where len < 2
result_nodup.drop(
result_nodup[result_nodup['Name'].str.len() < 2].index, inplace=True)


onlyN = result_nodup.drop(columns=['label'])
onlyN

# Finding all characters
all_chars = {c: (set(''.join(onlyN[c]))) for c in onlyN.columns}
all_chars

k = ['a',
     'b',
     'c',
     'd',
     'e',
     'f',
     'g',
     'h',
     'i',
     'j',
     'k',
     'l',
     'm',
     'n',
     'o',
     'p',
     'q',
     'r',
     's',
     't',
     'u',
     'v',
     'w',
     'x',
     'y',
     'z',
     'à',
     'á',
     'â',
     'ã',
     'ä',
     'å',
     'ç',
     'è',
     'é',
     'ê',
     'ë',
     'ì',
     'í',
     'î',
     'ï',
     'ñ',
     'ò',
     'ó',
     'ô',
     'õ',
     'ö',
     'ø',
     'ù',
     'ú',
     'ü',
     'ý',
     'ÿ',
     'č',
     'ń',
     'ō',
     'ŕ',
     'ř',
     'š',
     'ū',
     'ž',
     '-',
     '–']


k10 = ''.join(k)

# accepted characters 
accepted_chars = k10


# Parameters
predictor_col = 'Name'
result_col = 'label'
# Length of the input vector
word_vec_length = min(result_nodup[predictor_col].apply(len).max(), 25)
# Length of the character vector
char_vec_length = len(accepted_chars)
# Number of output labels
output_labels = 2

print(f"The input vector will have the shape {word_vec_length}x{char_vec_length}.")

# Define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(accepted_chars))
int_to_char = dict((i, c) for i, c in enumerate(accepted_chars))

# Removes all non accepted characters
def normalize(line):
    return [c.lower() for c in line if c.lower() in accepted_chars]

# Returns a list of n lists with n = word_vec_length
def name_encoding(name):
    # Encode input data to int
    integer_encoded = [char_to_int[char]
                       for i, char in enumerate(name) if i < word_vec_length]

    # Start one-hot-encoding
    onehot_encoded = list()

    for value in integer_encoded:
        # create a list of n zeros, where n is equal to the number of accepted characters
        letter = [0 for _ in range(char_vec_length)]
        letter[value] = 1
        onehot_encoded.append(letter)

    # Fill up list to the max length. Lists need do have equal length to be able to convert it into an array
    for _ in range(word_vec_length - len(name)):
        onehot_encoded.append([0 for _ in range(char_vec_length)])

    return onehot_encoded


X = np.array(result_nodup["Name"])
y = np.array(result_nodup["label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11111, shuffle=True)

# Convert both the input names as well as the output lables into the machine readable vector format
X_train = np.asarray([np.asarray(name_encoding(normalize(name)))
                      for name in X_train])

X_test = np.asarray([np.asarray(name_encoding(normalize(name)))
                     for name in X_test])


# In[8]:


model = Sequential()
model.add(LSTM(512, return_sequences=False,
               input_shape=(word_vec_length, char_vec_length)))
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=18,
                    batch_size=1024,
                    validation_split=0.2)


# In[9]:


plot_loss_accuracy(history)


# In[20]:


namecheck = [input("Name or Surname: ")]
test = np.asarray([np.asarray(name_encoding(normalize(name)))
                   for name in namecheck]).astype(np.float32)
pred = model.predict(np.asarray(test))
predictoin = np.squeeze(pred * 100)
temp = ''.join(namecheck)
if predictoin > 50:
    print("I am", predictoin, "sure", '"{}"'.format(temp), "is a name")
else:
    print("I am", (100 - predictoin), '"{}"'.format(temp), "is a surname")


# In[21]:


# save our model and data
#result_nodup.to_csv("result_nodup.csv")
model.save("model3.h5")
print("Saved model to disk")


# In[ ]:





# In[ ]:





# In[ ]:




