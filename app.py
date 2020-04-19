#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.python.keras.models import model_from_json, Sequential, load_model
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import GRU, Input, Dense, Flatten, Dropout


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1750)])
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# app
app = Flask(__name__)

# load model
model = load_model('model2same.h5')
#graph = tf.compat.v1.get_default_graph()


accepted_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÅÑÒÖàáâãäåçèéêëìíîïñòóôõöøùúüýÿčńŌōŕřšūž–'
word_vec_length = 25
char_vec_length = 95
output_labels = 2
char_to_int = dict((c, i) for i, c in enumerate(accepted_chars))
int_to_char = dict((i, c) for i, c in enumerate(accepted_chars))

# Removes all non accepted characters


def normalize(line):
    return [c.lower() for c in line if c.lower() in accepted_chars]
# Returns a list of n lists with n = word_vec_length


def name_encoding(name):
    # Encode input data to int, e.g. a->1, z->26
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


# routes
# @app.route('/predict', methods=['GET', 'POST'])
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = (request.get_json(force=True))
    print(data)
    test = np.asarray([np.asarray(name_encoding(normalize(name)))
                       for name in data]).astype(np.float32)

    result = model.predict(np.asarray(test))
    print(result)
    #output = int(result * 100)
    output = {'reslut': int(result * 100)}
    print(output)

    temp = ''.join(data)
    predictoin = np.squeeze(result * 100)
    if predictoin > 50:
        print("I am", predictoin, "sure", '"{}"'.format(temp), "is a name")
    else:
        print("I am", (100 - predictoin), '"{}"'.format(temp), "is a surname")

    # return data
    # return jsonify(results=output)
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=False)


# In[ ]:


# local url
url = 'http://127.0.0.1:5000'  # change to your url


# In[1]:


# pip freeze > requirements.txt


# In[ ]:


# In[ ]:


# kill -9 $(ps -A | grep python | awk '{print $1}')
