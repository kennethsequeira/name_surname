#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import flask
from tensorflow.python.keras.models import model_from_json, Sequential, load_model
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.layers import LSTM, Flatten, Dropout

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1750)])
# if 'session' in locals() and session is not None:
#     print('Close interactive session')
#     session.close()

# app
app = flask.Flask(__name__, template_folder='templates')

# Configure a secret SECRET_KEY
#app.config[‘SECRET_KEY’] = ‘someRandomKey’

# load model
model = load_model('model2same.h5')

accepted_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÅÑÒÖàáâãäåçèéêëìíîïñòóôõöøùúüýÿčńŌōŕřšūž–'
word_vec_length = 25
char_vec_length = 95
output_labels = 2
char_to_int = {c: i for i, c in enumerate(accepted_chars)}
int_to_char = {i: c for i, c in enumerate(accepted_chars)}

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

@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        name = flask.request.form['name']
        data = [name]

        test = np.asarray([np.asarray(name_encoding(normalize(name)))
                           for name in data]).astype(np.float32)

        result = model.predict(np.asarray(test))

        temp = ''.join(data)
        predictoin = np.squeeze(result * 100)
        if predictoin > 50:
            output = predictoin
            res = "name"
        else:
            output = (100 - predictoin)
            res = "surname"

        return flask.render_template('main.html',
                                     original_input={'name':name},
                                     result='{} % sure it is a '.format(int(output))+res
                                    )
                                     
if __name__ == '__main__':

    app.run()


# %%


# %%


# %%


# %%


# kill -9 $(ps -A | grep python | awk '{print $1}')
