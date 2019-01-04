"""
RNN able to predict gender from name
"""
import os
import json

from keras.layers import Dense, LSTM
from keras.layers.core import Activation, Dropout
from keras.models import Sequential, model_from_json

import numpy as np
import pandas as pd

def get_output(i, n):
    """
    Return vector of n "zeros" and a "one" at index i
    """
    tmp = np.zeros(n)
    tmp[i] = 1
    return tmp

def load_training(csv_file):
    """
    Load names.csv with columns=['name', 'gender']
    """
    data = pd.read_csv(csv_file)
    max_len = data.name.map(len).max()

    vocab = set(' '.join([str(i) for i in data['name']]))
    vocab.add('END')
    char_index = dict((c, i) for i, c in enumerate(vocab))
    vocab_len = len(vocab)

    msk = np.random.rand(len(data)) < 0.8
    data_train = data[msk]
    data_test = data[~msk]

    train_x = data_train['name']
    train_x = [list(i)+['END']*(max_len-len(i)) for i in train_x]
    train_x = [[get_output(char_index[j], vocab_len) for j in i] for i in train_x]
    train_y = [([1, 0] if i == 'M' else [0, 1]) for i in data_train['gender']]
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    test_x = data_test['name']
    test_x = [list(i)+['END']*(max_len-len(i)) for i in test_x]
    test_x = [[get_output(char_index[j], vocab_len) for j in i] for i in test_x]
    test_y = [([1, 0] if i == 'M' else [0, 1]) for i in data_test['gender']]
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    return train_x, train_y, test_x, test_y, int(max_len), int(vocab_len), char_index

def save_model(model, data, network_path):
    """
    Save model architecture and weights into files
    """
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    open(os.path.join(network_path, 'data.json'), 'w').write(json.dumps(data))
    open(os.path.join(network_path, 'architecture.json'), 'w').write(model.to_json())
    model.save_weights(os.path.join(network_path, 'weights.h5'), overwrite=True)

def read_model(network_path):
    """
    Load model architecture and weights from files
    """
    data = json.loads(open(os.path.join(network_path, 'data.json')).read())
    model = model_from_json(open(os.path.join(network_path, 'architecture.json')).read())
    model.load_weights(os.path.join(network_path, 'weights.h5'))
    return model, data

def train():
    """
    Train the model
    """
    # load training data
    train_x, train_y, test_x, test_y, max_len, vocab_len, char_index = load_training('names.csv')

    # build model
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(max_len, vocab_len)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train model
    model.fit(train_x, train_y, batch_size=128, nb_epoch=30, validation_data=(test_x, test_y))

    # test model
    loss, acc = model.evaluate(test_x, test_y)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    # save model
    data = {"max_len": max_len, "vocab_len": vocab_len, "char_index": char_index}
    save_model(model, data, 'model')

def test():
    """
    Load model from files and test it
    """
    # read trained model
    model, data = read_model('model')
    max_len = data['max_len']
    vocab_len = data['vocab_len']
    char_index = data['char_index']

    # testing data
    test_names = ['Omar', 'George', 'Alexandre', 'Julie', 'Nisrine', 'Charlotte']
    test_names = [s.lower() for s in test_names]
    test_names = [list(i)+['END']*(max_len-len(i)) for i in test_names]
    test_names = [[get_output(char_index[j], vocab_len) for j in i] for i in test_names]
    test_names = np.asarray(test_names)

    # run model
    out = model.predict(test_names)
    labels = [('M' if p[0] > p[1] else 'F') for p in out]
    print(labels)

if __name__ == '__main__':
    train()
    test()
