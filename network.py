from keras.layers.core import Dense, Activation, Dropout
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import pandas as pd
import numpy as np
import json
import os

def get_output(i, n):
    tmp = np.zeros(n)
    tmp[i] = 1
    return tmp

def load_training():
    data = pd.read_csv("names.csv")
    max_len = data.name.map(len).max()

    vocab = set(' '.join([str(i) for i in data['name']]))
    vocab.add('END')
    char_index = dict((c, i) for i, c in enumerate(vocab))
    vocab_len = len(vocab)

    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]

    train_X = train['name']
    train_X = [list(i)+['END']*(max_len-len(i)) for i in train_X]
    train_X = [[get_output(char_index[j], vocab_len) for j in i] for i in train_X]
    train_Y = [([1,0] if i=='M' else [0,1]) for i in train['gender']]
    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)

    test_X = test['name']
    test_X = [list(i)+['END']*(max_len-len(i)) for i in test_X]
    test_X = [[get_output(char_index[j], vocab_len) for j in i] for i in test_X]
    test_Y = [([1,0] if i=='M' else [0,1]) for i in test['gender']]
    test_X = np.asarray(test_X)
    test_Y = np.asarray(test_Y)

    return train_X, train_Y, test_X, test_Y, int(max_len), int(vocab_len), char_index

def save_model(model, data, network_path):
    if not os.path.exists(network_path):
        os.makedirs(network_path)
    open(os.path.join(network_path, 'data.json'), 'w').write(json.dumps(data))
    open(os.path.join(network_path, 'architecture.json'), 'w').write(model.to_json())
    model.save_weights(os.path.join(network_path, 'weights.h5'), overwrite=True)

def read_model(network_path):
    data = json.loads(open(os.path.join(network_path, 'data.json')).read())
    model = model_from_json(open(os.path.join(network_path, 'architecture.json')).read())
    model.load_weights(os.path.join(network_path, 'weights.h5'))
    return model, data

def main():
    train_X, train_Y, test_X, test_Y, max_len, vocab_len, char_index = load_training()

    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(max_len, vocab_len)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, batch_size=128, nb_epoch=30, validation_data=(test_X, test_Y))

    loss, acc = model.evaluate(test_X, test_Y)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    # saving model
    data = {"max_len": max_len, "vocab_len": vocab_len, "char_index": char_index}
    save_model(model, data, 'model')

    # model, data = read_model('model')
    # max_len = data['max_len']
    # vocab_len = data['vocab_len']
    # char_index = data['char_index']

    # testing on new data...
    test_names = ['omar', 'george', 'alexandre', 'julie', 'nisrine', 'charlotte']
    test_names = [list(i)+['END']*(max_len-len(i)) for i in test_names]
    test_names = [[get_output(char_index[j], vocab_len) for j in i] for i in test_names]
    test_names = np.asarray(test_names)
    print(model.predict(test_names))

if __name__ == '__main__':
    main()
