import os
import json
import numpy as np

from keras.models import load_model
import tensorflow as tf

class GenderAPI:
    def __init__(self):
        self.graph = tf.get_default_graph()
        self.model, data = self.read_model('model')
        self.max_len = data['max_len']
        self.vocab_len = data['vocab_len']
        self.char_index = data['char_index']

    def vector(self, i, n):
        tmp = np.zeros(n)
        tmp[i] = 1
        return tmp

    def read_model(self, network_path):
        if not os.path.exists(network_path):
            raise ValueError('Path not found : {}'.format(network_path))
        dat = json.loads(open(os.path.join(network_path, 'data.json')).read())
        mod = load_model(os.path.join(network_path, 'rnn.model'))
        return mod, dat

    def predict(self, names, labelize=True):
        """
        Returns gender of given names

        Args:
            names:      list of strings
            labelize:   returns 'M' or 'F' labels if set to True,
                        returns list of porbabilities otherwise
        """
        with self.graph.as_default():
            # format input
            names = [s.lower() for s in names]
            names = [list(i)+['END']*(self.max_len-len(i)) for i in names]
            names = [[self.vector(self.char_index[j], self.vocab_len) for j in i] for i in names]
            names = np.asarray(names)

            # predict gender
            out = self.model.predict(names).tolist()
            return [('M' if p[0] > p[1] else 'F') for p in out] if labelize else out
