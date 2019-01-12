# Gender Prediction

RNN trained to predict gender from French name. Names database is from [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/liste-des-prenoms-2004-a-2017/).

# Load pre-trained model

A pretrained model is in the `model` folder. You can load it as follow :

```python
import os, json, numpy
from network import read_model, vector
from keras.models import load_model

model, data = read_model('model')
max_len = data['max_len']
vocab_len = data['vocab_len']
char_index = data['char_index']

test_names = ['Bob', 'Alice']
test_names = [s.lower() for s in test_names]
test_names = [list(i)+['END']*(max_len-len(i)) for i in test_names]
test_names = [[vector(char_index[j], vocab_len) for j in i] for i in test_names]
test_names = numpy.asarray(test_names)

out = model.predict(test_names)
print(out)
```

# GenderAPI

You can use `GenderAPI` wrapper.

```python
from gender_api import GenderAPI
api = GenderAPI()
names = ['Bob', 'Alice']
labels = api.predict(names)
```

# Run GenderAPI with Flask

Run the flask server :

```
python server.py
```

Test it with curl :

```
curl -H 'Content-Type: application/json' --data '["Bob", "Alice"]' localhost:4000/predict
```

# Run GenderAPI in Docker

Build container,

```
docker-compose build
```

Then,

```
curl -H 'Content-Type: application/json' --data '["Bob", "Alice"]' localhost:4000/predict
```
