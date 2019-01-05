# Gender Prediction

RNN trained to predict gender from French name. Names database is from [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/liste-des-prenoms-2004-a-2017/).

# Pre-trained model

A pretrained model is in the `model` folder. You can load it as follow :

```python
model, data = read_model('model')
max_len = data['max_len']
vocab_len = data['vocab_len']
char_index = data['char_index']

test_names = ['George', 'Julie']
test_names = [s.lower() for s in test_names]
test_names = [list(i)+['END']*(max_len-len(i)) for i in test_names]
test_names = [[get_output(char_index[j], vocab_len) for j in i] for i in test_names]
test_names = np.asarray(test_names)

out = model.predict(test_names)
print(out)
```

# Docker

Using Flask you can easily make a GenderAPI. The image is on Docker.

```
docker pull omaraflak/genderapi:v1
```
