# Recipe

You can use `Recipe` to choose some presets for the `TimeSequencePredictor` fitting process by passing to the `recipe` field in the `fit` method.

### SmokeRecipe
A very simple Recipe for smoke test that runs one epoch and one iteration with only 1 random sample.

```python
SmokeRecipe()
```

### LSTMRandomGridRecipe
A recipe mixing random and grid using LSTM Model only
```python
LSTMGridRandomRecipe(num_rand_samples=1,epochs=5,training_iteration=10,look_back=2,lstm_1_units=[16, 32, 64, 128],lstm_2_units=[16, 32, 64], batch_size=[32, 64])
```

#### Arguments
* :param lstm_1_units: random search candidates for num of lstm_1_units
* :param lstm_2_units: grid search candidates for num of lstm_1_units
* :param batch_size: grid search candidates for batch size
* :param num_rand_samples: number of hyper-param configurations sampled randomly
* :param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
* :param training_iteration: no. of iterations for training (n epochs) in trials
* :param epochs: no. of epochs to train in each iteration

### MTNetRandomGridRecipe

A recipe mixing random and grid using MTNet Model only

```python
MTNetGridRandomRecipe(num_rand_samples=1,
                 epochs=5,
                 training_iteration=10,
                 time_step=[3, 4],
                 filter_size=[2, 4],
                 long_num=[3, 4],
                 ar_size=[2, 3],
                 batch_size=[32, 64])
```
#### Arguments
* :param num_rand_samples: number of hyper-param configurations sampled randomly
* :param training_iteration: no. of iterations for training (n epochs) in trials
* :param epochs: no. of epochs to train in each iteration
* :param time_step: random search candidates for model param "time_step"
* :param filter_size: random search candidates for model param "filter_size"
* :param long_num: random search candidates for model param "long_num"
* :param ar_size: random search candidates for model param "ar_size"
* :param batch_size: grid search candidates for batch size

### RandomRecipe
Pure random sample Recipe. Often used as baseline.

```python
RandomRecipe(num_rand_samples=1, look_back=2)
```

#### Arguments

* **num_rand_samples**: number of hyper-param configurations sampled randomly.

* **look_back**: The length to look back. It could be

    - A single int, which is a fixed length to look back. Note that the look back value should be larger than 1 to take the series information into account.
    - A tuple with 2 int values, which is in format is (min len, max len). The `min len` value should also be larger than 1.


### GridRandomRecipe
A recipe involves both grid search and random search. The arguments are the same with `RandomRecipe`.

```python
GridRandomRecipe(num_rand_samples=1, look_back=2)
```

### BayesRecipe
A recipe to search with Bayes Optimization. You need to pre-install `bayesian-optimization` before using the recipe.

```python
BayesRecipe(num_samples=1, look_back=2)
```
