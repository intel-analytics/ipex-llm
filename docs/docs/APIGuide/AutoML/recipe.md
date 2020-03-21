# Recipe

You can use `Recipe` to choose some presets for the `TimeSequencePredictor` fitting process by passing to the `recipe` field in the `fit` method.

### SmokeRecipe
A very simple Recipe for smoke test that runs one epoch and one iteration with only 1 random sample.

```python
SmokeRecipe()
```

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
