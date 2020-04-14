## Automated Time Series Prediction 



### Training a model using _TimeSequencePredictor_

_TimeSequencePredictor_ can be used to train a model on historical time sequence data and predict future sequences. Note that:   
  * We require input time series data to be uniformly sampled in timeline. Missing data points will lead to errors or unreliable prediction result. 

#### 1. Before training, init RayOnSpark.   

* Run ray on spark local mode, Example
     
```python
from zoo import init_spark_on_local
from zoo.ray import RayContext
sc = init_spark_on_local(cores=4)
ray_ctx = RayContext(sc=sc)
ray_ctx.init()
```
   
* run ray on yarn cluster, Example  
      
```python
from zoo import init_spark_on_yarn
from zoo.ray import RayContext
slave_num = 2
sc = init_spark_on_yarn(
        hadoop_conf=args.hadoop_conf,
        conda_name="ray36",
        num_executor=slave_num,
        executor_cores=4,
        executor_memory="8g ",
        driver_memory="2g",
        driver_cores=4,
        extra_executor_memory_for_ray="10g")
ray_ctx = RayContext(sc=sc, object_store_memory="5g")
ray_ctx.init()
```

#### 2. Create a _TimeSequencePredictor_

* `dt_col` and `target_col` are datetime cols and target column in the input dataframe 
* `future_seq_len` is how many data points ahead to predict. 
    
```python
from zoo.automl.regression.time_sequence_predictor import TimeSequencePredictor
tsp = TimeSequencePredictor(dt_col="datetime", target_col="value", extra_features_col=None, future_seq_len=1)
```

#### 3. Train on historical time sequence. 

* ```recipe``` contains parameters to control the search space, stop criteria and number of samples (e.g. for random search strategy, how many samples are taken from the search space). Some recipe with large number of samples may lead to a large trial pool and take very long time to finish. Current avaiable recipes are: _SmokeRecipe_, _RandomRecipe_, _GridRandomRecipe_ and _BayesRecipe_. _SmokeRecipe_ is a very simple Recipe for smoke test that runs one epoch and one iteration with only 1 random sample. Other recipes all have arguments ```num_random_samples``` and ```look_back```. ```num_random_samples``` is used to control the number of samples. Note that for GridRandomRecipe, the actual number of trials generated will be 2*```num_samples```, as it needs to do a grid search from 2 possble values for every random sample. ```look_back``` is the length of sequence you want to look back. The default values is 1. You can either put a tuple of (min_len, max_len) or a single int to control the look back sequence length search space. _BayesRecipe_ use bayesian-optimization package to perform sequential model-based hyperparameter optimization.
* ```distributed```specifies whether to run it in a distributed fashion. 
* ```fit``` returns a _Pipeline_ object (see next section for details). 
* Now we don't support resume training - i.e. calling ```fit``` multiple times retrains on the input data from scratch. 
* input train dataframe look like below: 

|datetime|value|
| --------|----- | 
|2019-06-06|1.2|
|2019-06-07|2.3|...|
  
```python
pipeline = tsp.fit(train_df, metric="mean_squared_error", recipe=RandomRecipe(num_samples=1), distributed=False)
```

#### 4. After training finished, stop RayOnSpark
 
```python
ray_ctx.stop()
```

### Saving and Loading a _TimeSequencePipeline_
* Save the _Pipeline_ object to a file

```python
pipeline.save("/tmp/saved_pipeline/my.ppl")
```
 
* Load the _Pipeline_ object from a file 
```python
from zoo.automl.pipeline.time_sequence import load_ts_pipeline
 
pipeline = load_ts_pipeline("/tmp/saved_pipeline/my.ppl")
```
 
### Prediction and Evaluation using _TimeSequencePipeline_ 

A _TimeSequencePipeline_ contains a chain of feature transformers and models, which does end-to-end time sequence prediction on input data. _TimeSequencePipeline_ can be saved and loaded for future deployment.      
 
* Prediction using _Pipeline_ object

Output dataframe look likes below (assume predict n values forward). col `datetime` is the starting timestamp.  

|datetime|value_0|value_1|...|value_{n-1}|
| --------|----- | ------|---|---- |
|2019-06-06|1.2|2.8|...|4.4|

```python
result_df = pipeline.predict(test_df)
```
 
* Evaluation using _Pipeline_ object

```python
#evaluate with MSE and R2 metrics
mse, rs = pipeline.evaluate(test_df, metrics=["mse", "rs"])
```

* Incremental training using _Pipeline_ object

```python
#fit with new data and train for 5 epochs
pipeline.fit(new_train_df,epoch_num=5)
```
