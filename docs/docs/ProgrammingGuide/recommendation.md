Analytics Zoo provides two Recommender models, including Wide and Deep(WND) learning model and Neural network-based Collaborative Filtering (NCF) model. 

**Highlights**

1. Easy-to-use models, could be fed into NNFrames or BigDL Optimizer for training.
2. Recommenders can handle either explict or implicit feedback, given corresponding features.
3. It provides three user-friendly APIs to predict user item pairs, and recommend items (users) for users (items).

The examples/notebooks are included in the Analytics Zoo source code.

1. Wide and Deep Learning Model.
    [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/WideAndDeepExample.scala)
    [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-wide-n-deep/wide_n_deep.ipynb)
2. NCF.
    [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/NeuralCFexample.scala)
    [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-ncf/ncf-explicit-feedback.ipynb)

---
## Wide and Deep

**Scala**

Build a WND model for recommendation. 
```scala
val wideAndDeep = WideAndDeep(modelType = "wide_n_deep", numClasses, columnInfo, hiddenLayers = Array(40, 20, 10))
```
Train a WND model using BigDL Optimizer.
```scala
val optimizer = Optimizer(
      model = wideAndDeep,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 8000)

optimizer
      .setOptimMethod(new Adam[Float](learningRate = 1e-2,learningRateDecay = 1e-5))
      .setEndWhen(Trigger.maxEpoch(10))
      .optimize()
```
Predict and recommend items(users) for users(items) with given features.
```scala
val userItemPairPrediction = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 3)
val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 3)
``` 
See more details in our[Recommender API](../APIGuide/Models/recommendation.md) and [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/WideAndDeepExample.scala).

**Python**

Build a WND model for recommendation. 
```python
wide_n_deep = WideAndDeep(class_num, column_info, model_type="wide_n_deep", hidden_layers=(40, 20, 10))
```
Train a WND model using BigDL Optimizer 
```python
optimizer = Optimizer(
    model=wide_n_deep,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=Adam(learningrate = 0.001, learningrate_decay=0.00005),
    end_trigger=MaxEpoch(10),
    batch_size=batch_size)
optimizer.optimize() 
```
Predict and recommend items(users) for users(items) with given features.
```python
userItemPairPrediction = wide_n_deep.predict_user_item_pair(valPairFeatureRdds)
userRecs = wide_n_deep.recommend_for_user(valPairFeatureRdds, 3)
itemRecs = wide_n_deep.recommend_for_item(valPairFeatureRdds, 3)
``` 
See more details in our [Recommender API](../APIGuide/Models/recommendation.md) and [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-wide-n-deep/wide_n_deep.ipynb).

---
## Neural network-based Collaborative Filtering

**Scala**

Build a NCF model for recommendation. 
```scala
val ncf = NeuralCF(userCount, itemCount, numClasses, userEmbed = 20, itemEmbed = 20, hiddenLayers = Array(40, 20, 10), includeMF = true, mfEmbed = 20)
```
Train a NCF model using BigDL Optimizer 
```scala
val optimizer = Optimizer(
      model = ncf,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 8000)

optimizer
      .setOptimMethod(new Adam[Float](learningRate = 1e-2,learningRateDecay = 1e-5))
      .setEndWhen(Trigger.maxEpoch(10))
      .optimize()
```
Predict and recommend items(users) for users(items) with given features.
```scala
val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)
val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)
``` 
See more details in our[Recommender API](../APIGuide/Models/recommendation.md) and [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/NeuralCFexample.scala)

**Python**

Build a NCF model for recommendation. 
```python
ncf=NeuralCF(user_count, item_count, class_num, user_embed=20, item_embed=20, hidden_layers=(40, 20, 10), include_mf=True, mf_embed=20)
```
Train a NCF model using BigDL Optimizer 
```python
optimizer = Optimizer(
    model=ncf,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=Adam(learningrate = 0.001, learningrate_decay=0.00005),
    end_trigger=MaxEpoch(10),
    batch_size=batch_size)
optimizer.optimize() 
```
Predict and recommend items(users) for users(items) with given features.
```python
userItemPairPrediction = ncf.predict_user_item_pair(valPairFeatureRdds)
userRecs = ncf.recommend_for_user(valPairFeatureRdds, 3)
itemRecs = ncf.recommend_for_item(valPairFeatureRdds, 3)
``` 
See more details in our [Recommender API](../APIGuide/Models/recommendation.md) and [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-ncf/ncf-explicit-feedback.ipynb).
