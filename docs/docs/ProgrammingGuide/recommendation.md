Analytics Zoo provides two Recommender models, including Wide and Deep(WND) learning model and Neural network-based Collaborative Filtering (NCF) model. 

**Highlights**

1. Easy-to-use Keras-Style defined models which provides compile and fit methods for training. Alternatively, they could be fed into NNFrames or BigDL Optimizer.
2. Recommenders can handle either explict or implicit feedback, given corresponding features.
3. It provides three user-friendly APIs to predict user item pairs, and recommend items (users) for users (items).

The examples/notebooks are included in the Analytics Zoo source code.

1. Wide and Deep Learning Model.
    [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/WideAndDeepExample.scala)
    [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-wide-n-deep/wide_n_deep.ipynb)
2. NCF.
    [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/NeuralCFexample.scala)
    [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-ncf/ncf-explicit-feedback.ipynb)
3. Session Recommender model.
    [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/SessionRecExp.scala)

---
## Wide and Deep

**Scala**

Build a WND model for recommendation. 
```scala
val wideAndDeep = WideAndDeep(modelType = "wide_n_deep", numClasses, columnInfo, hiddenLayers = Array(40, 20, 10))
```
Compile and train a WND model.
```scala
wideAndDeep.compile(optimizer = new Adam[Float](learningRate = 1e-2,learningRateDecay = 1e-5),
                    loss = SparseCategoricalCrossEntropy[Float](),
                    metrics = List(new Top1Accuracy[Float]()))
wideAndDeep.fit(trainRdds, batchSize, nbEpoch, validationRdds)
```
Predict and recommend items(users) for users(items) with given features.
```scala
val userItemPairPrediction = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 3)
val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 3)
``` 
See more details in our[Recommender API](../APIGuide/Models/recommendation.md) and [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/WideAndDeepExample.scala).

**Python**

Compile and train a WND model.
```python
wide_n_deep = WideAndDeep(class_num, column_info, model_type="wide_n_deep", hidden_layers=(40, 20, 10))
```
Train a WND model using BigDL Optimizer 
```python
wide_n_deep.compile(optimizer= Adam(learningrate = 1e-3, learningrate_decay=1e-6),
                    loss= "sparse_categorical_crossentropy",
                    metrics=['accuracy'])
wide_n_deep.fit(train_rdd, nb_epoch, batch_size, val_rdd)
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
Compile and train a NCF model
```scala
ncf.compile(optimizer = new Adam[Float](learningRate = 1e-2,learningRateDecay = 1e-5),
            loss = SparseCategoricalCrossEntropy[Float](),
            metrics = List(new Top1Accuracy[Float]()))
ncf.fit(trainRdds, batchSize, nbEpoch, validationRdds)
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
Compile and train a NCF model
```python
ncf.compile(optimizer= Adam(learningrate = 1e-3, learningrate_decay=1e-6),
            loss= "sparse_categorical_crossentropy",
            metrics=['accuracy'])
ncf.fit(train_rdd, nb_epoch, batch_size, val_rdd)
```
Predict and recommend items(users) for users(items) with given features.
```python
userItemPairPrediction = ncf.predict_user_item_pair(valPairFeatureRdds)
userRecs = ncf.recommend_for_user(valPairFeatureRdds, 3)
itemRecs = ncf.recommend_for_item(valPairFeatureRdds, 3)
``` 
See more details in our [Recommender API](../APIGuide/Models/recommendation.md) and [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-ncf/ncf-explicit-feedback.ipynb).

---
## Session Recommender Model

**Scala**

Build a Session Recommender model for recommendation. 
```scala
val sessionRecommender = SessionRecommender(itemCount, itemEmbed, sessionLength, includeHistory, mlpHiddenLayers, historyLength)
                                    
Compile and train a Session Recommender model
```scala
sessionRecommender.compile(optimizer = new RMSprop[Float](learningRate = 1e-2,learningRateDecay = 1e-5),
                           loss = SparseCategoricalCrossEntropy[Float](),
                           metrics = List(new Top1Accuracy[Float]()))
sessionRecommender.fit(trainRdds, batchSize, nbEpoch, validationRdds)
```
Predict and recommend items(users) for users(items) with given features.
```scala
val results = sessionRecommender.predict(testRdd)
val recommendations = model.recommendForSession(testRdd, 3, false)
``` 
See more details in our[Recommender API](../APIGuide/Models/recommendation.md) and [Scala example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/SessionRecExp.scala)

**Python**

Build a Session Recommender model for recommendation. 
```python
session_recommender=SessionRecommender(item_count, item_embed, rnn_hidden_layers=[40, 20], session_length=10, include_history=True, mlp_hidden_layers=[40, 20], history_length=5)
```
Compile and train a NCF model
```python
session_recommender.compile(optimizer= RMSprop(learningrate = 1e-3, learningrate_decay=1e-6),
                            loss= "sparse_categorical_crossentropy",
                            metrics=['top5Accuracy'])
session_recommender.fit(train, batch_size=4, nb_epoch=1, validation_data=test)
```
Predict and recommend items(users) for users(items) with given features.
```python
results1 = session_recommender.predict(test)
recommendations1 = session_recommender.recommend_for_session(rdd, 3, zero_based_label=False)
``` 