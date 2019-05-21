Analytics Zoo provides two Recommenders, including Wide and Deep (WND) model and Neural network-based Collaborative Filtering (NCF) model. Easy-to-use Keras-Style defined models which provides compile and fit methods for training. Alternatively, they could be fed into NNFrames or BigDL Optimizer.

Recommenders can handle models with either explict or implicit feedback, given corresponding features.

We also provide three user-friendly APIs to predict user item pairs, and recommend items (users) for users (items). See [here](#prediction-and-recommendation) for more details.

---
## Wide and Deep
Wide and Deep Learning Model, proposed by [Google, 2016](https://arxiv.org/pdf/1606.07792.pdf), is a DNN-Linear mixed model, which combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features (e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation.

After training the model, users can use the model to [do prediction and recommendation](#prediction-and-recommendation).

**Scala**
```scala
val wideAndDeep = WideAndDeep(modelType = "wide_n_deep", numClasses, columnInfo, hiddenLayers = Array(40, 20, 10))
```

* `modelType`: String. "wide", "deep", "wide_n_deep" are supported. Default is "wide_n_deep".
* `numClasses`: The number of classes. Positive integer.
* `columnInfo` An instance of [ColumnFeatureInfo](#columnfeatureinfo).
* `hiddenLayers`: Units of hidden layers for the deep model. Array of positive integers. Default is Array(40, 20, 10).

See [here](https://github.com/intel-analytics/zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/WideAndDeepExample.scala) for the Scala example that trains the WideAndDeep model on MovieLens 1M dataset and uses the model to do prediction and recommendation.


**Python**
```python
wide_and_deep = WideAndDeep(class_num, column_info, model_type="wide_n_deep", hidden_layers=(40, 20, 10))
```

* `class_num`: The number of classes. Positive int.
* `column_info`: An instance of [ColumnFeatureInfo](#columnfeatureinfo).
* `model_type`: String. 'wide', 'deep' and 'wide_n_deep' are supported. Default is 'wide_n_deep'.
* `hidden_layers`: Units of hidden layers for the deep model. Tuple of positive int. Default is (40, 20, 10).

See [here](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/recommendation/wide_n_deep.ipynb) for the Python notebook that trains the WideAndDeep model on MovieLens 1M dataset and uses the model to do prediction and recommendation.

---
## Neural network-based Collaborative Filtering
NCF ([He, 2015](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)) leverages a multi-layer perceptrons to learn the user–item interaction function. At the mean time, NCF can express and generalize matrix factorization under its framework. `includeMF`(Boolean) is provided for users to build a `NeuralCF` model with or without matrix factorization. 

After training the model, users can use the model to [do prediction and recommendation](#prediction-and-recommendation).

**Scala**
```scala
val ncf = NeuralCF(userCount, itemCount, numClasses, userEmbed = 20, itemEmbed = 20, hiddenLayers = Array(40, 20, 10), includeMF = true, mfEmbed = 20)
```

* `userCount`: The number of users. Positive integer.
* `itemCount`: The number of items. Positive integer.
* `numClasses`: The number of classes. Positive integer.
* `userEmbed`: Units of user embedding. Positive integer. Default is 20.
* `itemEmbed`: Units of item embedding. Positive integer. Default is 20.
* `hiddenLayers`: Units hiddenLayers for MLP. Array of positive integers. Default is Array(40, 20, 10).
* `includeMF`: Whether to include Matrix Factorization. Boolean. Default is true.
* `mfEmbed`: Units of matrix factorization embedding. Positive integer. Default is 20.

See [here](https://github.com/intel-analytics/analytics-zoo/blob/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation/NeuralCFexample.scala) for the Scala example that trains the NeuralCF model on MovieLens 1M dataset and uses the model to do prediction and recommendation.


**Python**
```python
ncf = NeuralCF(user_count, item_count, class_num, user_embed=20, item_embed=20, hidden_layers=(40, 20, 10), include_mf=True, mf_embed=20)
```

* `user_count`: The number of users. Positive int.
* `item_count`: The number of classes. Positive int.
* `class_num:` The number of classes. Positive int.
* `user_embed`: Units of user embedding. Positive int. Default is 20.
* `item_embed`: itemEmbed Units of item embedding. Positive int. Default is 20.
* `hidden_layers`: Units of hidden layers for MLP. Tuple of positive int. Default is (40, 20, 10).
* `include_mf`: Whether to include Matrix Factorization. Boolean. Default is True.
* `mf_embed`: Units of matrix factorization embedding. Positive int. Default is 20.

See [here](https://github.com/intel-analytics/analytics-zoo/blob/master/apps/recommendation/ncf-explicit-feedback.ipynb) for the Python notebook that trains the NeuralCF model on MovieLens 1M dataset and uses the model to do prediction and recommendation.


---
## Prediction and Recommendation

***Predict for user-item pairs***

Give prediction for each pair of user and item. Return RDD of [UserItemPrediction](#useritemprediction).

**Scala**
```scala
predictUserItemPair(featureRdd)
```

**Python**
```python
predict_user_item_pair(feature_rdd)
```

Parameters:

* `featureRdd`: RDD of [UserItemFeature](#useritemfeature).


***Recommend for users***

Recommend a number of items for each user. Return RDD of [UserItemPrediction](#useritemprediction).

**Scala**
```scala
recommendForUser(featureRdd, maxItems)
```

**Python**
```python
recommend_for_user(feature_rdd, max_items)
```

Parameters:

* `featureRdd`: RDD of [UserItemFeature](#useritemfeature).
* `maxItems`: The number of items to be recommended to each user. Positive integer.


***Recommend for items***

Recommend a number of users for each item. Return RDD of [UserItemPrediction](#useritemprediction).

**Scala**
```scala
recommendForItem(featureRdd, maxUsers)
```

**Python**
```python
recommend_for_item(feature_rdd, max_users)
```

Parameters:

* `featureRdd`: RDD of [UserItemFeature](#useritemfeature).
* `maxUsers`: The number of users to be recommended to each item. Positive integer.

---
## Model Save
After building and training a WideAndDeep or NeuralCF model, you can save it for future use.

**Scala**
```scala
wideAndDeep.saveModel(path, weightPath = null, overWrite = false)

ncf.saveModel(path, weightPath = null, overWrite = false)
```

* `path`: The path to save the model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path to save weights. Default is null.
* `overWrite`: Whether to overwrite the file if it already exists. Default is false.

**Python**
```python
wide_and_deep.save_model(path, weight_path=None, over_write=False)

ncf.save_model(path, weight_path=None, over_write=False)
```

* `path`: The path to save the model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path to save weights. Default is None.
* `over_write`: Whether to overwrite the file if it already exists. Default is False.

---
## Model Load
To load a WideAndDeep or NeuralCF model (with weights) saved [above](#model-save):

**Scala**
```scala
WideAndDeep.loadModel[Float](path, weightPath = null)

NeuralCF.loadModel[Float](path, weightPath = null)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path for pre-trained weights if any. Default is null.

**Python**
```python
WideAndDeep.load_model(path, weight_path=None)

NeuralCF.load_model(path, weight_path=None)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path for pre-trained weights if any. Default is None.

---
### UserItemFeature
Represent records of user-item with features.

Each record should contain the following fields:

* `userId`: Positive integer.
* `item_id`: Positive integer.
* `sample`: [Sample](https://bigdl-project.github.io/master/#APIGuide/Data/#sample) which consists of feature(s) and label(s).

**Scala**
```scala
UserItemFeature(userId, itemId, sample)
```

**Python**
```python
UserItemFeature(user_id, item_id, sample)
```

---
### UserItemPrediction
Represent the prediction results of user-item pairs.

Each prediction record will contain the following information:

* `userId`: Positive integer.
* `itemId`: Positive integer.
* `prediction`: The prediction (rating) for the user on the item.
* `probability`: The probability for the prediction.

**Scala**
```scala
UserItemPrediction(userId, itemId, prediction, probability)
```

**Python**
```python
UserItemPrediction(user_id, item_id, prediction, probability)
```

---
### ColumnFeatureInfo
An instance of `ColumnFeatureInfo` contains the same data information shared by the `WideAndDeep` model and its feature generation part.

You can choose to include the following information for feature engineering and the `WideAndDeep` model:

* `wideBaseCols`: Data of *wideBaseCols* together with *wideCrossCols* will be fed into the wide model.
* `wideBaseDims`: Dimensions of *wideBaseCols*. The dimensions of the data in *wideBaseCols* should be within the range of *wideBaseDims*.
* `wideCrossCols`: Data of *wideCrossCols* will be fed into the wide model.
* `wideCrossDims`: Dimensions of *wideCrossCols*. The dimensions of the data in *wideCrossCols* should be within the range of *wideCrossDims*.
* `indicatorCols`: Data of *indicatorCols* will be fed into the deep model as multi-hot vectors. 
* `indicatorDims`: Dimensions of *indicatorCols*. The dimensions of the data in *indicatorCols* should be within the range of *indicatorDims*.
* `embedCols`: Data of *embedCols* will be fed into the deep model as embeddings.
* `embedInDims`: Input dimension of the data in *embedCols*. The dimensions of the data in *embedCols* should be within the range of *embedInDims*.
* `embedOutDims`: The dimensions of embeddings for *embedCols*.
* `continuousCols`: Data of *continuousCols* will be treated as continuous values for the deep model.
* `label`: The name of the 'label' column. String. Default is "label".

__Remark:__

Fields that involve `Cols` should be an array of String (Scala) or a list of String (Python) indicating the name of the columns in the data.

Fields that involve `Dims` should be an array of integers (Scala) or a list of integers (Python) indicating the dimensions of the corresponding columns.

If any field is not specified, it will by default to be an empty array (Scala) or an empty list (Python).


**Scala**
```scala
ColumnFeatureInfo(
    wideBaseCols = Array[String](),
    wideBaseDims = Array[Int](),
    wideCrossCols = Array[String](),
    wideCrossDims = Array[Int](),
    indicatorCols = Array[String](),
    indicatorDims = Array[Int](),
    embedCols = Array[String](),
    embedInDims = Array[Int](),
    embedOutDims = Array[Int](),
    continuousCols = Array[String](),
    label = "label")
```

**Python**
```python
ColumnFeatureInfo(
    wide_base_cols=None,
    wide_base_dims=None,
    wide_cross_cols=None,
    wide_cross_dims=None,
    indicator_cols=None,
    indicator_dims=None,
    embed_cols=None,
    embed_in_dims=None,
    embed_out_dims=None,
    continuous_cols=None,
    label="label")
```